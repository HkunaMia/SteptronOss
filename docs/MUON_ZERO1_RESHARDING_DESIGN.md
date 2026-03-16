# Muon ZeRO-1 Resharding 设计文档

## 1. 概述

### 1.1 背景与问题定义

在大规模语言模型训练中，**Muon 优化器**与 **ZeRO-1** 存在根本性的兼容冲突：

| 组件 | 需求 | 冲突点 |
|------|------|--------|
| **Muon 优化器** | 需要完整的 2D 梯度矩阵进行 Newton-Schulz 正交化 | 无法处理分片梯度 |
| **ZeRO-1** | 通过 reduce-scatter 将梯度分片到不同 DP rank | 每个 rank 只有 1/N 的梯度 |

**传统解决方案（Megatron-LM）**：
- 使用 All-Reduce 替代 Reduce-Scatter 来恢复完整梯度
- **代价**：通信量几乎翻倍（~2×）

### 1.2 核心创新

StepTronOSS 采用 **Rank-Major 参数分配策略**，实现：
- **单次 Reduce-Scatter** 即可让每个 rank 获得其负责参数的**完整梯度**
- **通信量保持 1×**（vs All-Reduce 的 2×）
- **混合策略**：仅对专家参数应用此优化，非专家参数使用标准 DP All-Reduce

### 1.3 性能收益

根据论文数据：
- **端到端迭代时间减少**：~5%
- **额外内存开销**：< 4GB（来自 padding）

---

## 2. 架构与关键组件

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Muon ZeRO-1 Resharding                          │
├─────────────────────────────────────────────────────────────────────────┤
│  参数分桶层 (ParamBucketKey)                                             │
│  ├── dtype: torch.dtype (bf16/fp32)                                     │
│  ├── allreduce_group: "DP" | "EDP"                                      │
│  └── signature: "MUON:True" | "MUON:False"                              │
├─────────────────────────────────────────────────────────────────────────┤
│  参数分配层 (balanced_list_split)                                        │
│  ├── 输入：参数列表 + 大小列表                                           │
│  ├── 策略：贪心算法（从大到小分配）                                       │
│  └── 保证：每个参数完整分配给单个 rank                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  梯度缓冲区层 (GradBuffer)                                               │
│  ├── 连续内存分配（FP32）                                                │
│  ├── 双视图机制（FP32/BF16 共享存储）                                     │
│  └── Padding 到 max_split_size                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  通信层 (Zero1GradientManager)                                           │
│  ├── Reduce-Scatter：单次通信获得完整梯度                                 │
│  └── All-Gather：同步更新后的参数                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

#### 2.2.1 ParamBucketKey（多维度分桶键）

```python
# steptronoss/model/utils/comm_buffer.py:24-39
@dataclass(frozen=True)
class ParamBucketKey:
    """
    Composite identifier for bucketing parameters.
    
    The legacy implementation keyed everything by dtype, which breaks for Muon
    because multiple logical parameter buckets can share the same dtype.
    """
    dtype: torch.dtype              # 数据类型（bf16/fp32）
    allreduce_group: str = "DP"     # 通信组（DP/EDP）
    signature: int | str | None = None  # 额外标识（如 Muon 分组）
```

**分桶逻辑**：
```python
# steptronoss/model/utils/comm_buffer.py:68-82
def get_bucket_key_from_param_attrs(param: SteptronParameter):
    if getattr(param, "expert_model_parallel", False):
        allreduce_group = "EDP"   # 专家参数 → EDP 桶
    else:
        allreduce_group = "DP"    # 非专家参数 → DP 桶

    # 关键：将 Muon 参数和非 Muon 参数分到不同桶
    signature = f"MUON:{getattr(param, 'is_muon_param', False)}"
    
    return ParamBucketKey(
        dtype=param.dtype,
        allreduce_group=allreduce_group,
        signature=signature,
    )
```

#### 2.2.2 balanced_list_split（贪心分配算法）

```python
# steptronoss/utils/general.py:129-157
def balanced_list_split(data: list[T], sizes: list[int], split: int) -> list[list[T]]:
    """
    Greedy, order-agnostic split of `data` into `split` buckets.
    
    Items are processed from largest to smallest `size` and placed in the bucket
    whose score is minimal at that step.
    
    关键保证：每个参数完整地分配给某一个 bucket (DP rank)
    """
    import numpy as np

    new_data: list[list[T]] = [[] for i in range(split)]
    new_size = np.zeros(split, dtype=int)
    new_cout = np.zeros(split, dtype=int)
    
    total_sizes = sum(sizes) if number_balance else 0
    
    # 从大到小排序，贪心分配给当前负载最小的 rank
    for i in np.argsort(sizes)[::-1]:
        dst = (total_sizes * new_cout + new_size).argmin()
        new_data[dst].append(data[i])  # 整个参数分配给该 rank
        new_size[dst] += sizes[i]
        new_cout[dst] += 1
    
    return new_data
```

#### 2.2.3 Zero1GradientManager（梯度管理器）

```python
# steptronoss/optimizer/zero1_gradient_manager.py:164-200
@torch.no_grad()
def _reduce_model_grads(self):
    """Reduce gradients across data parallel ranks."""
    
    # 1. 处理序列并行和 micro-dp 的 all-reduce
    self._custom_allreduce(tag="sequence_parallel", group=PM.group_of("TP"))
    self._custom_allreduce(tag="micro_dp", group=PM.group_of("TP"))
    
    # 2. 处理专家并行梯度（拓扑感知缩放）
    self._process_expert_parallel_grads()
    
    # 3. 关键：单次 Reduce-Scatter 获得完整梯度
    with get_timers().record("grads-reduce-scatter", log_level=2):
        for bucket_key, buffer in self._grad_buffers.items():
            reduce_group = bucket_key.allreduce_group
            world_size = PM.size_of(reduce_group)
            group = PM.group_of(reduce_group)
            rank = PM.rank_in(reduce_group)
            local_size = buffer.numel() // world_size

            buffer /= world_size
            torch.distributed.reduce_scatter_tensor(
                output=buffer[rank * local_size : rank * local_size + local_size],
                input=buffer,
                group=group,
            )
```

---

## 3. 内存布局与数据流

### 3.1 内存布局对比

#### 标准 ZeRO-1（Megatron-LM）
```
参数梯度缓冲区 (4 DP ranks):
┌─────────────────────────────────────────────────────────────────┐
│  param1[0:1/4] │ param1[1/4:2/4] │ param1[2/4:3/4] │ param1[3/4:1] │
│  (DP0)         │  (DP1)          │  (DP2)          │  (DP3)        │
└─────────────────────────────────────────────────────────────────┘
After Reduce-Scatter:
DP0: [param1_sum[0:1/4]]  ← 只有 1/4，无法做 Newton-Schulz
```

#### StepTronOSS Rank-Major
```
参数梯度缓冲区 (4 DP ranks):
┌─────────────────────────────────────────────────────────────────┐
│  param1 (完整) │  param3 (完整)  │  padding  │  param2 (完整)  │
│  [4096×4096]   │  [4096×4096]    │           │  [4096×4096]    │
│  (DP0)         │  (DP0)          │           │  (DP1)          │
└─────────────────────────────────────────────────────────────────┘
After Reduce-Scatter:
DP0: [param1_grad_sum (完整), param3_grad_sum (完整)]  ← 完整梯度！
```

### 3.2 数据流图

```
Forward:
  Input → Model (BF16) → Output
              ↑
              └── 每个 rank 有完整参数（无 All-Gather）

Backward:
  GradOutput → Backward Pass → GradInput
                                      ↓
                              param.main_grad (FP32)
                                      ↓
                              ┌───────────────┐
                              │  Gradient Hook │
                              │  (BF16→FP32)  │
                              └───────────────┘

Optimizer Step:
  ┌─────────────────────────────────────────────────────────┐
  │  1. _reduce_model_grads()                               │
  │     ├── TP All-Reduce (sequence_parallel, micro_dp)     │
  │     ├── Expert Parallel 缩放 (tp_size/ep_size)          │
  │     └── Reduce-Scatter (单次通信获得完整梯度)            │
  │                                                         │
  │  2. _copy_model_grads_to_fp32_params()                  │
  │     └── main_grad → param.grad                          │
  │                                                         │
  │  3. optimizer.step()                                    │
  │     └── Muon: Newton-Schulz 正交化（需要完整梯度）        │
  │                                                         │
  │  4. _gather_model_params()                              │
  │     └── All-Gather 同步更新后的参数                      │
  └─────────────────────────────────────────────────────────┘
```

---

## 4. 关键算法

### 4.1 Newton-Schulz 迭代

Muon 优化器使用 Newton-Schulz 迭代进行梯度正交化：

```python
# steptronoss/optimizer/_helpers.py:66-98
def zeropower(G, steps, run_ns_in_fp16, coeffs):
    """
    Zero-power orthogonalization via Newton-Schulz iteration.
    
    数学公式：
    G₀ = G / ||G||
    Gₖ₊₁ = a·Gₖ + b·Gₖ·Gₖᵀ·Gₖ + c·Gₖ·Gₖᵀ·Gₖ·Gₖᵀ·Gₖ
    
    其中 a, b, c 是预计算的系数，确保收敛到正交矩阵。
    """
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    if run_ns_in_fp16:
        X = X.to(torch.float16)

    for a, b, c in coeffs:
        A = X @ X.mT
        if X.dim() == 3:
            B = torch.baddbmm(A, A, A, alpha=c, beta=b)
            X = torch.baddbmm(X, B, X, beta=a)
        elif X.dim() == 2:
            B = torch.addmm(A, A, A, alpha=c, beta=b)
            X = torch.addmm(X, B, X, beta=a)

    if G.size(-2) > G.size(-1):
        X = X.mT.contiguous()

    return X
```

**关键要求**：`G @ G.mT` 需要完整的矩阵 `G`，无法处理分片梯度。

### 4.2 拓扑感知缩放

在混合并行（TP+EP+DP）场景下，专家参数需要特殊缩放：

```python
# steptronoss/optimizer/base_gradient_manager.py:180-200
def _process_expert_parallel_grads(self) -> None:
    """
    场景对比：
    - Dense 训练：TP=1, DP=4，每张卡处理 8 tokens，共 32 tokens
      梯度需要除以 4（world_size）
    
    - MoE 训练：TP=1, EP=4, EDP=2
      - 每张卡处理 16 tokens（因为 EP，只处理路由到本地专家的 token）
      - 但 EDP 组只有 2 个 rank
      如果不缩放，梯度会是 Dense 的 2 倍！
    """
    tp_size = PM.size_of("TP")
    ep_size = PM.size_of("EP")
    
    if ep_size <= 1:
        return
        
    for bucket_key, gbuf in self._grad_buffers.items():
        if bucket_key.allreduce_group == "EDP":
            # 关键：对 EDP 桶进行 TP/EP 缩放
            gbuf.data *= tp_size / ep_size  # 例如 1/8
```

---

## 5. 与并行维度的协同

### 5.1 并行维度定义

StepTronOSS 支持以下并行维度：

| 维度 | 缩写 | 说明 |
|------|------|------|
| Tensor Parallel | TP | 张量并行，切分 Attention/FFN |
| Pipeline Parallel | PP | 流水线并行，切分层 |
| Data Parallel | DP | 数据并行，非专家参数 |
| Expert Parallel | EP | 专家并行，切分 MoE 专家 |
| Expert Data Parallel | EDP | 专家数据并行，专家参数梯度同步 |
| Expert Tensor Parallel | ETP | 专家张量并行 |

### 5.2 通信组关系

```
配置示例：8 GPUs, TP=2, PP=2, EP=2, EDP=2

GPU 拓扑：
┌─────────┬─────────┐
│ GPU 0   │ GPU 1   │  PP Stage 0
│ (TP=0)  │ (TP=1)  │
├─────────┼─────────┤
│ GPU 2   │ GPU 3   │  PP Stage 0
│ (TP=0)  │ (TP=1)  │
├─────────┼─────────┤
│ GPU 4   │ GPU 5   │  PP Stage 1
│ (TP=0)  │ (TP=1)  │
├─────────┼─────────┤
│ GPU 6   │ GPU 7   │  PP Stage 1
│ (TP=0)  │ (TP=1)  │
└─────────┴─────────┘

通信组：
- TP Group 0: [0, 1], [2, 3], [4, 5], [6, 7]
- PP Group: [0, 2, 4, 6], [1, 3, 5, 7]
- EP Group: [0, 1, 2, 3], [4, 5, 6, 7]
- EDP Group: [0, 4], [1, 5], [2, 6], [3, 7]
- DP Group: [0, 2], [1, 3], [4, 6], [5, 7]

参数分配：
- 非专家参数（Attention）：在 DP 组内同步
- 专家参数（MoE）：在 EDP 组内同步
```

### 5.3 与 TP 的协同

```python
# steptronoss/optimizer/muon.py:100-128
for p in group["params"]:
    # merge_op 用于处理 TP 切分的参数
    merge_op: ReshapeOp = getattr(p, "merge_op", None)
    if merge_op is None:
        merge_op = Identity()

    g: torch.Tensor = p.grad
    
    # 更新 momentum
    buf.mul_(momentum).add_(g)
    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
    
    # 关键：通过 merge_op 恢复完整梯度
    merged_g = merge_op.forward({"grad": g})
    for k, v in list(merged_g.items()):
        # 现在 v 是完整梯度，可以做 Newton-Schulz
        adjusted_lr = lr * adjust_ratio_for_muon(matched_adamw_rms, v.shape)
        merged_g[k] = self.newtonschulz_fn(v, steps=ns_steps) * -adjusted_lr

    # 通过 merge_op 的 backward 回到分片形式
    update = merge_op.backward(merged_g)["grad"]
```

---

## 6. 实际配置示例

### 6.1 Muon 配置

```python
# steptronoss/exp/optimizer.py:65-127
class MuonConfig(OptimizerConfig):
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")
    
    # Muon 特定参数
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_matched_adamw_rms: float = 0.2
    muon_ns_steps: int = 5
    muon_run_ns_in_fp16: bool = True
    muon_newtonschulz_fn: str = "default"  # 或 "polar_express"
    
    # 参数选择
    muon_exclude_embeddings: bool = True
    muon_exclude_names: tuple[str, ...] = ()

    def mark_muon_params(self, model: Module) -> NoReturn:
        """标记哪些参数使用 Muon（ndim >= 2 的 2D 参数）"""
        for name, param in model.named_parameters():
            if param.ndim == 2:
                param.is_muon_param = True
            else:
                param.is_muon_param = False
```

### 6.2 Step3.5 Muon 配置（实际使用）

```python
# playground/sft/step3/muon_optimizer.py
from steptronoss.exp.optimizer import MuonConfig


class Step3p5MuonConfig(MuonConfig):
    def __init__(self):
        super().__init__()
        self.weight_decay_on_1d_params = True
        self.muon_ns_steps = 6
        self.muon_newtonschulz_fn = "polar_express"

    def mark_muon_params(self, model) -> None:
        """Step3.5 特定的 Muon 参数标记逻辑"""
        # 排除 embedding
        embedding_params = set()
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                embedding_params.add(module.weight)
        
        for name, param in model.named_parameters():
            if param in embedding_params:
                param.is_muon_param = False
            elif param.ndim >= 2:  # 2D 及以上参数使用 Muon
                param.is_muon_param = True
            else:
                param.is_muon_param = False

    def mark_gather_ops(self, model) -> None:
        """为 TP 切分参数附加 merge_op"""
        # 处理 GQA、FFN、MoE 等特殊结构的 merge_op
        # ...
```

### 6.3 实验配置示例

```python
# playground/sft/step3/step3_toy_sft_step3_data_muon.py
from playground.sft.step3.muon_optimizer import Step3p5MuonConfig

class Exp(BaseExp):
    optimizer_cfg = Step3p5MuonConfig  # 使用 Muon 优化器
    
    def __init__(self):
        super().__init__()
        self.trainer_cfg.global_batch_size = 32
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 2
        self.model_cfg.parallel_cfg.pipeline_model_parallel_size = 1
```

---

## 7. 适用场景与局限性

### 7.1 最适合的场景

| 维度 | 最佳配置 | 原因 |
|------|----------|------|
| **模型类型** | MoE 模型 | 专家参数多且均匀，padding 开销小 |
| **模型规模** | > 10B 参数 | 大模型更需要 ZeRO-1 内存节省 |
| **优化器** | Muon | 需要完整梯度进行 Newton-Schulz |
| **DP size** | ≤ 16 | 避免 padding 开销激增 |
| **并行配置** | TP + EP + DP | 专家参数走 EDP，非专家走 DP |

### 7.2 不适用场景

| 场景 | 原因 | 推荐方案 |
|------|------|----------|
| **DP > 64** | padding 开销随 DP size 增长 | 仅对专家参数使用，或改用 FSDP |
| **参数极度不均衡** | 如超大 vocab embedding，padding 严重 | 标准 ZeRO-1 |
| **纯 Dense 模型** | 无专家参数，无需特殊处理 | 标准 ZeRO-1 |
| **不使用 Muon** | 无需完整梯度 | 标准 ZeRO-1 |

### 7.3 性能对比

| 方案 | 通信量 | 内存开销 | 适用场景 |
|------|--------|----------|----------|
| **Megatron Naive** | ~2× | 高 | 简单实现 |
| **标准 ZeRO-1** | 1× | 低 | 普通参数 |
| **StepTronOSS** | ~1× | <4GB padding | Muon + MoE |

---

## 8. 代码文件索引

| 组件 | 文件路径 | 关键函数/类 |
|------|----------|-------------|
| **分桶键定义** | `steptronoss/model/utils/comm_buffer.py` | `ParamBucketKey`, `get_bucket_key_from_param_attrs()` |
| **缓冲区构建** | `steptronoss/model/utils/comm_buffer.py` | `build_grad_buffers()` |
| **平衡分配** | `steptronoss/utils/general.py` | `balanced_list_split()` |
| **ZeRO-1 梯度管理** | `steptronoss/optimizer/zero1_gradient_manager.py` | `Zero1GradientManager`, `_reduce_model_grads()` |
| **专家梯度处理** | `steptronoss/optimizer/base_gradient_manager.py` | `_process_expert_parallel_grads()` |
| **Muon 优化器** | `steptronoss/optimizer/muon.py` | `Muon`, `step()` |
| **Newton-Schulz** | `steptronoss/optimizer/_helpers.py` | `zeropower()`, `ZEROPOWER_COEFFS` |
| **Muon 配置** | `steptronoss/exp/optimizer.py` | `MuonConfig`, `mark_muon_params()` |
| **Step3.5 Muon 配置** | `playground/sft/step3/muon_optimizer.py` | `Step3p5MuonConfig` |
| **并行状态管理** | `steptronoss/core/parallel_state.py` | `ParallelManager`, `PM` |

---

## 9. 总结

**Muon ZeRO-1 Resharding** 是为大规模 MoE 模型 + Muon 优化器量身定制的优化方案：

1. **核心创新**：通过 Rank-Major 参数分配，单次 Reduce-Scatter 获得完整梯度
2. **性能收益**：~5% 迭代时间减少，< 4GB 额外内存
3. **关键限制**：DP size 不宜过大，参数应相对均匀
4. **最佳实践**：仅对专家参数应用此优化，非专家参数使用标准 DP All-Reduce

该优化在 Step3.5 等大规模 MoE 模型的 SFT 训练中得到实际应用。
