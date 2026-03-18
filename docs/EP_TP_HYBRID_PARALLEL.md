# EP+TP 混合并行策略详解

> 本文档深入讲解 StepTronOSS 中 MoE 模型的 Expert Parallel (EP) 与 Tensor Parallel (TP) 混合并行策略。

---

## 1. 基本概念

### 1.1 并行维度定义

StepTronOSS 中 MoE 模型涉及以下并行维度：

| 缩写 | 全称 | 作用范围 | 关键操作 |
|------|------|---------|---------|
| **TP** | Tensor Parallel | Attention 层 | All-Reduce |
| **EP** | Expert Parallel | Expert 分发 | All-to-All |
| **ETP** | Expert Tensor Parallel | 单个 Expert 内部 | All-Reduce |
| **PP** | Pipeline Parallel | 层间切分 | P2P 通信 |
| **EDP** | Expert Data Parallel | Expert 数据并行 | All-Reduce |

### 1.2 为什么需要 EP+TP 混合？

**纯 TP 的问题**:
- 所有 GPU 存储所有专家，显存无法扩展
- 每卡计算所有专家，无法利用专家稀疏性

**纯 EP 的问题**:
- 当单个专家参数量太大时，单卡放不下
- 需要更细粒度的切分

**EP+TP 混合优势**:
```
假设: 256 专家, 每专家 1B 参数量, 使用 64 GPUs

纯 EP: 每卡 256/64=4 专家 → 4B 参数量/卡 ✓
       但如果每专家 10B → 40B/卡 ✗

EP+TP: EP=8, ETP=8 
       每卡专家数: 256/8=32 专家的 1/8 → 4B/卡 ✓
       可以扩展到任意大小的专家
```

---

## 2. 并行策略配置

### 2.1 配置类定义

**文件**: `steptronoss/exp/base_exp.py:308`

```python
class ParallelConfig(AbstractParallelConfig):
    # 并行维度定义 (使用 einops 风格的语法)
    parallel_definition: dict[str, str] = {
        "TP":  "(p d t) -> (p d) t",           # Attention TP
        "EP":  "(p edp ep etp) -> (p edp etp) ep",  # Expert Parallel
        "ETP": "(p edp ep etp) -> (p edp ep) etp",  # Expert TP
        "EDP": "(p edp ep etp) -> (p ep etp) edp",  # Expert DP
    }
    
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = Ref(".tensor_model_parallel_size")
```

### 2.2 配置约束检查

```python
def sanity_check(self):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    # Attention 部分的并行规模
    attn_model_parallel_size = (
        PP * TP * CP
    )
    
    # MoE 部分的并行规模
    moe_model_parallel_size = (
        PP * ETP * EP
    )
    
    # 约束条件
    assert world_size % attn_model_parallel_size == 0
    assert world_size % moe_model_parallel_size == 0
```

### 2.3 典型配置示例

```python
# 场景 1: 8 GPUs, 小专家 (不需要 ETP)
parallel_cfg.tensor_model_parallel_size = 2      # TP=2
parallel_cfg.expert_model_parallel_size = 4      # EP=4
parallel_cfg.expert_tensor_parallel_size = 1     # ETP=1
# 验证: 2*4*1 = 8 ✓

# 场景 2: 64 GPUs, 大专家 (需要 ETP)
parallel_cfg.tensor_model_parallel_size = 4      # TP=4
parallel_cfg.expert_model_parallel_size = 8      # EP=8
parallel_cfg.expert_tensor_parallel_size = 2     # ETP=2
# 验证: 4*8*2 = 64 ✓

# 场景 3: 128 GPUs, 超大规模
parallel_cfg.tensor_model_parallel_size = 4      # TP=4
parallel_cfg.pipeline_model_parallel_size = 2    # PP=2
parallel_cfg.expert_model_parallel_size = 8      # EP=8
parallel_cfg.expert_tensor_parallel_size = 2     # ETP=2
# 验证: Attention=2*4=8, MoE=2*8*2=32, 128%8=0, 128%32=0 ✓
```

---

## 3. EP 与 ETP 的实现机制

### 3.1 专家参数切分

**文件**: `steptronoss/model/common/moe_block.py:125`

```python
class GroupedExperts(nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id=0):
        # EP: 将专家切分到不同 GPU
        self.num_global_experts = cfg.moe_num_experts
        self.num_local_experts = cfg.moe_num_experts // PM.size_of("EP")
        
        # ETP: 将单个专家切分到不同 GPU
        self.w1 = torch.nn.Parameter(
            torch.empty(
                size=(
                    cfg.moe_num_experts // PM.size_of("EP"),     # EP 切分
                    cfg.moe_hidden_size * 2 // PM.size_of("ETP"), # ETP 切分
                    cfg.hidden_size,
                ),
                ...
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                size=(
                    cfg.moe_num_experts // PM.size_of("EP"),     # EP 切分
                    cfg.hidden_size,
                    cfg.moe_hidden_size // PM.size_of("ETP"),     # ETP 切分
                ),
                ...
            )
        )
        
        # 标记为 MoE 参数，用于梯度处理
        self.w1.expert_model_parallel = True
        self.w2.expert_model_parallel = True
```

**参数分布示意** (EP=4, ETP=2, 共 8 个专家):

```
全局专家: [E0, E1, E2, E3, E4, E5, E6, E7]

GPU 0 (EP=0, ETP=0): E0[:], E1[:]  (w1, w2 的 ETP=0 部分)
GPU 1 (EP=0, ETP=1): E0[:], E1[:]  (w1, w2 的 ETP=1 部分)
GPU 2 (EP=1, ETP=0): E2[:], E3[:]  (w1, w2 的 ETP=0 部分)
GPU 3 (EP=1, ETP=1): E2[:], E3[:]  (w1, w2 的 ETP=1 部分)
...
```

### 3.2 两种前向传播模式

MoE 层根据 EP 大小自动选择传播模式：

**文件**: `steptronoss/model/common/moe_block.py:376`

```python
def forward(self, x: torch.FloatTensor, recompute: bool = False):
    # 1. Router 计算 (所有 GPU 都有完整 router)
    logits = self.gate(x)
    token_expert_ids, token_weights, aux_loss = self.forward_router(logits)
    
    # 2. 根据并行策略选择传播模式
    if PM.size_of("EP") > 1:
        # EP > 1: 使用 all-to-all 分发 token
        assert PM.size_of("ETP") == 1  # DeepEP 目前不支持 ETP
        output = self.forward_experts_ep(x, token_expert_ids, token_weights)
    else:
        # EP = 1: 使用 TP 方式
        output = self.forward_experts_tp(x, token_expert_ids, token_weights)
```

### 3.3 EP 模式 (forward_experts_ep)

**文件**: `steptronoss/model/common/moe_block.py:358`

```python
def forward_experts_ep(self, x, token_expert_ids, token_weights):
    # 1. Token Dispatch: 将 token 发送到目标专家所在的 GPU
    #    使用 all-to-all 通信
    x, token_expert_ids, token_weights = self.dispatcher.dispatch(
        x, token_expert_ids, token_weights
    )
    
    # 2. Local Expert Compute: 只计算分配给本地的专家
    x = self.experts(x, token_expert_ids, token_weights)
    
    # 3. Token Combine: 将计算结果传回原 GPU
    #    使用 all-to-all 通信
    x = self.dispatcher.combine(x)
    
    return x
```

**通信流程**:

```
Step 1: Dispatch (All-to-All)
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ GPU 0   │      │ GPU 1   │      │ GPU 2   │      │ GPU 3   │
│ E0, E1  │      │ E2, E3  │      │ E4, E5  │      │ E6, E7  │
│ [T0→E0] │  ──► │ [T0→E0] │      │         │      │         │
│ [T1→E2] │      │ [T1→E2] │ ──►  │         │      │         │
│ [T2→E4] │      │         │      │ [T2→E4] │      │         │
└─────────┘      └─────────┘      └─────────┘      └─────────┘

Step 2: Expert Compute (Local)
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ Compute │      │ Compute │      │ Compute │      │ Compute │
│ E0, E1  │      │ E2, E3  │      │ E4, E5  │      │ E6, E7  │
└─────────┘      └─────────┘      └─────────┘      └─────────┘

Step 3: Combine (All-to-All)
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ [T0]    │  ◄── │ [T0]    │      │         │      │         │
│ [T1]    │      │ [T1]    │ ◄──  │         │      │         │
│ [T2]    │      │         │      │ [T2]    │ ◄──  │         │
└─────────┘      └─────────┘      └─────────┘      └─────────┘
```

### 3.4 TP 模式 (forward_experts_tp)

**文件**: `steptronoss/model/common/moe_block.py:339`

```python
def forward_experts_tp(self, x, token_expert_ids, token_weights):
    # 1. 如果使用 Sequence Parallel，先 gather
    if self.sequence_parallel:
        x = gather_from_sequence_parallel_region(x, "ETP")
        token_expert_ids = gather_from_sequence_parallel_region(token_expert_ids, "ETP")
        token_weights = gather_from_sequence_parallel_region(token_weights, "ETP")
    
    # 2. 计算 (每个 GPU 计算部分专家，然后 all-reduce)
    x = self.experts(x, token_expert_ids, token_weights)
    
    # 3. 如果需要 Sequence Parallel，再 slice
    if self.sequence_parallel:
        x = slice_to_sequence_parallel_region(x, group="ETP")
    
    return x
```

**注意**: TP 模式下所有 GPU 都有所有专家的部分参数，通过 all-reduce 聚合结果。

---

## 4. ETP 的 All-Reduce 机制

### 4.1 在 GroupedExperts 中的实现

**文件**: `steptronoss/model/common/moe_block.py:168`

```python
def forward(self, x: torch.FloatTensor, token_expert_ids, token_weights):
    # routed_grouped_ffn: 执行 Grouped GEMM 计算
    x = routed_grouped_ffn(
        self.w1, self.w2, self.activation,
        x, token_expert_ids, token_weights
    )
    
    # 关键: 对 ETP 组进行 all-reduce，聚合结果
    x = reduce_from_tensor_model_parallel_region(x, group="ETP")
    return x
```

### 4.2 Tensor Parallel 层对 ETP 的支持

**文件**: `steptronoss/core/tensor_parallel/layers.py:701`

```python
class ColumnParallelLinear(MegatronModule):
    def __init__(self, ..., use_moe=False):
        # 自动选择 TP 或 ETP
        tp_world_size = PM.size_of("ETP") if use_moe else PM.size_of("TP")
        self.output_size_per_partition = safediv(output_size, tp_world_size)
```

### 4.3 Sequence Parallel 支持

```python
# 在 forward_experts_tp 中支持 ETP 的 Sequence Parallel
if self.sequence_parallel:
    # Gather 完整的序列
    x = gather_from_sequence_parallel_region(x, "ETP")
    
# 计算...

if self.sequence_parallel:
    # Slice 回并行分布
    x = slice_to_sequence_parallel_region(x, group="ETP")
```

---

## 5. 随机种子与初始化

### 5.1 ETP 独立的随机种子

**文件**: `steptronoss/initialize.py:78`

```python
def set_mpu_random_seed(seed_):
    # TP 种子 (用于 Attention)
    tp_rank_specific_seed = offset + PM.rank_in("TP")
    
    # ETP 种子 (用于 Expert)
    # 每个 (EP, ETP) 对需要独立的种子
    expert_offset = seed + 3718
    etp_rank_specific_seed = expert_offset + PM.rank_in("ETP") + PM.rank_in("EP") * PM.size_of("ETP")
    
    # 注册到 RNG tracker
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tp_rank_specific_seed)
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_MODEL_PARALLEL_RNG_TRACKER_NAME, etp_rank_specific_seed)
```

**为什么需要独立种子**:
- 不同 (EP, ETP) 位置的 Expert 应该独立初始化
- 但相同 ETP 位置的 Expert 参数应该相同 (用于 TP all-reduce)

---

## 6. 梯度处理

### 6.1 MoE 参数的梯度归约

**文件**: `steptronoss/optimizer/clip_grads.py:131`

```python
# 判断是否为 MoE 参数
is_moe_param = getattr(param, "expert_model_parallel", False)

if is_moe_param:
    # MoE 参数: 只在 EP 组内去重
    is_not_duplicate = param_is_not_expert_parallel_duplicate(param)
else:
    # Attention 参数: 只在 TP 组内去重
    is_not_duplicate = param_is_not_tensor_parallel_duplicate(param)
```

### 6.2 梯度累积与 All-Reduce

```python
# ETP 梯度: 在 ETP 组内 all-reduce
dist.all_reduce(grad, group=PM.group_of("ETP"))

# EP 梯度: 在 EDP (Expert DP) 组内 all-reduce
dist.all_reduce(grad, group=PM.group_of("EDP"))
```

---

## 7. 配置最佳实践

### 7.1 小规模 (8 GPUs)

```python
# 适用于 Qwen3-1.7B MoE
parallel_cfg.tensor_model_parallel_size = 2
parallel_cfg.expert_model_parallel_size = 4
parallel_cfg.expert_tensor_parallel_size = 1

# 专家分布:
# - EP=4: 每卡 1/4 专家
# - ETP=1: 专家不切分
# 适合: 单专家参数量较小 (几百 MB)
```

### 7.2 中规模 (64 GPUs)

```python
# 适用于 Step3.5-Flash
parallel_cfg.tensor_model_parallel_size = 4
parallel_cfg.expert_model_parallel_size = 8
parallel_cfg.expert_tensor_parallel_size = 2

# 专家分布:
# - EP=8: 每卡 1/8 专家
# - ETP=2: 每个专家切分为 2 份
# 适合: 单专家参数量较大 (几 GB)
```

### 7.3 大规模 (256+ GPUs)

```python
# 适用于超大模型
parallel_cfg.tensor_model_parallel_size = 8
parallel_cfg.pipeline_model_parallel_size = 2
parallel_cfg.expert_model_parallel_size = 16
parallel_cfg.expert_tensor_parallel_size = 2

# 专家分布:
# - EP=16: 每卡 1/16 专家
# - ETP=2: 每个专家切分为 2 份
# 适合: 数千个专家，单专家参数量巨大
```

### 7.4 快速计算公式

```python
def check_parallel_config(tp, ep, etp, pp=1, world_size=8):
    """验证并行配置是否合法"""
    attn_mp = pp * tp
    moe_mp = pp * ep * etp
    
    assert world_size % attn_mp == 0, f"WORLD_SIZE 必须能被 Attention MP ({attn_mp}) 整除"
    assert world_size % moe_mp == 0, f"WORLD_SIZE 必须能被 MoE MP ({moe_mp}) 整除"
    
    dp = world_size // attn_mp
    edp = world_size // moe_mp
    
    print(f"配置有效:")
    print(f"  Data Parallel (Attention): {dp}")
    print(f"  Data Parallel (Expert): {edp}")
    print(f"  每卡专家数: 总专家数/{ep}")
    print(f"  每专家分片: 1/{etp}")
    
    return True

# 示例
check_parallel_config(tp=4, ep=8, etp=2, pp=1, world_size=64)
# 输出:
# 配置有效:
#   Data Parallel (Attention): 16
#   Data Parallel (Expert): 4
#   每卡专家数: 总专家数/8
#   每专家分片: 1/2
```

---

## 8. 常见问题与调试

### 8.1 "Expert tensor parallel size should be 1"

**原因**: 当前使用 DeepEP 时，EP > 1 的情况下不支持 ETP

**解决**: 
```python
if use_deepep:
    assert PM.size_of("ETP") == 1, "DeepEP 目前不支持 ETP"
else:
    # 可以使用 ETP
    pass
```

### 8.2 专家参数初始化不一致

**原因**: ETP 组内的参数应该相同，但随机种子设置错误

**解决**: 检查 `initialize.py` 中的种子计算是否正确

### 8.3 梯度不归约

**原因**: `expert_model_parallel` 标记未设置

**解决**: 确保 Expert 参数设置了 `param.expert_model_parallel = True`

---

## 9. 关键代码索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 并行配置定义 | `steptronoss/exp/base_exp.py` | 308-358 |
| 专家参数切分 | `steptronoss/model/common/moe_block.py` | 125-157 |
| EP 前向传播 | `steptronoss/model/common/moe_block.py` | 358-374 |
| TP 前向传播 | `steptronoss/model/common/moe_block.py` | 339-349 |
| ETP All-Reduce | `steptronoss/model/common/moe_block.py` | 168 |
| 随机种子设置 | `steptronoss/initialize.py` | 78 |
| 梯度处理 | `steptronoss/optimizer/clip_grads.py` | 131 |
| TP 层 ETP 支持 | `steptronoss/core/tensor_parallel/layers.py` | 701 |

---

> 文档版本: 2025-03
> 基于 StepTronOSS EP+TP 混合并行实现整理
