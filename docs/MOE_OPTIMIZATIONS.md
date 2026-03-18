# StepTronOSS MoE 模型加速技术详解

> 本文档详细介绍 StepTronOSS 中 MoE (Mixture of Experts) 模型的各项加速优化技术。

---

## 目录

1. [并行策略优化](#1-并行策略优化)
2. [通信优化](#2-通信优化)
3. [计算优化](#3-计算优化)
4. [负载均衡优化](#4-负载均衡优化)
5. [内存优化](#5-内存优化)
6. [优化启用指南](#6-优化启用指南)

---

## 1. 并行策略优化

### 1.1 Expert Parallel (EP) - 专家并行

EP 是 MoE 最核心的并行策略，将不同的专家分布在不同的 GPU 上。

```python
# 配置 EP
parallel_cfg.expert_model_parallel_size = 8  # EP=8
```

**实现原理** (`steptronoss/model/common/moe_block.py:358`):

```
输入 Token
    │
    ▼
Router (Gate) ──► 选择 top-k 专家
    │
    ▼
TokenDispatcher.dispatch()  # All-to-All 通信
    │
    ▼
各 GPU 只计算本地专家
    │
    ▼
TokenDispatcher.combine()   # All-to-All 通信回传
    │
    ▼
加权聚合输出
```

**通信过程**:
1. **Dispatch**: 将 token 发送到拥有目标专家的 GPU (all_to_all)
2. **Expert Compute**: 每个 GPU 只计算分配到的专家
3. **Combine**: 将计算结果传回原 GPU (all_to_all)

### 1.2 Expert Tensor Parallel (ETP) - 专家张量并行

在 EP 的基础上，对单个专家内部使用 TP 切分。

```python
parallel_cfg.expert_tensor_parallel_size = 2  # ETP=2
```

**适用场景**: 单个专家参数量过大，无法放入单卡显存。

**代码路径** (`steptronoss/model/common/moe_block.py:339`):

```python
def forward_experts_tp(self, x, token_expert_ids, token_weights):
    if self.sequence_parallel:
        x = gather_from_sequence_parallel_region(x, "ETP")
    x = self.experts(x, token_expert_ids, token_weights)
    if self.sequence_parallel:
        x = slice_to_sequence_parallel_region(x, group="ETP")
    return x
```

### 1.3 EP + TP 混合并行策略

```python
# 典型配置示例 (8 GPUs)
parallel_cfg.tensor_model_parallel_size = 2      # TP=2 (Attention 用)
parallel_cfg.expert_model_parallel_size = 4      # EP=4
parallel_cfg.expert_tensor_parallel_size = 2     # ETP=2
# 验证: TP * EP * ETP = 2 * 4 * 2 = 16 ... 需要 16 GPUs
```

**并行维度关系**:
- **Attention**: TP (张量并行)
- **MoE Router**: DP (数据并行，每个副本都有完整 router)
- **Experts**: EP × ETP (专家并行 × 专家张量并行)

---

## 2. 通信优化

### 2.1 DeepEP - 融合通信内核

DeepEP 是由 DeepSeek 开发的高效 EP 通信库，使用 fused kernels 优化 all-to-all 通信。

**文件**: `steptronoss/model/ep_dispatcher/deepep_dispatcher.py`

**配置启用**:
```python
from steptronoss.utils.optimizable import set_optimization
set_optimization(TokenDispatcher="deep_ep")
```

**核心特性**:
- NVLink 优化
- RDMA 支持
- 低延迟模式
- 自动 buffer 管理

```python
class DeepEPDispatcher:
    def __init__(self, ...):
        self._buffer = Buffer(
            group=self.group,
            num_nvl_bytes=num_nvl_bytes,      # NVLink buffer
            num_rdma_bytes=num_rdma_bytes,    # RDMA buffer
            low_latency_mode=True,
        )
```

### 2.2 原生 TokenDispatcher

默认的纯 PyTorch 实现，使用 `torch.distributed.nn.functional.all_to_all`。

**文件**: `steptronoss/model/ep_dispatcher/token_dispatcher.py`

```python
@optimizable(alternatives={"deep_ep": DeepEPDispatcher})
class TokenDispatcher:
    def dispatch(self, hidden_states, token_expert_ids, token_expert_weights):
        # 使用 all_to_all 发送 token 到目标 rank
        recv_hidden = distnn.all_to_all(send_hidden, group=self.group)
        return recv_hidden, recv_indices, recv_probs
    
    def combine(self, hidden_states):
        # 使用 all_to_all 收集计算结果
        recv_hidden = distnn.all_to_all(send_hidden, group=self.group)
        # index_add 还原顺序
        output.index_add_(0, recv_token_ids, recv_hidden)
        return output
```

### 2.3 通信-计算重叠

通过异步通信和 buffer 预分配减少通信延迟：

1. **Buffer 预分配**: DeepEP 根据历史通信量预分配 buffer
2. **双缓冲**: 当前 step 计算与下一个 step 的通信重叠
3. **梯度压缩**: 支持 FP16/BF16 通信减少带宽

---

## 3. 计算优化

### 3.1 Grouped GEMM - 专家计算融合

MoE 的核心计算是多个专家的前向传播。传统的逐专家计算会导致大量小 kernel launch，Grouped GEMM 将多个专家的矩阵乘法融合为单个 kernel。

**文件**: `steptronoss/model/utils/moe_utils.py`

**实现方式**:

```python
@optimizable(
    alternatives={
        "nv_grouped_gemm": nv_grouped_gemm,          # NVIDIA CUTLASS
        "triton_grouped_gemm": triton_grouped_gemm,  # Triton 实现
        "function_imple": function_imple_grouped_gemm,  # 纯 PyTorch
    }
)
def grouped_gemm(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    """
    mat_a_flat: [total_tokens, hidden]
    mat_b: [num_experts, expert_hidden, hidden] 或 [num_experts, hidden, expert_hidden]
    batch_sizes: [num_experts] 每个专家的 token 数量
    """
    # 默认实现: 逐个专家计算
    for i, size in enumerate(batch_sizes_list):
        outputs.append(mat_a_flat[start:start+size] @ mat_b[i])
```

**启用方式**:
```python
set_optimization(grouped_gemm="nv_grouped_gemm")
```

**性能提升**: 相比逐个专家计算，Grouped GEMM 可减少 30-50% 的 kernel launch 开销。

### 3.2 Routed Grouped FFN - 路由+计算融合

更进一步，将 token 路由和专家计算融合。

**文件**: `steptronoss/model/optimizations/routed_grouped_ffn/triton.py`

```python
@optimizable(
    alternatives={
        "fused": triton_routed_grouped_ffn_fused,
    }
)
def routed_grouped_ffn(w1, w2, act, x, token_expert_ids, token_weights):
    """
    标准实现:
    1. Scatter tokens to experts
    2. Grouped GEMM (w1)
    3. Activation
    4. Grouped GEMM (w2)
    5. Gather and weighted sum
    
    Fused 实现: 上述步骤融合为单个 Triton kernel
    """
```

**计算流程**:

```
输入 x [S, hidden]
    │
    ├──► moe_scatter ──► 按专家排序 x [S, hidden]
    │
    ├──► grouped_gemm(w1) ──► [S, ffn_hidden*2]
    │
    ├──► activation (SwiGLU) ──► [S, ffn_hidden]
    │
    ├──► grouped_gemm(w2) ──► [S, hidden]
    │
    └──► moe_weighted_gather ──► 输出 [S, hidden]
```

### 3.3 MoE Scatter/Gather 优化

Token 重排操作也有专门的优化实现。

**文件**: `steptronoss/model/utils/moe_utils.py:407`

```python
@optimizable(
    alternatives={
        "triton": triton_moe_scatter,
    }
)
def moe_scatter(input, index):
    """将 token 按专家顺序排列"""
    
@optimizable(
    alternatives={
        "triton": triton_moe_weighted_gather,
    }
)
def moe_weighted_gather(expert_output, index, weight):
    """收集专家输出并按权重加权"""
```

### 3.4 SwiGLU 激活裁剪

防止特定专家的激活值过大导致梯度爆炸。

**文件**: `steptronoss/model/common/feed_forward.py:20`

```python
def activation(self, x, swiglu_limit=None):
    l, r = torch.chunk(x, 2, dim=-1)
    l = F.silu(l)
    if swiglu_limit is not None:
        # 裁剪防止数值溢出
        l = l.clamp(min=None, max=swiglu_limit)
        r = r.clamp(min=-swiglu_limit, max=swiglu_limit)
    return l * r
```

**配置**:
```python
# 为特定层设置不同的裁剪值
moe_cfg.expert_swiglu_limits = {43: 7.0, 44: 7.0}
moe_cfg.shared_expert_swiglu_limit = {44: 16.0}
```

---

## 4. 负载均衡优化

### 4.1 Aux-Loss-Free Load Balancing - 无辅助损失负载均衡

传统 MoE 使用辅助损失 (auxiliary loss) 强制负载均衡，但这会损害模型质量。StepTronOSS 实现了无辅助损失的负载均衡方案。

**文件**: `steptronoss/model/common/moe_block.py:199`

**核心思想**:
1. Router 输出概率分布
2. 为每个专家维护一个 bias 项
3. 根据历史负载动态调整 bias
4. 选择时使用 (prob + bias)，但权重仍使用原始 prob

```python
def forward_router(self, logits):
    gate_prob = F.softmax(logits, dim=1)
    
    if self.cfg.enable_auxiliary_loss_free_load_balance:
        # 使用 bias 调整选择，但不改变权重
        biased_prob = gate_prob + self.router_balance_bias.unsqueeze(0)
        topk_expert_ids = biased_prob.topk(self.moe_top_k).indices
        token_weights = gate_prob.gather(1, topk_expert_ids)  # 原始 prob
```

**Bias 更新** (`steptronoss/model/common/moe_block.py:443`):

```python
@staticmethod
def update_router_balance_bias_per_gbs(models):
    """每个 global batch 调用一次"""
    # 1. 收集每个专家的 token 数量
    # 2. 跨 EP/ETP/EDP 聚合
    dist.all_reduce(tokens_per_expert, group=PM.group_of("EP"))
    dist.all_reduce(tokens_per_expert, group=PM.group_of("ETP"))
    dist.all_reduce(tokens_per_expert, group=PM.group_of("EDP"))
    
    # 3. 更新 bias
    average_tokens = tokens_per_expert.mean()
    offset = average_tokens - tokens_per_expert
    bias += sign(offset) * update_rate
```

**配置启用**:
```python
moe_cfg.enable_auxiliary_loss_free_load_balance = True
moe_cfg.router_bias_update_rate = 0.01  # 更新步长
```

### 4.2 Sigmoid Router

使用 Sigmoid 替代 Softmax，配合归一化实现更稳定的路由。

```python
moe_cfg.enable_sigmoid_router = True
moe_cfg.norm_expert_weight = True
```

```python
if self.use_sigmoid_router:
    gate_prob = F.sigmoid(logits)
    # 需要显式归一化
    token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True)
else:
    gate_prob = F.softmax(logits, dim=1)
```

### 4.3 Expert 容量限制 (Capacity Factor)

防止单个专家过载，丢弃超出容量的 token。

```python
# 在 TokenDispatcher 中实现
# 如果某专家接收 token 超过 capacity，则丢弃多余部分
```

---

## 5. 内存优化

### 5.1 FP32 Router Bias

保持 router balance bias 为 FP32，避免低精度导致的 routing 错误。

**文件**: `steptronoss/model/common/moe_block.py:423`

```python
def maintain_float32_router_balance_bias(self):
    """确保 bias 为 FP32"""
    if self.router_balance_bias.dtype != torch.float32:
        self.router_balance_bias.data = self.router_balance_bias.data.to(torch.float32)
```

**在 save/load 时自动处理**:
```python
def state_dict(self, ...):
    self.maintain_float32_router_balance_bias()
    return super().state_dict(...)
```

### 5.2 梯度检查点 (Recompute)

MoE 层的激活值较大，启用梯度检查点可显著降低显存。

```python
model_cfg.recompute = ["feed_forward"]  # 只重计算 FFN
```

**注意**: MoE 的 router 不重计算（见 `steptronoss/model/decoder_model.py:173`）

```python
# MoE 内部处理 recompute
out = h + self.feed_forward(ffn_in, recompute="feed_forward" in self.recompute)
```

### 5.3 Expert 参数共享

部分层使用共享专家 (Shared Expert)，减少参数总量。

```python
moe_cfg.share_expert_dim = 1280  # 共享专家维度
```

**文件**: `steptronoss/model/common/moe_share_expert_ffn.py`

```python
class MoeShareExpertFFN(nn.Module):
    def __init__(self):
        self.moe = MoEBlock(...)           # Routed experts
        self.share_expert = FeedForward(...)  # Shared expert
    
    def forward(self, x):
        moe_out = self.moe(x)
        share_out = self.share_expert(x)
        return moe_out + share_out
```

---

## 6. 优化启用指南

### 6.1 推荐配置

```python
# playground/pretrain/step3p5/step3p5_flash.py 参考配置

class Step3p5FlashMoEConfig(MoEConfig):
    def __init__(self):
        # 1. 基础 MoE 配置
        self.moe_num_experts = 288
        self.moe_top_k = 8
        self.moe_hidden_size = 1280
        self.share_expert_dim = 1280
        
        # 2. 负载均衡优化
        self.enable_auxiliary_loss_free_load_balance = True
        self.router_bias_update_rate = 0
        self.enable_sigmoid_router = True
        self.norm_expert_weight = True
        
        # 3. 激活裁剪
        self.expert_swiglu_limits = {43: 7.0, 44: 7.0}
        self.shared_expert_swiglu_limit = {44: 16.0}

class Step3p5FlashParallelConfig(ParallelConfig):
    def __init__(self):
        # 4. 并行策略
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 8
        self.context_parallel_size = 8
        self.expert_model_parallel_size = 8      # EP
        self.expert_tensor_parallel_size = 1     # ETP
```

### 6.2 计算优化启用

```python
def configure_optimizable(self):
    from steptronoss.utils.optimizable import set_optimization
    
    set_optimization(
        # 基础优化
        default="torch_compile",
        
        # Attention 优化
        AttentionCore="flash-attn",
        
        # MoE 通信优化
        TokenDispatcher="deep_ep",  # 需要安装 DeepEP
        
        # MoE 计算优化
        grouped_gemm="nv_grouped_gemm",  # 或 "triton_grouped_gemm"
        routed_grouped_ffn="fused",       # 融合 kernel
        moe_scatter="triton",
        moe_weighted_gather="triton",
    )
```

### 6.3 性能调优建议

| 优化项 | 显存节省 | 速度提升 | 适用场景 |
|--------|---------|---------|---------|
| EP (Expert Parallel) | ★★★ | ★★★ | 必需，所有 MoE 场景 |
| Grouped GEMM | - | ★★★ | 专家数多、token 数多 |
| DeepEP | - | ★★☆ | 多节点、NVLink 环境 |
| Aux-Loss-Free | - | ★☆☆ | 追求模型质量 |
| 梯度检查点 | ★★★ | ★☆☆ | 显存紧张 |
| Routed FFN Fused | - | ★★☆ | 小 batch、小专家 |

### 6.4 调试命令

```bash
# 查看 MoE 相关配置
uv run cfshow your_exp.py -k model_cfg.ffn_cfg.moe_cfg

# 检查优化是否启用
python -c "
from steptronoss.utils.optimizable import list_optimizations
print(list_optimizations())
"

# 内存诊断
export MEM_DIAGNOSE=1
uv run torchrun your_exp.py
```

---

## 7. 关键文件索引

| 功能 | 文件 |
|------|------|
| MoE 主实现 | `steptronoss/model/common/moe_block.py` |
| MoE 工具函数 | `steptronoss/model/utils/moe_utils.py` |
| Token 分发 | `steptronoss/model/ep_dispatcher/token_dispatcher.py` |
| DeepEP 实现 | `steptronoss/model/ep_dispatcher/deepep_dispatcher.py` |
| 共享专家 FFN | `steptronoss/model/common/moe_share_expert_ffn.py` |
| Routed FFN Triton | `steptronoss/model/optimizations/routed_grouped_ffn/triton.py` |
| Grouped GEMM Triton | `steptronoss/model/optimizations/grouped_gemm/triton.py` |

---

> 文档版本: 2025-03
> 基于 StepTronOSS MoE 实现整理
