# GPU Kernel 融合优化算子文档

本文档汇总了 StepTronOSS 框架中的 GPU Kernel 级融合优化实现，对应论文中提到的 "GPU Kernels Optimization" 优化项。

## 概述

框架通过以下策略实现 Kernel 级优化：
- **算子融合**：将多个小算子融合为单个 Kernel，减少 Kernel 启动开销和内存搬运
- **Triton 自定义 Kernel**：使用 Triton 编写高性能融合算子
- **多后端支持**：通过 `@optimizable` 装饰器支持多种实现后端（Triton / CUDA / PyTorch）

---

## 1. Attention 优化

### 1.1 FlashAttention 多后端支持

**位置**: `steptronoss/model/common/attention_core.py`

**功能**: 提供多种 Attention 计算后端，自动选择最优实现

| 后端 | 实现类 | 说明 |
|------|--------|------|
| `flash-attn` | `FlashAttention` | 标准 Flash Attention |
| `flash-attn-3` | `FlashAttention3` | Hopper GPU 优化的 FA3 |
| `sdpa` | `AttentionCore` | PyTorch 原生 SDPA (fallback) |

**使用方式**:
```python
from steptronoss.model.common.attention_core import AttentionCore

# 通过 set_optimization 切换后端
from steptronoss.utils.optimizable import set_optimization
set_optimization("attention_core", "flash-attn")  # 或 "flash-attn-3"
```

**特性**:
- 支持变长序列 (`cu_seqlens`)
- 支持滑动窗口注意力
- 支持 Context Parallel

### 1.2 QK Normalization + RoPE 融合

**状态**: ⚠️ **配置预留，Kernel 未实现**

**配置位置**: 
- `steptronoss/model/common/grouped_query_attention.py` (第 40 行)
- `playground/pretrain/step3p5/step3p5_flash.py` (AGENTS.md 提及)

**当前实现** (非融合版本):
```python
# steptronoss/model/common/grouped_query_attention.py:272-274
if self.use_qk_norm:
    xq = self.q_norm(xq.contiguous())  # RMSNorm
    xk = self.k_norm(xk.contiguous())  # RMSNorm

# 然后分别应用 RoPE
xq, xk = self.forward_rope(xq, xk, pos_id_q=position_id, pos_id_k=position_id)
```

**说明**: 配置参数 `use_fused_qknorm_and_rope` 和 `recompute_qknorm_rope` 已预留，但当前实现为分离的两个操作，尚未提供融合 Triton Kernel。

---

## 2. MoE (Mixture of Experts) 融合优化

MoE 模块是优化的重点，实现了完整的融合计算链路。

### 2.1 核心融合算子概览

```
输入 Token
    ↓
[Router] → top-k 专家选择
    ↓
[triton_histogram] → 统计各专家负载
    ↓
[triton_index_compute] → 计算 scatter 索引
    ↓
[triton_moe_scatter] → Token 分发到专家缓冲区
    ↓
[triton_grouped_gemm] → W1 投影 (Gate + Up)
    ↓
[Activation] → SwiGLU 激活
    ↓
[triton_grouped_gemm] → W2 投影 (Down)
    ↓
[triton_moe_weighted_gather] → 按权重收集结果
    ↓
输出 Token
```

### 2.2 Grouped GEMM

**位置**: `steptronoss/model/optimizations/grouped_gemm/`

**功能**: 将多个专家的矩阵乘法分组批量执行，提高 GPU 利用率

**文件结构**:
```
grouped_gemm/
├── __init__.py          # 导出接口
├── function_imple.py    # PyTorch 实现 (FunctionImpleGroupedGemm)
└── triton.py           # Triton 实现 (triton_grouped_gemm)
```

**接口定义**:
```python
# steptronoss/model/utils/moe_utils.py:435-452
@optimizable(
    alternatives={
        "nv_grouped_gemm": nv_grouped_gemm,      # CUTLASS 实现 (nv-grouped-gemm)
        "triton_grouped_gemm": triton_grouped_gemm,  # Triton 实现
        "function_imple": function_imple_grouped_gemm,  # PyTorch 实现
    }
)
def grouped_gemm(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    """
    Args:
        mat_a_flat: [total_tokens, K] 展平的输入
        mat_b: [num_experts, K, N] 或 [num_experts, N, K] 专家权重
        batch_sizes: [num_experts] 每个专家的 token 数
        trans_b: 是否转置 mat_b
    Returns:
        output: [total_tokens, N]
    """
```

**Triton Kernel 特性**:
- 自动调优 (auto-tune) 多种 tile 配置
- 支持 Hopper TMA (Tensor Memory Accelerator)
- 支持 FP8 计算

### 2.3 MoE Scatter (Token 分发)

**位置**: `steptronoss/model/optimizations/moe_scatter/triton.py`

**功能**: 将 token 按专家索引分发到对应的专家缓冲区

**接口**:
```python
# steptronoss/model/utils/moe_utils.py:402-420
@optimizable(
    alternatives={
        "triton": triton_moe_scatter,
    }
)
def moe_scatter(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    将 token 按索引 scatter 到专家缓冲区
    
    Args:
        input: [token_num, hidden_dim] 输入 token
        index: [token_num, top_k] 每个 token 选择的专家位置
    Returns:
        output: [num_expert_tokens, hidden_dim] 专家缓冲区
    """
```

**Triton Kernel**: `_moe_scatter_kernel`
- 并行处理每个 (token, top_k) 对
- 支持无效索引 (-1) 的掩码处理

### 2.4 MoE Weighted Gather (带权结果收集)

**位置**: `steptronoss/model/optimizations/moe_gather/triton.py`

**功能**: 将各专家的输出按权重收集回原始 token 顺序

**接口**:
```python
# steptronoss/model/utils/moe_utils.py:310-334
@optimizable(
    alternatives={
        "triton": triton_moe_weighted_gather,
    }
)
def moe_weighted_gather(
    input: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    按权重收集专家输出
    
    Args:
        input: [num_expert_tokens, hidden_dim] 专家输出
        index: [token_num, top_k] 专家位置索引
        weight: [token_num, top_k] 专家权重
    Returns:
        output: [token_num, hidden_dim]
    """
```

**Triton Kernel**: `_mritonMoEWeightedGather`
- 前向: `_moe_weighted_gather_kernel`
- 反向: `_moe_weighted_gather_grad_in_kernel` (使用 atomic_add)

### 2.5 MoE Routing 索引计算

**位置**: `steptronoss/model/optimizations/moe_routing/triton.py`

#### 2.5.1 Histogram (专家负载统计)

**接口**:
```python
# steptronoss/model/utils/moe_utils.py:85-129
@optimizable(
    alternatives={
        "triton": triton_histogram,
    }
)
def histogram(top_k_rank: torch.Tensor, expert_num: int) -> torch.Tensor:
    """统计每个专家分配的 token 数量"""
```

**Triton Kernel**: `_histogram_kernel`
- 使用 `tl.atomic_add` 并行统计
- 自动处理无效索引 (-1)

#### 2.5.2 Index Compute (索引计算)

**接口**:
```python
# steptronoss/model/utils/moe_utils.py:132-214
@optimizable(
    alternatives={
        "triton": triton_index_compute,
    }
)
def index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
    """计算稳定的 scatter 索引，保持原始顺序"""
```

**Triton Kernel**: `_count_per_block_kernel` + `_index_compute_kernel`
- 两阶段算法：先分块计数，再计算全局偏移

#### 2.5.3 Index Scatter (融合索引计算 + Scatter)

**接口**:
```python
def triton_index_scatter(
    input: torch.Tensor,
    indices: torch.Tensor,
    expert_histogram: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合索引计算和 scatter 操作"""
```

### 2.6 Routed Grouped FFN (端到端融合)

**位置**: `steptronoss/model/optimizations/routed_grouped_ffn/triton.py`

**功能**: 将 MoE 前向传播的多个操作融合为单个函数

**接口**:
```python
# steptronoss/model/utils/moe_utils.py:217-244
@optimizable(
    alternatives={
        "fused": triton_routed_grouped_ffn_fused,
    }
)
def routed_grouped_ffn(
    w1: torch.Tensor,
    w2: torch.Tensor,
    act,
    x: torch.Tensor,
    token_expert_ids: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
    """
    端到端 MoE FFN 计算
    
    融合操作:
    1. histogram - 统计专家负载
    2. index_compute - 计算索引
    3. moe_scatter - 分发 token
    4. grouped_gemm (W1) - 投影到专家空间
    5. activation - SwiGLU 激活
    6. grouped_gemm (W2) - 投影回原始空间
    7. moe_weighted_gather - 收集结果
    """
```

---

## 3. 通信优化

### 3.1 DeepEP 融合通信

**位置**: `steptronoss/model/ep_dispatcher/deepep_dispatcher.py`

**功能**: 使用 DeepEP 的融合 Kernel 进行专家并行通信

**特性**:
- 融合的 Dispatch/Combine 操作
- 支持 NVLink 和 RDMA
- 低延迟模式 (low_latency_mode)
- 自动缓冲区管理

**接口**:
```python
class DeepEPDispatcher:
    def dispatch(self, hidden_states, token_expert_ids, token_expert_weights):
        """分发 token 到各 EP rank"""
        
    def combine(self, hidden_states):
        """收集各 EP rank 的结果"""
```

---

## 4. FeedForward 优化

### 4.1 SwiGLU + W2 融合

**位置**: `steptronoss/model/common/feed_forward.py`

**功能**: 将 SwiGLU 激活与 W2 投影融合，减少中间结果存储

**配置**:
```python
# FeedForwardConfig
swiglu_recompute_silu_out_proj: bool  # 启用融合
```

**实现**:
```python
def _forward(self, x):
    if self.fuse_activation_w2:
        # 融合路径：W1 → (存储激活值) → W2 (在 W2 内部执行 SwiGLU)
        x = self.w1(x)[0]
        output = self.w2(x)[0]  # W2 内部执行 activation
    else:
        # 分离路径
        x = self.activation(self.w1(x)[0])
        output = self.w2(x)[0]
```

---

## 5. 使用指南

### 5.1 启用 Triton 优化

```python
from steptronoss.utils.optimizable import set_optimization

# 启用 Triton Grouped GEMM
set_optimization("grouped_gemm", "triton_grouped_gemm")

# 启用 NV Grouped GEMM (需要安装 nv-grouped-gemm)
set_optimization("grouped_gemm", "nv_grouped_gemm")

# 启用 Triton MoE Scatter
set_optimization("moe_scatter", "triton")

# 启用 Triton MoE Gather
set_optimization("moe_weighted_gather", "triton")

# 启用融合 Routed FFN
set_optimization("routed_grouped_ffn", "fused")
```

### 5.2 配置文件中设置

```python
# 在实验配置中设置优化选项
class MyExpConfig(BaseExp):
    def __init__(self):
        super().__init__()
        # 启用 FlashAttention-3 (Hopper)
        self.model_cfg.attn_cfg.attention_backend = "flash-attn-3"
        
        # 启用 SwiGLU 融合
        self.model_cfg.ffn_cfg.swiglu_recompute_silu_out_proj = True
```

### 5.3 环境变量

```bash
# 启用内存诊断 (查看各算子内存使用)
export MEM_DIAGNOSE=1

# 启用时间统计
export STEPTRON_TIMERS=1
```

---

## 6. 性能调优建议

### 6.1 MoE 优化建议

1. **优先使用 Triton 实现**: 在大多数 GPU 上，Triton 实现的 MoE 算子性能优于 PyTorch 原生实现
2. **使用 NV Grouped GEMM**: 如果在 H100 等支持 Tensor Core 的 GPU 上，安装 `nv-grouped-gemm` 可获得最佳性能
3. **启用融合 Routed FFN**: 减少中间结果的内存搬运

### 6.2 Attention 优化建议

1. **安装 Flash Attention**: `pip install flash-attn --no-build-isolation`
2. **Hopper GPU 使用 FA3**: 如果在 H100/H800 上，安装 `flash-attn-3`
3. **长序列使用 Context Parallel**: 配置 `context_parallel_size > 1`

---

## 7. 实现状态汇总

| 优化项 | 状态 | 位置 | 备注 |
|--------|------|------|------|
| FlashAttention | ✅ 完成 | `attention_core.py` | 支持 FA2/FA3/SDPA |
| QK Norm + RoPE 融合 | ⚠️ 未实现 | - | 配置预留，Kernel 待开发 |
| Grouped GEMM | ✅ 完成 | `grouped_gemm/` | 3 种后端 |
| MoE Scatter | ✅ 完成 | `moe_scatter/` | Triton Kernel |
| MoE Gather | ✅ 完成 | `moe_gather/` | Triton Kernel |
| MoE Routing | ✅ 完成 | `moe_routing/` | Histogram + Index |
| Routed FFN 融合 | ✅ 完成 | `routed_grouped_ffn/` | 端到端融合 |
| DeepEP 通信 | ✅ 完成 | `deepep_dispatcher.py` | 需安装 DeepEP |
| SwiGLU + W2 融合 | ✅ 完成 | `feed_forward.py` | PyTorch 实现 |

---

## 8. 参考文档

- [Triton Acceleration Workflow](TRITON_ACCELERATION_WORKFLOW.md)
- [MoE Optimizations](MOE_OPTIMIZATIONS.md)
- [Parallelism Optimizations](PARALLELISM_OPTIMIZATIONS.md)
