# SteptronOss 并行策略与优化手段详解

本文档系统梳理 SteptronOss 框架在各并行维度（TP、PP、CP、EP）中使用的通算融合与 MFU 优化技术，以及优化器层面和模型层面的优化手段。

---

## 目录

- [一、并行策略的统一管理架构](#一并行策略的统一管理架构)
  - [1.1 配置声明层：einops 风格的并行定义](#11-配置声明层einops-风格的并行定义)
  - [1.2 进程组管理层：ParallelManager (PM)](#12-进程组管理层parallelmanager-pm)
  - [1.3 算子使用层：各并行维度如何消费 PM](#13-算子使用层各并行维度如何消费-pm)
- [二、TP（Tensor Parallel）优化](#二tptensor-parallel优化)
  - [2.1 异步 AllReduce 与计算重叠](#21-异步-allreduce-与计算重叠)
  - [2.2 Sequence Parallel：ReduceScatter 替代 AllReduce](#22-sequence-parallelreducescatter-替代-allreduce)
  - [2.3 Gradient Accumulation Fusion](#23-gradient-accumulation-fusion)
  - [2.4 MoE Column Parallel 激活分片保存](#24-moe-column-parallel-激活分片保存)
  - [2.5 Pre-Recompute Function 融合（SiLU Recompute）](#25-pre-recompute-function-融合silu-recompute)
- [三、PP（Pipeline Parallel）优化](#三pppipeline-parallel优化)
  - [3.1 Overlap P2P Communication](#31-overlap-p2p-communication)
  - [3.2 Deallocate Output Tensor（伪释放）](#32-deallocate-output-tensor伪释放)
  - [3.3 VPP Interleaved 1F1B 减少 Bubble](#33-vpp-interleaved-1f1b-减少-bubble)
  - [3.4 固定 Shape 跳过 Shape 通信](#34-固定-shape-跳过-shape-通信)
  - [3.5 Activation CPU Offload](#35-activation-cpu-offload)
- [四、CP（Context Parallel）优化](#四cpcontext-parallel优化)
  - [4.1 Balanced CP 互补分片](#41-balanced-cp-互补分片)
  - [4.2 Attention 中的 CP 实现](#42-attention-中的-cp-实现)
  - [4.3 TP-SP + CP 联合工作时的通信流程](#43-tp-sp--cp-联合工作时的通信流程)
- [五、MoE / EP（Expert Parallel）优化](#五moe--epexpert-parallel优化)
  - [5.1 Triton 融合核](#51-triton-融合核)
  - [5.2 Auxiliary-Loss-Free Load Balancing](#52-auxiliary-loss-free-load-balancing)
  - [5.3 ETP/EP 分离的通信组](#53-etpep-分离的通信组)
- [六、优化器层面优化](#六优化器层面优化)
  - [6.1 ZeRO-1 ReduceScatter + AllGather](#61-zero-1-reducescatter--allgather)
  - [6.2 分桶 + 连续梯度缓冲区](#62-分桶--连续梯度缓冲区)
  - [6.3 混合精度梯度管理](#63-混合精度梯度管理)
  - [6.4 Expert Parallel 梯度处理](#64-expert-parallel-梯度处理)
  - [6.5 Muon 优化器](#65-muon-优化器)
- [七、模型层面优化](#七模型层面优化)
  - [7.1 @optimizable 装饰器](#71-optimizable-装饰器)
  - [7.2 分层 Recompute 策略](#72-分层-recompute-策略)
  - [7.3 Distribute Saved Activations（激活分布式保存）](#73-distribute-saved-activations激活分布式保存)
  - [7.4 RNG 状态管理](#74-rng-状态管理)
- [八、总结对照表](#八总结对照表)

---

## 一、并行策略的统一管理架构

整个并行系统分为三层：**配置声明层 → 进程组管理层 → 算子使用层**。

### 1.1 配置声明层：einops 风格的并行定义

核心在 `ParallelConfig`（`steptronoss/exp/base_exp.py`）中：

```python
class ParallelConfig(Config):
    parallel_definition: dict[str, str] = {
        "TP":   "(p d t) -> (p d) t",
        "PP":   "(p d t) -> (d t) p",
        "DP":   "(p d t) -> (p t) d",
        "CP":   "(p d c t) -> (p d t) c",
        "EP":   "(p edp ep etp) -> (p edp etp) ep",
        "ETP":  "(p edp ep etp) -> (p edp ep) etp",
        "EETP": "(p edp ep etp) -> (p edp) (ep etp)",
        "EDP":  "(p edp ep etp) -> (p ep etp) edp",
        "EMP":  "(p edp ep etp) -> edp (p ep etp)",
        "MP":   "(p d t) -> d (p t)",
    }
    tensor_model_parallel_size: int = 1    # t
    pipeline_model_parallel_size: int = 1  # p
    context_parallel_size: int = 1         # c
    expert_model_parallel_size: int = 1    # ep
    expert_tensor_parallel_size: int = 1   # etp
```

每种并行用一个 **einops rearrange 表达式**来声明：

- `(p d t)` 是全局 rank 的逻辑视图，其中 `p`=PP size, `d`=DP size（自动推导）, `t`=TP size
- `->` 左边括号内是"同一组内的 ranks"，右边是"分组维度"

以 8 GPU、TP=2、PP=2 为例（`d` 自动推导为 2）：

```
全局 ranks: [0, 1, 2, 3, 4, 5, 6, 7]
逻辑视图 (p=2, d=2, t=2):
  [[[0,1], [2,3]],   # p=0
   [[4,5], [6,7]]]   # p=1

TP "(p d t) -> (p d) t":
  → [[0,1], [2,3], [4,5], [6,7]]  (每组 2 个 rank)

PP "(p d t) -> (d t) p":
  → [[0,4], [1,5], [2,6], [3,7]]  (每组 2 个 rank)

DP "(p d t) -> (p t) d":
  → [[0,2], [1,3], [4,6], [5,7]]  (每组 2 个 rank)
```

DP size 不需要手动指定，由 `world_size / (p * t)` 自动推导。

### 1.2 进程组管理层：ParallelManager (PM)

`ParallelManager`（`steptronoss/core/parallel_state.py`）是全局单例，提供统一的进程组管理：

```python
PM = ParallelManager()  # 全局单例
```

**初始化流程**：

```
PM.initialize()                       # 初始化 torch.distributed
  ↓
PM.set_mesh(parallel_cfg)             # 传入 ParallelConfig
  ↓
parallel_cfg.build_parallel()         # 调用 einops rearrange 生成 ranks 分组
  ↓
对每个并行维度调用 PM.new_parallel(name, ranks)
  → torch.distributed.new_group(ranks)   # 创建实际的 ProcessGroup
  → 存入 PM.parallels[name] = ParallelGroups(...)
```

**统一查询 API**（所有并行维度通过名字访问，接口完全一致）：

| API | 用途 | 示例 |
|-----|------|------|
| `PM.size_of("TP")` | 该并行组的大小 | `4` |
| `PM.rank_in("TP")` | 我在组内的 rank | `2` |
| `PM.i_am("TP", 0)` | 我是不是组内第 0 号 | `True/False` |
| `PM.group_of("TP")` | 获取 ProcessGroup（用于通信） | `<ProcessGroup>` |
| `PM.ranks_of("PP")` | 我所在组的所有全局 rank | `[0, 4]` |

**多 Mesh 支持**（`use_mesh` 上下文管理器）：

```python
with PM.use_mesh(actor_parallel_cfg):
    # 此 scope 内 PM 的查询指向 actor 的并行组
    ...
with PM.use_mesh(critic_parallel_cfg):
    # 此 scope 内切换为 critic 的并行组
    ...
```

这对 RLVR 场景（actor 和 critic 用不同并行策略）非常关键。内部通过栈结构 `_stack` 保存和恢复上下文。

### 1.3 算子使用层：各并行维度如何消费 PM

所有并行层的通信算子都通过 `PM.group_of(name)` 获取通信组，**不硬编码任何进程组**。

```
用户配置:
  parallel_cfg.tensor_model_parallel_size = 4
  parallel_cfg.pipeline_model_parallel_size = 2
  parallel_cfg.context_parallel_size = 2
                    ↓
einops rearrange 生成进程组分组:
  "(p d c t) -> (p d t) c" + {p=2, c=2, t=4}
  → 每个 CP 组 = 2 个 rank
                    ↓
PM.new_parallel() 创建 torch ProcessGroup
  PM.parallels["CP"] = ParallelGroups(groups=[...], ranks=[...])
                    ↓
模型层使用 PM 查询:
  ColumnParallelLinear:  PM.size_of("TP") → 切分权重
  scatter_to_balanced_cp_region:  PM.size_of("CP"), PM.rank_in("CP") → 切分序列
  PPScheduler:  PM.size_of("PP"), PM.rank_in("PP") → 调度 1F1B
  all_reduce:  PM.group_of("DP") → 梯度同步
```

---

## 二、TP（Tensor Parallel）优化

### 2.1 异步 AllReduce 与计算重叠

**代码位置**：`steptronoss/core/tensor_parallel/layers.py:311-313`

```python
# ColumnParallelLinear backward
if ctx.async_grad_allreduce and not ctx.use_moe:
    handle = torch.distributed.all_reduce(
        grad_input, group=PM.group_of("TP"), async_op=True
    )
    # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 保证通信先于 weight grad 计算被调度
```

**原理**：

反向传播时，ColumnParallelLinear 需要做两件事：
1. `grad_input` 的 AllReduce（TP 组内通信）
2. `grad_weight = grad_output.T @ total_input`（本地矩阵乘法）

这两个操作之间**没有数据依赖**。通过 `async_op=True`，AllReduce 通信与 `grad_weight` 的矩阵乘法计算**在不同 CUDA stream 上并行执行**。

关键前提：设置环境变量 `CUDA_DEVICE_MAX_CONNECTIONS=1`，强制 CUDA driver 按代码调用顺序排队 kernel。这保证通信 kernel 先入队（优先级更高的通信 stream），使得通信与计算真正 overlap。

**数据流示意**：

```
时间线 ──────────────────────────────────>

Stream 0 (compute):  [grad_input = grad_out @ W] ─── [grad_weight = grad_out.T @ input] ───
Stream 1 (nccl):                                  ─── [AllReduce(grad_input)]            ───
                                                       ↑ 通信与 grad_weight 计算并行 ↑
```

### 2.2 Sequence Parallel：ReduceScatter 替代 AllReduce

**代码位置**：`steptronoss/core/tensor_parallel/layers.py:200-206, 317-332`

```python
# Forward: all-gather 输入，拼接后做矩阵乘法
torch.distributed.all_gather_into_tensor(
    all_gather_buffer, input, group=PM.group_of("TP")
)
output = torch.matmul(total_input, weight.t())

# Backward: reduce-scatter 梯度
torch.distributed.reduce_scatter_tensor(
    sub_grad_input, grad_input, group=PM.group_of("TP"), async_op=True
)
```

**与非 SP 模式对比**：

| 项目 | 非 SP 模式 | SP 模式 |
|------|-----------|---------|
| LayerNorm 等非 TP 区域 | 全量序列冗余计算 | 每个 rank 只处理 `S/TP_size` |
| 前向通信 | 无（冗余计算） | AllGather（通信量 N） |
| 反向通信 | AllReduce（通信量 2N） | ReduceScatter（通信量 N） |
| 显存占用 | 全量激活 | 1/TP_size 激活 |

SP 模式总通信量相同（前向 N + 反向 N = 2N），但**非 TP 区域的计算量和显存都减少为 1/TP_size**。

**反向传播的 ReduceScatter 同样利用了 async_op 实现与 grad_weight 计算的重叠**：

```python
# reduce_scatter 先入队
handle = torch.distributed.reduce_scatter_tensor(
    sub_grad_input, grad_input, group=PM.group_of("TP"), async_op=True
)
# grad_weight 计算与 reduce_scatter 并行
grad_weight = get_grad_weight(total_input, grad_output, weight)
# 最后等待
handle.wait()
```

### 2.3 Gradient Accumulation Fusion

**代码位置**：`steptronoss/core/tensor_parallel/layers.py:244-248`

```python
if ctx.gradient_accumulation_fusion:
    if weight.main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input, grad_output, weight.main_grad
        )
        grad_weight = None  # 不返回 grad_weight，直接累加到 main_grad
```

**原理**：

正常流程需要两步：
1. `grad_weight = grad_output.T @ total_input`（GEMM）
2. `main_grad += grad_weight`（accumulate）

使用 APEX 的 fused CUDA kernel，将这两步**合并为一个 kernel**：
- 省去 `grad_weight` 中间 tensor 的显存分配
- 减少一次 kernel launch 的开销
- 直接在 FP32 的 `main_grad` 上累加，保证精度

**使用要求**：需要安装 APEX 并启用 `--cpp_ext --cuda_ext`，且要求 CUDA >= 11。

### 2.4 MoE Column Parallel 激活分片保存

**代码位置**：`steptronoss/core/tensor_parallel/layers.py:184-196`

```python
if use_moe and is_column_parallel:
    if input.size()[0] <= (world_size - 1) ** 2:
        ctx.moe_should_split_input = False
        ctx.save_for_backward(input, weight)
    else:
        ctx.moe_should_split_input = True
        moe_saved_input, ctx.moe_input_pad_size = split_along_first_dim_with_padding(input)
        # 只存 1/TP 的激活，反向时再 AllGather 恢复
        ctx.save_for_backward(moe_saved_input.clone(), weight)
```

**原理**：

MoE 的 expert 输入 token 数量可能很大（所有被路由到该 expert 的 token）。前向时只保存当前 TP rank 对应的 `1/TP_size` 切片，反向时再 AllGather 恢复完整输入。

**用通信换显存**：额外的 AllGather 通信开销可以接受，因为腾出的显存可以支持更大的 batch 或更长的序列，从而提高整体吞吐。

反向恢复的逻辑：

```python
# backward 中恢复完整输入
if is_moe_gather_activation:
    moe_saved_input, weight = ctx.saved_tensors
    gathered_input_list = [torch.empty_like(moe_saved_input) for _ in range(world_size)]
    gathered_input_list[rank] = moe_saved_input
    gathered_input_handler = torch.distributed.all_gather(
        gathered_input_list, moe_saved_input,
        group=PM.group_of("TP"), async_op=True,
    )
    # AllGather 与 grad_input = grad_output @ weight 计算并行
    grad_input = grad_output.matmul(weight)
    gathered_input_handler.wait()
    total_input = torch.cat(gathered_input_list, dim=0)
```

注意这里的 AllGather 也是 `async_op=True`，与 `grad_input` 的计算重叠。

### 2.5 Pre-Recompute Function 融合（SiLU Recompute）

**代码位置**：`steptronoss/model/common/feed_forward.py:52-58`

```python
class FeedForward(torch.nn.Module):
    def __init__(self, cfg: FeedForwardConfig, layer_id: int):
        ...
        self.fuse_activation_w2 = cfg.swiglu_recompute_silu_out_proj

        self.w2 = tensor_parallel.RowParallelLinear(
            cfg.ffn_hidden_size, cfg.hidden_size,
            bias=False, input_is_parallel=True,
            custom_pre_recompute_function=(
                self.activation if self.fuse_activation_w2 else None
            ),
            ...
        )

    def _forward(self, x) -> torch.FloatTensor:
        if self.fuse_activation_w2:
            x = self.w1(x)[0]       # 保存 w1 的输出（SiLU 之前）
            output = self.w2(x)[0]   # SiLU 在 w2 内部执行，不保存中间结果
        else:
            x = self.activation(self.w1(x)[0])  # 保存 SiLU 的输出
            output = self.w2(x)[0]
```

**原理**：

SwiGLU 激活函数：`output = SiLU(W1_a @ x) * (W1_b @ x)`

- 非融合模式：需要保存 SiLU 的输出，维度为 `ffn_hidden_size`（通常是 `hidden_size` 的 4 倍），占用大量显存
- 融合模式：只保存 SiLU **之前**的输入（`W1` 的输出），反向传播时在 `LinearWithGradAccumulationAndAsyncCommunicationWithPrefunction` 中**先重算 SiLU，再计算梯度**

```python
# layers.py:416-423  WithPrefunction 的 backward
# 重新计算 pre_function（即 SiLU）
detached_inputs = detach_variable((input, custom_pre_recompute_function_input))
with torch.enable_grad():
    pre_func_output = ctx.custom_pre_recompute_function(*detached_inputs)
input = pre_func_output
# 然后用重算的结果计算 grad_weight
```

**效果**：SiLU 的计算量很小（element-wise），但保存的中间结果显存很大（`ffn_hidden_size` 维度），所以重算的 ROI 很高。

---

## 三、PP（Pipeline Parallel）优化

### 3.1 Overlap P2P Communication

**代码位置**：`steptronoss/core/pipeline_parallel/p2p_communication.py:305-306`

```python
if cfg.overlap_p2p_comm:
    p2p_func = _p2p_ops        # 独立 isend/irecv，返回 handles
else:
    p2p_func = _batched_p2p_ops  # batch_isend_irecv，同步等待
```

**两种 P2P 模式对比**：

| 模式 | 实现 | 特点 |
|------|------|------|
| `_batched_p2p_ops` | `dist.batch_isend_irecv(ops)` | 一次系统调用批量提交，但必须一起等待 |
| `_p2p_ops` | 独立 `dist.isend`/`dist.irecv` | 按奇偶 rank 交错发送/接收，可单独 wait |

`_p2p_ops` 中的奇偶交错防止死锁：

```python
# p2p_communication.py:156-231
if mpu.get_pipeline_model_parallel_rank() % 2 == 0:
    # 偶数 rank: 先 send_next → recv_prev → send_prev → recv_next
else:
    # 奇数 rank: 先 recv_prev → send_next → recv_next → send_prev
```

`overlap_p2p_comm=True` 时，通信函数返回 `waiter` 回调，调度器可以**在计算完成后才调用 wait**：

```python
# p2p_communication.py:478-485  send_forward_recv_forward
if overlap_p2p_comm:
    def waiter():
        for req in wait_handles:
            req.wait()
        timers("forward-send-forward-recv").stop()
    return input_tensor, waiter
```

**VPP 调度器中的实际使用**（`schedules.py:496-576`）：

```python
# VPP 1F1B steady state
for k in range(num_microbatches_remaining):
    fwd_waiter()   # 等上一轮的前向通信完成

    output_tensor = forward_step_helper(forward_k)  # 前向计算

    # 发送前向输出 + 接收下一个前向输入（异步）
    input_tensor, fwd_waiter = p2p_comm.send_forward_recv_forward(
        ..., overlap_p2p_comm=True
    )

    bwd_waiter()   # 等上一轮的反向通信完成

    input_tensor_grad = backward_step_helper(backward_k)  # 反向计算

    # 发送反向梯度 + 接收下一个反向梯度（异步）
    output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
        ..., overlap_p2p_comm=True
    )
```

这样前向通信与反向计算重叠，反向通信与下一轮前向计算重叠。

### 3.2 Deallocate Output Tensor（伪释放）

**代码位置**：`steptronoss/core/pipeline_parallel/schedules.py:26-40`

```python
def deallocate_output_tensor(out):
    """Pseudo-deallocate the output tensor's '.data' field."""
    assert isinstance(out, torch.Tensor)
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

**原理**：

PP 中间 stage 的输出 tensor 在发送给下一个 stage 后，**数据部分已无用**，只有 `.grad_fn` 属性对反向传播有价值（用于构建计算图）。通过将 `.data` 替换为标量 tensor：
- **释放了激活数据的显存**
- **保留了计算图**（`.grad_fn` 不受影响）

反向传播时使用 `custom_backward` 直接调用 C++ autograd engine：

```python
# schedules.py:43-73
def custom_backward(output, grad_output):
    """Directly call C++ autograd engine."""
    assert output.numel() == 1, "output should be pseudo-'freed' in schedule"
    # 跳过 PyTorch 的 shape 检查（因为 data 已被替换为标量）
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False, create_graph=False,
        inputs=tuple(), allow_unreachable=True, accumulate_grad=True,
    )
```

### 3.3 VPP Interleaved 1F1B 减少 Bubble

**代码位置**：`steptronoss/core/pipeline_parallel/schedules.py:319-693`

框架实现了三种 PP 调度器，自动选择：

```python
# base_exp.py:396-410  MegatronPPModelConfig.get_pp_scheduler()
if PM.size_of("PP") > 1:
    if get_vpp_size() > 1:
        return VPPScheduler(config=self)   # VPP > 1: Interleaved 1F1B
    else:
        return PPScheduler(config=self)    # VPP = 1: 普通 1F1B
else:
    return FWBWScheduler(config=self)      # PP = 1: 简单前向后向
```

**VPP Interleaved 1F1B 的核心逻辑**：

将每个 PP stage 的层拆成 `num_model_chunks` 份交错放置。例如 PP=4、VPP=2 时：

```
Stage 0: [Layer 0-3] + [Layer 16-19]
Stage 1: [Layer 4-7] + [Layer 20-23]
Stage 2: [Layer 8-11] + [Layer 24-27]
Stage 3: [Layer 12-15] + [Layer 28-31]
```

Warmup 阶段的 micro-batch 数量计算：

```python
# schedules.py:352-358
if forward_num == PM.size_of("PP"):
    num_warmup_microbatches = num_microbatches
    all_warmup_microbatches = True
else:
    num_warmup_microbatches = (PM.size_of("PP") - PM.rank_in("PP") - 1) * 2
    num_warmup_microbatches += (num_model_chunks - 1) * PM.size_of("PP")
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
```

**Bubble 率对比**：
- 普通 1F1B：`bubble = (p-1) / m`（p=PP size, m=micro-batch 数）
- VPP Interleaved 1F1B：`bubble ≈ (p-1) / (m * v)`（v=VPP size）

VPP=2 时 bubble 率约减半。

**数据预取**：调度器在每个 iteration 开始时一次性预取所有 micro-batch 的数据，避免在 forward_chunk 中逐个取：

```python
# schedules.py:157-175
def _prefetch_iteration_data(self, forward_num: int):
    self._prefetched_data = [[] for _ in self.models]
    for vp_rank, data_iter in enumerate(self.data_iterators):
        set_vpp_rank(vp_rank)
        for _i in range(forward_num):
            self._prefetched_data[vp_rank].append(self.data_sync_fn(data_iter))
```

### 3.4 固定 Shape 跳过 Shape 通信

**代码位置**：`steptronoss/core/pipeline_parallel/p2p_communication.py:270-278`

```python
if not cfg.variable_seq_lengths:
    recv_prev_shape = cfg.pp_comm_shape   # 预设形状，跳过 shape 通信
else:
    recv_prev_shape, recv_next_shape = _communicate_shapes(
        tensor_send_next, tensor_send_prev, recv_prev, recv_next
    )
```

**原理**：

`variable_seq_lengths=False`（预训练默认）时，所有 micro-batch 的 activation tensor 形状相同（`[seq_len, micro_batch_size, hidden_size]`），可以直接用预设的 `pp_comm_shape` 分配接收缓冲区。

`variable_seq_lengths=True`（SFT/推理）时，每个 micro-batch 的序列长度可能不同，需要**先通信 shape 再通信 data**。`_communicate_shapes` 通过 `batch_isend_irecv` 传输 3 个 int64 的 shape tensor。

**效果**：预训练场景下省掉每个 micro-batch 的一轮 P2P shape 通信，减少 PP 阶段的通信延迟。

### 3.5 Activation CPU Offload

**代码位置**：`steptronoss/core/pipeline_parallel/schedules.py:240-246`

```python
# PPScheduler.run()
activation_cpu_offload = self.training and self.config.pipeline_activation_cpu_offload

def forward_step(input_tensor):
    saved_tensor_ctx = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if activation_cpu_offload else nullcontext()
    )
    with saved_tensor_ctx:
        return self.forward_chunk(input_tensor=input_tensor, loss_scale=1 / forward_num)
```

**原理**：

PP warmup 阶段会累积多个 micro-batch 的前向激活（等待反向传播），这些激活占用大量 GPU 显存。开启 `pipeline_activation_cpu_offload` 后：
- 前向传播中 `torch.autograd` 自动保存的 tensor **自动 offload 到 CPU pinned memory**
- 反向传播时，当需要这些 tensor 时再**自动从 CPU 回传到 GPU**
- `pin_memory=True` 确保 CPU 内存使用 pinned memory，加速 CPU↔GPU 数据传输

**VPP 调度器中同样支持**（`schedules.py:337, 383-386`）。

---

## 四、CP（Context Parallel）优化

### 4.1 Balanced CP 互补分片

**代码位置**：`steptronoss/core/context_parallel/context_parallel.py:120-148`

```python
def scatter_to_balanced_cp_region(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    """Given a tensor with sequence number [0, 1, 2, 3, 4, 5, 6, 7], CP=2, TP=2
    CP partition: CP0: [0, 1, 6, 7], CP1: [2, 3, 4, 5]
    """
    chunk_size = S // (cp_size * 2)
    left_start = cp_rank * chunk_size          # 头部
    right_start = (2 * cp_size - 1 - cp_rank) * chunk_size  # 尾部（互补位置）

    left = torch.arange(left_start, left_start + chunk_size)
    right = torch.arange(right_start, right_start + chunk_size)
    indices = torch.cat((left, right), dim=0).to(tensor.device)
    output = torch.index_select(tensor, dim=dim, index=indices).contiguous()
```

**为什么不是简单均分**？

对于 causal attention（每个 token 只能 attend 到自己及之前的 token）：

```
简单均分:
  CP0: [0,1,2,3]  → KV range = [0..3]  (4 个 KV)
  CP1: [4,5,6,7]  → KV range = [0..7]  (8 个 KV)
  → CP1 的计算量是 CP0 的 2 倍，严重负载不均！

互补分片:
  CP0: [0,1,6,7]  → KV range = [0..7]  (8 个 KV, 分两半处理)
  CP1: [2,3,4,5]  → KV range = [0..5]  (6 个 KV, 分两半处理)
  → 两个 CP rank 的计算量接近，负载均衡
```

**聚集（反向传播时）的恢复逻辑**：

```python
# context_parallel.py:151-178
def gather_from_balanced_cp_region(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    """
    tensor: [0, 1, 6, 7], [2, 3, 4, 5]
    inp   : [0, 1, 6, 7, 2, 3, 4, 5]
    res   : [0, 1, 2, 3, 4, 5, 6, 7]  ← 恢复原始顺序
    """
    inp = _GatherFromCPRegion.apply(tensor)  # AllGather
    inp0, inp1 = inp.reshape(cp_size, S_local, *raw_shape).chunk(2, dim=1)
    inp1 = inp1.flip(0)  # 翻转右半部分
    res = torch.cat((inp0, inp1), dim=0).reshape(...)
```

`_GatherFromCPRegion` 是 `torch.autograd.Function`，前向做 AllGather，反向自动做 ReduceScatter，保证梯度正确传播。

### 4.2 Attention 中的 CP 实现

**代码位置**：`steptronoss/model/common/grouped_query_attention.py:276-328`

```python
if self.cp:
    # 1. 计算当前 CP rank 的左右两半的 Q/KV 范围
    info0, info1 = cu_seqlens_to_balanced_cp(
        cu_seqlens, cp_rank=PM.rank_in("CP"), cp_size=PM.size_of("CP")
    )
    q_range0, q_cumlen0, max_q0, k_range0, k_cumlen0, max_k0 = info0
    q_range1, q_cumlen1, max_q1, k_range1, k_cumlen1, max_k1 = info1

    # 2. AllGather KV 到完整序列，然后按范围切片
    xk_full = gather_from_balanced_cp_region(xk)
    xv_full = gather_from_balanced_cp_region(xv)
    xk0 = xk_full[k_range0[0]:k_range0[1]]
    xv0 = xv_full[k_range0[0]:k_range0[1]]
    xk1 = xk_full[k_range1[0]:k_range1[1]]
    xv1 = xv_full[k_range1[0]:k_range1[1]]

    # 3. Q 分两半
    xq0, xq1 = xq.chunk(2, 0)

    # 4. 分别做 Attention（两次 FlashAttention 调用）
    out0 = self.forward_attention_core(xq0, xk0, xv0,
        cu_seqlens={"q": q_cumlen0, "k": k_cumlen0}, ...)
    out1 = self.forward_attention_core(xq1, xk1, xv1,
        cu_seqlens={"q": q_cumlen1, "k": k_cumlen1}, ...)
    output = torch.cat([out0, out1], dim=1)
```

`cu_seqlens_to_balanced_cp` 函数（`context_parallel.py:51-117`）处理 packed sequence 的场景：

- 输入是 `cu_seqlens`（多个样本 pack 在一起的累积长度）
- 输出是每个 CP rank 左右两半的 `q_range, q_cumlen, k_range, k_cumlen`
- **正确处理样本边界**：确保 CP 分片不会跨越样本边界导致错误的 attention mask

### 4.3 TP-SP + CP 联合工作时的通信流程

当 Tensor Parallel（TP）开启 Sequence Parallel（SP）模式，同时启用 Context Parallel（CP）时，序列维度经历**两级切分**，通信模式也相应叠加。本节以 `TP=2, CP=2` 为例，详细分析一个 TransformerBlock 的完整前后向通信链路。

#### 4.3.1 序列维度的两级切分

```
全局序列长度 S
  ├── CP 切分（Balanced 互补分片，在模型入口处执行）
  │     每个 CP rank 持有: S_cp = S / CP
  │
  └── SP 切分（Embedding 层输出时 Scatter）
        每个 TP rank 持有: S_local = S_cp / TP = S / (CP × TP)
```

**代码位置**：

- CP 切分：`decoder_model.py:247-248`，在 `forward_head()` 中通过 `scatter_to_balanced_cp_region(input_ids, dim=1)` 对 token ids 做本地 index_select（无通信）
- SP 切分：`parallel_embedding.py:77-78`，Embedding 输出后通过 `scatter_to_sequence_parallel_region(embeddings)` 将 `[S_cp, B, H]` 切分为 `[S_local, B, H]`

因此进入 TransformerBlock 时，每个 rank 持有的 hidden states 形状为 `[S_local, B, H]`，其中 `S_local = S / (CP × TP)`。

#### 4.3.2 前向通信流程（一个 TransformerBlock）

```
输入 x: [S_local, B, H]
  │
  ├─ 1. AttnNorm (RMSNorm，逐元素操作，无通信)
  │     输出: [S_local, B, H]
  │
  ├─ 2. wqkv (ColumnParallelLinear)
  │     ┌─ AllGather(TP): [S_local, B, H] → [S_cp, B, H]
  │     │  代码: layers.py:200-206
  │     │  torch.distributed.all_gather_into_tensor(buffer, input, group=PM.group_of("TP"))
  │     │
  │     └─ MatMul: [S_cp, B, H] × W_qkv^T → [S_cp, B, (H_q+2*H_kv)/TP]
  │        输出形状: Q=[S_cp, B, n_local_heads, head_dim], K/V=[S_cp, B, n_local_kv_heads, head_dim]
  │
  ├─ 3. CP Attention
  │     ┌─ AllGather(CP) for K,V: [S_cp, B, n_kv, d] → [S, B, n_kv, d]
  │     │  代码: grouped_query_attention.py:285-286
  │     │  xk_full = gather_from_balanced_cp_region(xk)  # 内部调用 _GatherFromCPRegion.apply()
  │     │  xv_full = gather_from_balanced_cp_region(xv)  # → dist.all_gather_into_tensor(group=CP)
  │     │
  │     ├─ Q 分两半: xq0, xq1 = xq.chunk(2, 0)
  │     │  每半: [S_cp/2, B, n_local_heads, head_dim]
  │     │
  │     ├─ KV 按范围切片:
  │     │  xk0 = xk_full[k_range0], xk1 = xk_full[k_range1]  (causal mask 决定的 KV 范围)
  │     │
  │     ├─ 2× FlashAttention (无通信，本地计算)
  │     │  out0 = flash_attn(xq0, xk0, xv0)
  │     │  out1 = flash_attn(xq1, xk1, xv1)
  │     │
  │     └─ concat: output = cat([out0, out1])  → [S_cp, B, n_local_heads, head_dim]
  │        reshape → [S_cp, B, H/TP]
  │
  ├─ 4. wo (RowParallelLinear)
  │     ┌─ MatMul: [S_cp, B, H/TP] × W_o^T → [S_cp, B, H]  (每个 TP rank 的部分结果)
  │     │
  │     └─ ReduceScatter(TP): [S_cp, B, H] → [S_local, B, H]
  │        代码: layers.py:957-958
  │        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
  │
  ├─ 5. Residual Add: h = x + output  (无通信)
  │
  ├─ 6. FFNNorm (RMSNorm，无通信)
  │
  ├─ 7. w1 (ColumnParallelLinear, gate+up projection)
  │     ┌─ AllGather(TP): [S_local, B, H] → [S_cp, B, H]
  │     └─ MatMul → [S_cp, B, 2*FFN_H/TP]
  │
  ├─ 8. SiLU Activation (无通信)
  │
  ├─ 9. w2 (RowParallelLinear, down projection)
  │     ┌─ MatMul → [S_cp, B, H]
  │     └─ ReduceScatter(TP): [S_cp, B, H] → [S_local, B, H]
  │
  └─ 10. Residual Add → 输出 [S_local, B, H]
```

**前向通信汇总（每层）**：

| 通信操作 | 通信组 | 次数 | 数据量 |
|---------|--------|------|--------|
| AllGather(TP) | TP group | 2 次（wqkv, w1） | `S_local × B × H × (TP-1)` 每次 |
| ReduceScatter(TP) | TP group | 2 次（wo, w2） | `S_local × B × H × (TP-1)` 每次 |
| AllGather(CP) | CP group | 2 次（K 和 V） | `S_cp × B × n_kv/TP × d × (CP-1)` 每次 |

#### 4.3.3 反向通信流程（一个 TransformerBlock）

反向通信与前向**互为对偶**：前向的 AllGather 在反向变成 ReduceScatter，反向的 ReduceScatter 在前向变成 AllGather。

```
grad_output: [S_local, B, H]
  │
  ├─ 1. w2 backward (RowParallelLinear)
  │     ┌─ AllGather(TP) grad_output: [S_local, B, H] → [S_cp, B, H]
  │     │  代码: mappings.py:298-300, _ReduceScatterToSequenceParallelRegion.backward()
  │     │  前向是 ReduceScatter → 反向自动变 AllGather
  │     │
  │     └─ MatMul: grad_input, grad_weight
  │
  ├─ 2. w1 backward (ColumnParallelLinear)
  │     ┌─ grad_input = grad_output @ weight
  │     │
  │     ├─ ReduceScatter(TP, async): [S_cp, B, H] → [S_local, B, H]
  │     │  代码: layers.py:317-332
  │     │  ↑ 此通信与下方 grad_weight 计算并行
  │     │
  │     ├─ AllGather(TP, async): 恢复 saved_input 用于 grad_weight
  │     │  代码: layers.py:262-273
  │     │  ↑ 利用 CUDA_DEVICE_MAX_CONNECTIONS=1 保证先入队
  │     │
  │     └─ grad_weight = grad_output.T @ total_input  (等 AllGather 完成)
  │
  ├─ 3. wo backward (RowParallelLinear)
  │     ┌─ AllGather(TP) grad_output → [S_cp, B, H]
  │     └─ MatMul
  │
  ├─ 4. Attention backward
  │     ┌─ 2× FlashAttention backward (本地计算)
  │     │
  │     └─ ReduceScatter(CP) for K,V gradients
  │        代码: context_parallel.py:46-47, _GatherFromCPRegion.backward()
  │        前向 AllGather(CP) → 反向自动变 ReduceScatter(CP)
  │
  ├─ 5. wqkv backward (ColumnParallelLinear)
  │     ┌─ ReduceScatter(TP, async) grad_input
  │     ├─ AllGather(TP, async) saved_input
  │     └─ grad_weight 计算 (与上两项通信重叠)
  │
  └─ 最终 grad_input: [S_local, B, H]
```

**反向通信汇总（每层）**：

| 通信操作 | 通信组 | 次数 | 说明 |
|---------|--------|------|------|
| AllGather(TP) | TP group | 4 次 | 2× grad_output 恢复 + 2× saved_input 恢复 |
| ReduceScatter(TP) | TP group | 2 次 | wqkv 和 w1 的 grad_input |
| ReduceScatter(CP) | CP group | 2 次 | K 和 V 的梯度 |

#### 4.3.4 对偶关系总结

SP 和 CP 的通信对偶性都由 `torch.autograd.Function` 自动保证：

```python
# SP 对偶 (mappings.py)
class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    def forward(ctx, input_):   return _reduce_scatter_along_first_dim(input_)  # FWD: ReduceScatter
    def backward(ctx, grad):    return _gather_along_first_dim(grad)             # BWD: AllGather

# CP 对偶 (context_parallel.py)
class _GatherFromCPRegion(torch.autograd.Function):
    def forward(ctx, input_):   return _gather_along_first_dim(input_)           # FWD: AllGather
    def backward(ctx, grad):    return _reduce_scatter_along_first_dim(grad)      # BWD: ReduceScatter
```

#### 4.3.5 通信量对比：CP 的效率优势

以 hidden_size=H, n_kv_heads=n_kv, head_dim=d, 序列长度 S 为例，每层每个 rank 的通信量：

| 通信类型 | 每次数据量 | 前向+反向次数 | 总量 |
|---------|-----------|-------------|------|
| TP（AllGather + ReduceScatter） | `S/(CP×TP) × B × H × (TP-1)/TP` | 8 次 | `8 × S×B×H × (TP-1) / (CP×TP²)` |
| CP（AllGather + ReduceScatter） | `S/CP × B × n_kv/TP × d × (CP-1)/CP` | 4 次 | `4 × S×B×n_kv×d × (CP-1) / (TP×CP²)` |

**关键结论**：

- **TP 通信量与 H（hidden_size）成正比**，涉及完整的 hidden states
- **CP 通信量仅与 `n_kv × d`（KV 维度）成正比**，远小于 H（因为 GQA 中 `n_kv << n_heads`）
- 例如 Qwen3-8B: H=4096, n_kv=4, d=128 → `n_kv × d = 512`，仅为 H 的 1/8
- CP 用**少量 KV 通信**换取**序列长度的线性扩展**，这是 CP 设计的核心效率优势

#### 4.3.6 序列长度对齐约束

TP-SP + CP 联合使用时，打包序列长度必须对齐到 `TP × CP × 2` 的倍数：

```python
# exp/rl.py:158-160  PackedPPOSamples.from_samples()
num_pad = cu_seqlens[-1] % (TP * CP * 2)
if num_pad != 0:
    num_pad = TP * CP * 2 - num_pad
```

这个约束来自两层需求的叠加：

1. **CP 的 Balanced 互补分片**要求 `S % (2 × CP) == 0`：序列被分成 `2 × CP` 个 chunk，每个 CP rank 取首尾各一个
2. **SP 的均匀切分**要求切分后的子序列仍能被 TP 整除

合在一起即 `S % (TP × CP × 2) == 0`，对应 `scatter_to_balanced_cp_region` 中的断言：

```python
# context_parallel.py:132-134
assert S % (2 * cp_size * tp_size) == 0,
    f"size ({S}) should be divisible by 2 * context parallel size ({cp_size}) x sequence parallel size({tp_size})"
```

`cu_valid_sizes` 记录 padding 前的真实长度，下游的 loss 计算通过 mask 忽略 padding 部分。

---

## 五、MoE / EP（Expert Parallel）优化

### 5.1 Triton 融合核

**代码位置**：`steptronoss/model/optimizations/`

#### 5.1.1 路由融合

```python
# moe_routing/triton.py
triton_histogram()        # 高效计算 token 到 expert 的分布直方图
triton_index_compute()    # 计算散射索引
triton_index_scatter()    # token 预处理散射
```

使用 Triton 实现替代 PyTorch 原生操作，减少多次 kernel launch 的开销。通过 `@optimizable` 装饰器注册。

#### 5.1.2 散射-GEMM-聚集融合

```python
# grouped_gemm/triton.py
triton_routed_grouped_ffn_fused()  # 融合 scatter → Grouped GEMM → gather
```

MoE 前向传播的标准流程：
1. **Scatter**：将 token 按路由结果分配到各 expert 的缓冲区
2. **Grouped GEMM**：每个 expert 对自己的 token 做矩阵乘法
3. **Gather**：将各 expert 的输出按原始顺序聚合，加权求和

融合核将这三步**合并为一个 Triton kernel**，避免：
- 三次 kernel launch 的开销
- scatter 和 gather 的中间缓冲区分配
- 数据在 HBM 和 L2 cache 之间的多次搬运

#### 5.1.3 加权聚集优化

```python
# moe_gather/triton.py
triton_moe_weighted_gather()         # 高效加权聚集（带梯度支持）
_compute_grad_weight_chunked()       # 分块计算梯度权重，避免 OOM（512MB/块）
```

**分块梯度计算**避免大 batch 时一次性分配过大的临时 tensor 导致 OOM。

#### 5.1.4 Grouped GEMM 自动调参

```python
# grouped_gemm/triton.py
grouped_matmul_kernel()  # 支持多种块大小配置（64x64、128x128 等）
```

Triton 的 `@triton.autotune` 自动从多种 tile size 配置中选择最优的，适配不同的矩阵形状和 GPU 架构。

### 5.2 Auxiliary-Loss-Free Load Balancing

**代码位置**：`steptronoss/model/common/moe_block.py`

```python
# 在前向传播中维护 router_balance_bias
def update_router_balance_bias_per_gbs():
    """在每个梯度累积步骤后调用"""
    # 跟踪每个 expert 的本地 token 数量
    # 定期更新路由器偏置以实现负载均衡

maintain_float32_router_balance_bias()  # 在 bf16 中保持 fp32 精度的偏置
```

**传统方法 vs 免辅助损失方案**：

| 项目 | 传统方法（z-loss / aux loss） | 免辅助损失方案 |
|------|------|------|
| 实现 | 额外的 loss 项加入总 loss | 只在前向路由时加偏置 |
| 反向计算 | 需要额外的反向传播计算 | **零额外反向开销** |
| 超参 | 需要调 `aux_loss_coef` | 只需设 `router_bias_update_rate` |
| 效果 | 可能干扰主 loss | 不影响梯度信号 |

### 5.3 ETP/EP 分离的通信组

**代码位置**：`steptronoss/exp/base_exp.py:308-344`

```python
parallel_definition = {
    "TP":  "(p d t) -> (p d) t",        # Dense 层使用
    "EP":  "(p edp ep etp) -> (p edp etp) ep",   # Expert 并行
    "ETP": "(p edp ep etp) -> (p edp ep) etp",   # Expert 内张量并行
}
expert_tensor_parallel_size: int = Ref(".tensor_model_parallel_size")  # 默认等于 TP size
```

在模型层中的使用：

```python
# layers.py:703  ColumnParallelLinear
tp_world_size = PM.size_of("ETP") if use_moe else PM.size_of("TP")
self.output_size_per_partition = safediv(output_size, tp_world_size)

# layers.py:879  RowParallelLinear
tp_world_size = PM.size_of("ETP") if use_moe else PM.size_of("TP")
self.input_size_per_partition = safediv(input_size, tp_world_size)
```

**优势**：
- Dense 层和 MoE 层可以使用**不同的 TP size**（通过独立设置 `tensor_model_parallel_size` 和 `expert_tensor_parallel_size`）
- 例如 Dense 用 TP=8，MoE 用 ETP=4 + EP=2，灵活适配不同的通信/计算比
- Expert 参数标记 `param.expert_model_parallel = True`，使得优化器和梯度管理可以区分 Dense 参数和 Expert 参数

---

## 六、优化器层面优化

### 6.1 ZeRO-1 ReduceScatter + AllGather

**代码位置**：`steptronoss/optimizer/zero1_gradient_manager.py`

```python
# 梯度同步：reduce-scatter（每个 rank 只保留 1/DP 的梯度）
def _reduce_model_grads():
    # reduce-scatter 将梯度分散到各 rank
    ...

# 参数恢复：all-gather（更新后收集完整参数）
def _gather_model_params():
    # all-gather 从各 rank 收集更新后的参数
    ...
```

**与普通 AllReduce 对比**：

| 项目 | AllReduce | ZeRO-1 |
|------|----------|--------|
| 优化器状态 | 每个 rank 保存完整状态 | 每个 rank 只保存 1/DP |
| 梯度同步 | AllReduce | ReduceScatter |
| 参数恢复 | 不需要 | AllGather |
| 显存占用 | 全量 | 约 1/DP（优化器状态部分） |

### 6.2 分桶 + 连续梯度缓冲区

**代码位置**：`steptronoss/model/utils/comm_buffer.py`, `steptronoss/optimizer/base_gradient_manager.py`

```python
# 按 (dtype, allreduce_group, muon_signature) 分桶
class ParamBucketKey:
    dtype: torch.dtype
    allreduce_group: str
    muon_bucket_signature: str  # Muon 优化器的分组标识

# 构建连续梯度缓冲区
def build_grad_buffers():
    # 所有同桶参数的梯度存在一块连续内存中
    # 使用 balanced_list_split() 跨 DP rank 均匀分配，避免在参数边界处切割
    ...
```

**优势**：
- 一次 ReduceScatter/AllReduce 通信整个 bucket，**减少 NCCL kernel launch 次数**
- 连续内存避免碎片，**提升 NCCL 通信效率**（NCCL 对连续大块内存的吞吐更高）
- 平衡分配确保各 rank 的通信量接近

**双缓冲区视图**：

```python
# 单个 fp32 缓冲区的多个 dtype 视图（通过 untyped_storage()）
# 减少内存占用并改进缓存局部性
```

### 6.3 混合精度梯度管理

**代码位置**：`steptronoss/optimizer/gradient_manager.py`

```python
# 梯度累积到 main_grad
def _make_grad_accumulate_hooks(self, model):
    """自定义反向钩子：将梯度累积到 fp32 的 main_grad 缓冲区"""
    def backward_hook(param, *args):
        if param.grad is not None:
            param.main_grad.add_(param.grad.data)  # fp16/bf16 grad → fp32 main_grad
            param.grad = None  # 立即释放模型参数梯度以节省显存
    ...

# 维护 bf16 参数和 fp32 主副本
def replace_optimizer_with_fp32():
    """创建 fp32 主参数副本"""
    ...

def _copy_model_grads_to_fp32_params():
    """梯度类型转换：bf16 grad → fp32 main_grad"""
    ...
```

**关键点**：
- 模型参数保持 bf16（省显存 + 利用 Tensor Core）
- 梯度累积在 fp32 的 `main_grad` 中（保证精度）
- 优化器状态维护 fp32 主副本
- 每步反向传播后**立即释放 bf16 梯度**（`param.grad = None`），只保留 fp32 的 `main_grad`

### 6.4 Expert Parallel 梯度处理

**代码位置**：`steptronoss/optimizer/base_gradient_manager.py`

```python
def _process_expert_parallel_grads():
    """应用拓扑不变缩放"""
    # 对 EDP 缓冲区进行梯度缩放：
    gbuf.data *= tp_size / ep_size
    # 正确处理混合并行设置下的梯度规范化
```

**原理**：Expert 参数的梯度在 EDP（Expert Data Parallel）组内做 AllReduce，而 Dense 参数在 DP 组内做 AllReduce。由于 EP 和 TP 的组大小不同，需要**拓扑不变缩放**来保证最终梯度的量级一致。

### 6.5 Muon 优化器

**代码位置**：`steptronoss/optimizer/muon.py`

```python
class Muon(Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz"""

    def step(self):
        for p in group['params']:
            if p.ndim >= 2:
                # Newton-Schulz 迭代正交化动量
                G = momentum_buffer
                for _ in range(ns_steps):
                    A = G @ G.T
                    G = (3*G - A @ G) / 2  # Newton-Schulz 迭代
                p.data.add_(G, alpha=-lr)
            else:
                # 0D/1D 参数回退到 AdamW
                ...
```

**优化特性**：

1. **Newton-Schulz 正交化**：对 2D 参数（Linear 权重），通过正交化动量来改善更新方向的稳定性，允许使用更大的学习率
2. **参数特定策略**：2D 参数用 Muon，0D/1D 参数（bias、LayerNorm）用 AdamW
3. **学习率自动调整**：

```python
def adjust_ratio_for_muon():
    """按 sqrt(max(A,B)) 缩放以匹配 AdamW RMS"""
    # 自动适配不同 shape 的参数
```

---

## 七、模型层面优化

### 7.1 @optimizable 装饰器

**代码位置**：`steptronoss/utils/optimizable.py`

```python
@optimizable(alternatives={
    "flash-attn": FlashAttention,
    "flash-attn-3": FlashAttention3,
})
class AttentionCore(nn.Module):
    """纯 PyTorch SDPA 作为默认回退"""
    ...

# 运行时切换（在实验配置中）
def configure_optimizable(self):
    set_optimization(AttentionCore="flash-attn", default="torch_compile")
```

**机制**：

- `@optimizable` 装饰器为一个算子注册多个替代实现
- `set_optimization()` 在运行时选择使用哪个实现
- 支持 `"torch_compile"` 作为通用加速选项
- **用户不改模型代码，通过配置选择最优 kernel**

**已注册的可优化算子**：

| 算子 | 默认实现 | 可选实现 |
|------|---------|---------|
| `AttentionCore` | PyTorch SDPA | `flash-attn`, `flash-attn-3` |
| `histogram` | PyTorch | `triton` |
| `index_scatter` | PyTorch | `triton` |
| `moe_weighted_gather` | PyTorch | `triton` |
| `grouped_ffn` | PyTorch | `triton` (融合版) |

### 7.2 分层 Recompute 策略

**代码位置**：`steptronoss/model/decoder_model.py`

```python
class DecoderLLMConfig(Megatron3DParallelModelConfig):
    recompute: list[str] | bool = []
    # 可选值：'attention', 'attn_norm', 'feed_forward', 'ffn_norm'

    def pp_vp_allocation(self, abs_pp_rank):
        """为每个 PP stage 的每一层指定 recompute 策略"""
        return [{"recompute": True}] * self.num_layers    # 全层重算
        # 或者细粒度控制：
        # [{"recompute": ["attention"]}] * 18 +            # 前 18 层只重算 attention
        # [{"recompute": ["feed_forward"]}] * 18           # 后 18 层只重算 FFN
```

**层级控制能力**：

| 粒度 | 配置 | 效果 |
|------|------|------|
| 全关 | `recompute=False` 或 `recompute=[]` | 不重算，最大计算效率 |
| 全开 | `recompute=True` | 全部模块重算，最省显存 |
| 只重算 Attention | `recompute=["attention"]` | Attention 激活不保存 |
| 只重算 FFN | `recompute=["feed_forward"]` | FFN 激活不保存 |
| 逐层不同 | `pp_vp_allocation` 返回不同配置 | 对显存紧张的层开启，计算密集的层关闭 |

**与 recompute 配合的激活检查点**：

```python
# feed_forward.py:61-65
def forward(self, x, recompute=False, **kwargs):
    if recompute:
        return tensor_parallel.checkpoint(
            self._forward, self.distribute_saved_activations, x
        )
    else:
        return self._forward(x)
```

### 7.3 Distribute Saved Activations（激活分布式保存）

**代码位置**：`steptronoss/core/tensor_parallel/random.py`

```python
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        with torch.no_grad():
            outputs = run_function(*args)

        # 如果开启分布式激活保存
        if distribute_saved_activations:
            # 将激活 scatter 到 TP 组的各 rank
            # 每个 rank 只保存 1/TP 的激活
            ...

        ctx.save_for_backward(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors

        if ctx.distribute_saved_activations:
            # 反向时 all-gather 恢复完整激活
            ...

        # 重新前向计算
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)
        # 反向传播
        torch.autograd.backward(outputs, args)
```

**效果**：激活检查点保存的中间结果进一步切分到 TP 组内的各 rank，**激活显存进一步降为 1/TP**。代价是反向时需要额外的 AllGather 通信来恢复完整激活。

### 7.4 RNG 状态管理

**代码位置**：`steptronoss/core/parallel_state.py:168-219`, `steptronoss/core/tensor_parallel/random.py`

```python
# ParallelManager 中的 RNG 管理
class ParallelManager:
    def register_rng(self, name: str, seed: int, diff_across: list[str]):
        """注册一个命名的 RNG 状态，seed 根据并行维度偏移"""
        offset = 0
        stride = 1
        for pname in diff_across:
            offset += self.rank_in(pname) * stride
            stride *= self.size_of(pname)
        rng_seed = int(seed) + offset
        ...

    @contextmanager
    def use_rng(self, name: str):
        """上下文管理器：临时切换到指定的 RNG 状态"""
        raw_rng_state = _get_rng_state()
        _set_rng_state(self.rng_states[name])
        yield
        self.rng_states[name] = _get_rng_state()
        _set_rng_state(raw_rng_state)
```

**用途**：
- **TP Dropout 一致性**：同一 TP 组内的 rank 需要使用相同的 dropout mask，通过 `register_rng("tp", seed, diff_across=["DP"])` 确保
- **重计算一致性**：checkpoint recompute 时需要恢复前向时的 RNG 状态，否则 dropout 结果不一致
- **CudaRNGStatesTracker**：`fork()` 上下文管理器自动保存/恢复 RNG 状态

```python
# grouped_query_attention.py:201-209
if not self.sequence_parallel:
    with tensor_parallel.get_cuda_rng_tracker().fork():
        output = self.core_attention(xq, xk, xv, ...)
else:
    output = self.core_attention(xq, xk, xv, ...)
```

非 SP 模式下，Attention 的 dropout 需要 fork RNG 来保证 TP 组内一致性。SP 模式下每个 rank 处理不同的序列片段，不需要 fork。

---

## 八、总结对照表

| 优化手段 | 并行维度 | 优化类型 | 核心效果 |
|---------|---------|---------|---------|
| 异步 AllReduce + `CUDA_DEVICE_MAX_CONNECTIONS=1` | TP | 通算重叠 | 通信隐藏在 grad_weight 计算中 |
| Sequence Parallel（ReduceScatter 替代 AllReduce） | TP | 通信优化 + 显存优化 | 非 TP 区域计算量/显存减为 1/TP |
| Gradient Accumulation Fusion | TP | 计算优化 | GEMM + 累加合并为一个 fused kernel |
| MoE 激活分片保存 + 异步 AllGather 恢复 | TP (MoE) | 显存优化 + 通算重叠 | 激活存储减为 1/TP，恢复时通信与计算重叠 |
| SiLU Recompute Fusion（Pre-Recompute Function） | TP | 显存优化 | 省 ffn_hidden_size 大小的激活，重算开销极小 |
| Overlap P2P + waiter 回调模式 | PP | 通算重叠 | P2P 通信与前向/反向计算并行 |
| VPP Interleaved 1F1B | PP | Bubble 优化 | Bubble 率从 (p-1)/m 降到约 (p-1)/(m*v) |
| Deallocate Output Tensor | PP | 显存优化 | 释放中间 stage 激活数据，保留计算图 |
| 固定 Shape 跳过 Shape 通信 | PP | 通信优化 | 预训练场景省一轮 P2P shape 通信 |
| Activation CPU Offload | PP | 显存优化 | warmup 激活搬到 CPU pinned memory |
| Balanced CP 互补分片 | CP | 负载均衡 | causal attention 下各 CP rank 计算量均匀 |
| Triton 融合核（scatter → GEMM → gather） | MoE | 计算优化 | 减少 kernel launch + 省中间缓冲区显存 |
| Auxiliary-Loss-Free Load Balancing | MoE | 计算优化 | 省辅助损失的反向计算开销 |
| ETP/EP 分离通信组 | MoE | 通信优化 | Dense 和 MoE 可用不同 TP/EP size |
| ZeRO-1 + 分桶连续缓冲区 | 优化器 | 显存 + 通信优化 | 优化器状态 1/DP + 高效批量通信 |
| 混合精度梯度管理（bf16 参数 + fp32 main_grad） | 优化器 | 精度 + 显存 | 保精度的同时省显存 |
| Expert Parallel 拓扑不变梯度缩放 | 优化器 | 正确性 | 混合并行下梯度量级一致 |
| Muon 优化器（Newton-Schulz 正交化） | 优化器 | 收敛优化 | 更稳定的更新方向，支持更大学习率 |
| `@optimizable` 运行时选实现 | 全局 | 灵活性 | 零代码切换 PyTorch/FlashAttn/Triton |
| 分层 Recompute + `pp_vp_allocation` | 全局 | 显存优化 | 逐层/逐模块精细化显存-计算 trade-off |
| Distribute Saved Activations | 全局 | 显存优化 | 检查点激活分散到 TP 组，进一步降为 1/TP |
| RNG 状态管理（fork + 命名 RNG） | 全局 | 正确性 | TP dropout 一致性 + recompute 正确性 |
