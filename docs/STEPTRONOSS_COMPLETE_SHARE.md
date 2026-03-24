# SteptronOss：大规模MoE模型训练框架技术分享（完整版）

**文档版本**: 2026-03-24  
**基于**: Step 3.5 Flash论文 + StepTronOss代码库 (https://github.com/HkunaMia/SteptronOss)  
**分享时长**: 45-60分钟  
**目标读者**: 训练Infra工程师

---

## 目录

1. [背景与Step 3.5 Flash模型架构](#一背景与step-35-flash模型架构)
2. [SteptronOss框架架构设计](#二steptronoss框架架构设计)
3. [核心技术优化之一：解耦并行方案](#三核心技术优化之一解耦并行方案)
4. [核心技术优化之二：通信优化](#四核心技术优化之二通信优化)
5. [核心技术优化之三：Muon优化器与ZeRO-1 Resharding](#五核心技术优化之三muon优化器与zero-1-resharding)
6. [核心技术优化之四：算子融合](#六核心技术优化之四算子融合)
7. [核心技术优化之五：细粒度重计算](#七核心技术优化之五细粒度重计算)
8. [RLVR后训练流程](#八rlvr后训练流程)
9. [总结与最佳实践](#九总结与最佳实践)

---

## 一、背景与Step 3.5 Flash模型架构

### 1.1 模型定位

Step 3.5 Flash是由阶跃星辰发布的大语言模型，采用MoE(Mixture of Experts)架构，核心特点是**拥有强大的表达能力，但推理成本却只相当于一个11B的稠密模型**。

**模型规格**:
| 参数 | 值 |
|-----|---|
| 总参数 | 196B |
| 激活参数 | 11B |
| Transformer层数 | 45层 |
| 注意力头数 | 64头(每层) |
| KV头数 | 8头(GQA设计) |
| 隐藏层维度 | 4096 |
| Head维度 | 128 |
| 上下文窗口 | 128K |
| 词表大小 | 128,896 |

### 1.2 Hybrid Attention (S3F1)

Step 3.5 Flash最独特的架构设计是**混合注意力机制**。采用**S3F1设计**——3层Sliding Window Attention配合1层Full Attention交替排列。

**为什么这样设计？**

| 注意力类型 | 优势 | 劣势 | 适用场景 |
|-----------|------|------|---------|
| **Full Attention** | 捕捉长距离依赖，支持128K上下文 | 计算复杂度O(n²)，成本高 | 需要全局信息的层 |
| **Sliding Window Attention** | 只关注局部512个token，复杂度低 | 无法捕捉长距离依赖 | 局部特征提取 |

**S3F1模式**: 3层SWA + 1层FA交替，在保持长上下文能力的同时，显著降低训练和推理成本。

模型还引入了**Head-wise Gated Attention**，这是一种对注意力头的门控机制，让模型可以自适应地选择关注哪些注意力头。

### 1.3 MoE设计创新

**专家配置**:
- 64个专家(代码配置中显示288个，存在差异待核实)
- 每个token激活8个专家(top-k=8)

**关键创新**:

#### 1.3.1 Aux-Loss-Free负载均衡

传统MoE训练需要辅助损失(auxiliary loss)来强制负载均衡，但这会影响模型质量。Step 3.5 Flash的方案是：

```python
# 为每个专家维护一个偏置项
self.router_balance_bias: torch.FloatTensor

# 选择时用概率加偏置
biased_prob = gate_prob + self.router_balance_bias.unsqueeze(0)
topk_expert_ids = biased_prob.topk(self.moe_top_k).indices

# 但权重仍用原始概率
token_weights = gate_prob.gather(1, topk_expert_ids)
```

这样既保证了负载均衡，又不损害模型效果。

**偏置更新机制**:
```python
@staticmethod
def update_router_balance_bias_per_gbs(models):
    """每个global batch调用一次"""
    # 1. 收集每个专家的token数量
    # 2. 跨EP/ETP/EDP聚合
    dist.all_reduce(tokens_per_expert, group=PM.group_of("EP"))
    dist.all_reduce(tokens_per_expert, group=PM.group_of("ETP"))
    dist.all_reduce(tokens_per_expert, group=PM.group_of("EDP"))
    
    # 3. 更新bias
    average_tokens = tokens_per_expert.mean()
    offset = average_tokens - tokens_per_expert
    bias += sign(offset) * update_rate
```

#### 1.3.2 Sigmoid Router

使用Sigmoid替代Softmax，配合归一化实现更稳定的路由:

```python
if self.use_sigmoid_router:
    gate_prob = F.sigmoid(logits)
    # 需要显式归一化
    token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True)
else:
    gate_prob = F.softmax(logits, dim=1)
```

#### 1.3.3 Router精度保持

Router Bias保持FP32精度，确保路由稳定性:

```python
def maintain_float32_router_balance_bias(self):
    """确保bias为FP32"""
    if self.router_balance_bias.dtype != torch.float32:
        self.router_balance_bias.data = self.router_balance_bias.data.to(torch.float32)
```

#### 1.3.4 激活裁剪

针对特定层设置SwiGLU激活裁剪，防止数值溢出:

```python
# 配置示例
moe_cfg.expert_swiglu_limits = {43: 7.0, 44: 7.0}
moe_cfg.shared_expert_swiglu_limit = {44: 16.0}

# 实现
def activation(self, x, swiglu_limit=None):
    l, r = torch.chunk(x, 2, dim=-1)
    l = F.silu(l)
    if swiglu_limit is not None:
        l = l.clamp(min=None, max=swiglu_limit)
        r = r.clamp(min=-swiglu_limit, max=swiglu_limit)
    return l * r
```

### 1.4 多Token预测 (MTP-3)

模型采用MTP-3，即一次预测3个未来token，这提高了训练效率，也为推理加速提供了可能。

---

## 二、SteptronOss框架架构设计

### 2.1 框架定位

SteptronOss是一个**轻量级的大规模语言模型训练框架**。相比于TorchTitan这样的全功能框架，SteptronOss更专注于大规模训练的核心需求，同时保持模块化和可配置性。

**与TorchTitan对比**:

| 特性 | TorchTitan | StepTronOss |
|------|-----------|-------------|
| 配置方式 | TOML文件 | Python类配置 |
| 并行策略 | 依赖PyTorch DTensor | 自定义TP/PP/DP/CP/EP策略 |
| 支持工作流 | 主要聚焦预训练 | 原生支持SFT、RLVR、Pretrain |

### 2.2 代码结构

```
steptronoss/
├── core/                    # 并行和训练基础设施
│   ├── parallel_state.py    # ParallelManager实现
│   ├── pipeline_parallel/   # 流水线并行调度器
│   ├── tensor_parallel/     # 张量并行层
│   └── trainers/            # 各种Trainer实现
├── model/                   # 模型架构
│   ├── common/              # Attention、MoE、FFN等
│   ├── decoder_model.py     # 主模型
│   └── optimizations/       # 优化算子(Triton)
├── exp/                     # 实验配置核心
│   ├── base_exp.py          # 基础配置类
│   ├── ntp.py               # 预训练实验
│   ├── sft.py               # 监督微调
│   └── rl.py                # RLVR实验
├── data/                    # 数据加载
├── optimizer/               # 优化器和梯度管理
│   ├── muon.py              # Muon优化器
│   ├── zero1_gradient_manager.py  # ZeRO-1实现
│   └── base_gradient_manager.py
├── checkpointing/           # 检查点保存/加载
└── generation/              # 推理生成(vLLM集成)
```

### 2.3 configurize配置系统

configurize配置系统是StepTronOss的一大特色。它允许用Python类来声明配置，而不是写死板的配置文件。

**Ref引用机制**: 通过`"..."`可以向上引用2级配置，`".."`引用1级。这让复杂的嵌套配置变得简洁。

```python
class MyExpConfig(BaseExp):
    # 使用Ref引用上层配置
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")
```

### 2.4 核心架构：解耦并行

**核心理念**: 允许模型的不同模块(Attention和MoE)采用独立的并行策略。

传统大模型训练框架通常采用"一刀切"的并行策略，这在处理混合架构(Dense Attention + Sparse MoE)时会导致通信效率低下。

**SteptronOss方案**:
- **Attention模块**: 使用CP(Context Parallel)处理长序列，避免TP的频繁AllReduce
- **MoE模块**: 使用EP(Expert Parallel)扩展专家数量，每个token只激活少量专家
- **独立的梯度同步组**: 注意力参数和专家参数分别在不同的并行组内同步

---

## 三、核心技术优化之一：解耦并行方案

### 3.1 为什么需要解耦并行

传统大模型训练框架通常采用"一刀切"的并行策略，比如统一的TP+PP+DP。但在处理混合架构——比如Dense Attention加Sparse MoE时，这种统一策略会导致通信效率低下。

**为什么？** 因为Attention和MoE有截然不同的并行需求：
- **Attention**: 计算密集，适合用TP张量并行，但TP会带来频繁的AllReduce通信
- **MoE**: 特点是稀疏激活，每个token只访问少数专家，更适合用EP专家并行

如果我们强行用同一套并行策略，必然有一方是低效的。

### 3.2 解耦并行核心思想

SteptronOss的解耦并行方案允许不同模块采用独立的并行策略：
- Attention模块使用CP上下文并行处理长序列，避免TP的频繁AllReduce
- MoE模块使用EP专家并行扩展专家数量
- Attention参数和专家参数分别在独立的并行组内同步梯度

### 3.3 Step 3.5 Flash并行配置

以Step 3.5 Flash为例，配置是：

```python
class Step3p5FlashParallelConfig(ParallelConfig):
    tensor_model_parallel_size = 1          # TP = 1，Attention不切分
    pipeline_model_parallel_size = 8        # PP = 8，8个流水线阶段
    virtual_pipeline_model_parallel_size = 3 # VPP = 3
    context_parallel_size = 8               # CP = 8，序列并行度
    expert_model_parallel_size = 8          # EP = 8，专家并行度
    expert_tensor_parallel_size = 1         # ETP = 1，专家不切分
```

**硬件需求计算**:
```
Attention MP Size = PP × TP × CP = 8 × 1 × 8 = 64
MoE MP Size = PP × ETP × EP = 8 × 1 × 8 = 64
World Size = 64（8节点 × 8 GPU）
```

**各并行维度含义**:

| 维度 | 值 | 作用范围 | 说明 |
|:---:|:---:|:---:|:---|
| **TP** | 1 | - | 不使用张量并行，避免Attention的AllReduce |
| **PP** | 8 | 跨Node | 8个流水线阶段，每个Node一个Stage |
| **VPP** | 3 | Node内部 | 每个Stage内3个虚拟阶段，交错执行减少气泡 |
| **CP** | 8 | Node内部 | 8张卡分割序列长度（16K tokens/卡） |
| **EP** | 8 | Node内部 | 8张卡分割专家（36专家/卡） |
| **ETP** | 1 | - | 专家不切分 |

### 3.4 45层网络的分配算法

模型使用`list_split`算法将45层分配到24个虚拟槽位（PP × VPP = 8 × 3 = 24）：

```python
def list_split(data, split):
    chunk_size = len(data) // split  # 45 // 24 = 1
    extra_data = len(data) % split   # 45 % 24 = 21
    # 前3个chunk各1层，后21个chunk各2层
    ...
```

**分配结果**:
- 3个槽位：1层（chunk 0, 1, 2）
- 21个槽位：2层（chunk 3-23）

**Stage划分（Node级别）**:

```
Stage 0 (Node 0): VP0[0], VP1[13,14], VP2[29,30] = 5层
Stage 1 (Node 1): VP0[1], VP1[15,16], VP2[31,32] = 5层
Stage 2 (Node 2): VP0[2], VP1[17,18], VP2[33,34] = 5层
Stage 3 (Node 3): VP0[3,4], VP1[19,20], VP2[35,36] = 6层
Stage 4 (Node 4): VP0[5,6], VP1[21,22], VP2[37,38] = 6层
Stage 5 (Node 5): VP0[7,8], VP1[23,24], VP2[39,40] = 6层
Stage 6 (Node 6): VP0[9,10], VP1[25,26], VP2[41,42] = 6层
Stage 7 (Node 7): VP0[11,12], VP1[27,28], VP2[43,44] = 6层
```

**关键观察**:
- 层号不是连续的，而是交错分散在不同Stage
- Layer 0在Node 0，Layer 1在Node 1，以此类推
- 每个Stage通过VPP在本地交错执行

### 3.5 通信组详解

#### 3.5.1 CP组（Context Parallel）

**组构成**: Node内部的8张卡

```
CP Group 0 = [0, 1, 2, 3, 4, 5, 6, 7]     # Node 0
CP Group 1 = [8, 9, 10, 11, 12, 13, 14, 15] # Node 1
...
CP Group 7 = [56, 57, 58, 59, 60, 61, 62, 63] # Node 7
```

**通信操作**:
- **AllGather**: 在Ring Attention中聚合key/value
- **AllReduce**: 序列并行后的梯度同步

**数据流**:
```
输入序列: [batch, 128K, hidden]
         ↓ scatter_to_balanced_cp_region
GPU 0:  [batch, 0:16K, hidden]
GPU 1:  [batch, 16K:32K, hidden]
...
GPU 7:  [batch, 112K:128K, hidden]

计算后AllGather同步
```

#### 3.5.2 EP组（Expert Parallel）

**组构成**: Node内部的8张卡（与CP组相同，但逻辑不同）

```
EP Group 0 = [0, 1, 2, 3, 4, 5, 6, 7]  # Node 0的8张卡
```

**专家分配**（以Layer 0为例，共288专家）:
```
GPU 0: 专家0-35      (EP rank 0)
GPU 1: 专家36-71     (EP rank 1)
GPU 2: 专家72-107    (EP rank 2)
...
GPU 7: 专家252-287   (EP rank 7)
```

**通信操作**:
- **All-to-All (Dispatch)**: 根据router输出，将token发送到持有目标专家的GPU
- **All-to-All (Combine)**: 收集专家计算结果，返回原始位置

**通信流程**:
```python
# TokenDispatcher.dispatch()
token_expert_ranks = token_expert_ids // num_local_experts
# 确定每个token应该去哪个EP rank

# All-to-All发送
distnn.all_to_all(recv_hidden, send_hidden, group=PM.group_of("EP"))
```

#### 3.5.3 PP组（Pipeline Parallel）

**组构成**: 跨Node的对应位置GPU

```
PP Group 0 = [0, 8, 16, 24, 32, 40, 48, 56]   # 各Node的第0张卡
PP Group 1 = [1, 9, 17, 25, 33, 41, 49, 57]   # 各Node的第1张卡
...
PP Group 7 = [7, 15, 23, 31, 39, 47, 55, 63]  # 各Node的第7张卡
```

**通信操作**:
- **P2P Send/Recv**: 相邻Stage间传递激活值（Forward）和梯度（Backward）

**VPP下的通信**:
每个micro-batch在每个Stage触发3次P2P通信（对应VP0/VP1/VP2）

#### 3.5.4 DP/EDP组（Data Parallel）

**DP组**（非专家参数）:
- 组构成：同Stage的8张卡
- 通信：AllReduce（标准数据并行）
- 包含参数：Attention (wqkv, wo), LayerNorm, Shared Expert, Router

**EDP组**（专家参数）:
- 组构成：同Stage的8张卡
- 通信：ReduceScatter（ZeRO-1优化）
- 包含参数：MoE Experts (w1, w2)
- **特殊处理**: 梯度缩放 `gbuf *= tp_size / ep_size = 1/8`

### 3.6 并行方案管理模式

StepTronOSS通过`ParallelManager`（PM）集中管理所有并行状态，提供统一的API接口和动态切换能力。

#### 3.6.1 核心管理器ParallelManager

PM是全局单例，贯穿整个训练生命周期：

```python
# steptronoss/core/parallel_state.py
PM = ParallelManager()  # 全局单例

class ParallelManager:
    def __init__(self):
        # 当前激活的并行组（运行时状态）
        self.parallels: dict[str, ParallelGroups] = {}
        
        # 所有创建过的mesh缓存（key是配置的字符串表示）
        self.all_parallels: dict[str, dict[str, ParallelGroups]] = {}
        
        # Mesh切换栈（支持嵌套切换）
        self._stack: list[ParallelConfig] = []
        
        # 当前配置
        self._cur_cfg: ParallelConfig = None
```

**核心API**:

```python
# 查询接口
PM.size_of("EP")      # 返回EP组的大小（如8）
PM.rank_in("EP")      # 返回当前rank在EP组内的位置(0-7)
PM.ranks_of("EP")     # 返回当前EP组的所有ranks[0,1,2,3,4,5,6,7]
PM.group_of("EP")     # 返回torch.distributed.ProcessGroup
PM.i_am("PP", 0)      # 判断当前rank是否是PP组的rank 0
```

#### 3.6.2 Mesh的概念与表示

**Mesh**是分布式训练中对"并行拓扑结构"的抽象，即将world_size个GPU按照并行维度进行逻辑排列形成的网格。

**声明式Mesh定义**:

StepTronOSS使用**einops rearrange**模式定义并行组：

```python
# steptronoss/exp/base_exp.py
class ParallelConfig:
    parallel_definition = {
        "TP": "(p d t) -> (p d) t",     # 标准TP
        "PP": "(p d t) -> (d t) p",     # PP跨节点
        "CP": "(p d c t) -> (p d t) c", # 上下文并行
        "EP": "(p edp ep etp) -> (p edp etp) ep",  # 专家并行
        "EDP": "(p edp ep etp) -> (p ep etp) edp", # 专家数据并行
    }
```

#### 3.6.3 动态Mesh切换

StepTronOSS的ParallelManager提供了`switch_mesh`上下文管理器，可以在运行时动态切换当前活跃的并行组。

```python
with PM.switch_mesh("CP"):
    output = attention(x)
with PM.switch_mesh("EP"):
    output = moe(x)
```

这让我们可以在同一层内先以CP模式计算Attention，再切换到EP模式计算MoE，全程自动处理通信组的切换和RNG状态的保存恢复。

### 3.7 CP的Balanced Complementary Sharding

CP的实现采用了Balanced Complementary Sharding策略。128K序列被切成16个8K的chunk，8个CP rank每个取首尾各一个chunk。

**为什么这样设计？** 为了负载均衡——每个rank处理的计算量是相同的。

对比Ring Attention，Balanced Complementary Sharding的优势在于通信量更少。Ring Attention需要每个rank和相邻rank多次通信，而Balanced方案只需要AllGather和Reduce-Scatter，在节点内部通过NVLink可以高效完成。

---

## 四、核心技术优化之二：通信优化

### 4.1 关于论文中提到的通信优化

**重要说明**：在StepTronOSS相关论文中提到两种通信优化：

1. **感知结构的通信调度**: 将DP流量划分为节点内NVLink阶段和节点间RoCE阶段，进行流水线处理
2. **感知通信的rank放置**: 利用作业级通信配置文件在交换机间放置rank，减少跳数

**经代码库搜索，上述两种优化在当前开源代码中未找到具体实现**。可能原因：
- 属于内部优化，未开源
- 在基础设施/集群调度层面实现，而非训练框架代码
- 未来工作计划

本文档所述优化均为代码库中**已实现的优化手段**。

### 4.2 分桶连续梯度缓冲区

**核心概念**: 分桶连续梯度缓冲区是对传统分布式训练梯度同步机制的深度优化，解决了参数分散、通信次数多、内存碎片等问题。

**关键组件**:
- **ParamBucketKey**: 桶的标识符，按数据类型、通信组、特殊签名分桶
- **连续缓冲区**: 每个桶预分配大块连续GPU内存
- **双视图机制**: 同一块物理内存的FP32/BF16视图

**分桶策略**:

```python
# steptronoss/model/utils/comm_buffer.py:24-39
@dataclass(frozen=True)
class ParamBucketKey:
    dtype: torch.dtype              # 数据类型（bf16/fp32）
    allreduce_group: str = "DP"     # 通信组（DP/EDP）
    signature: int | str | None = None  # 额外标识（如Muon分组）
```

**分桶逻辑**:

```python
# steptronoss/model/utils/comm_buffer.py:68-82
def get_bucket_key_from_param_attrs(param: SteptronParameter):
    """根据参数属性决定其所属的桶"""
    if getattr(param, "expert_model_parallel", False):
        allreduce_group = "EDP"   # 专家参数 → EDP桶
    else:
        allreduce_group = "DP"    # 非专家参数 → DP桶

    signature = f"MUON:{getattr(param, 'is_muon_param', False)}"
    
    return ParamBucketKey(
        dtype=param.dtype,
        allreduce_group=allreduce_group,
        signature=signature,
    )
```

**连续缓冲区分配**:

```python
# steptronoss/model/utils/comm_buffer.py:120-171
def build_grad_buffers(module):
    for bucket_key, params in bucket_params.items():
        dp_size = PM.size_of(bucket_key.allreduce_group)
        
        # 平衡分配：确保每个DP rank分到的参数大小相近
        param_sizes = [p.numel() for p in params]
        splited_params_ids = balanced_list_split(param_ids, sizes=param_sizes, split=dp_size)
        
        # 分配连续内存
        num_elements_padded = max_split_size * dp_size
        grad_buffer = torch.zeros(num_elements_padded, dtype=torch.float32, device="cuda")
        
        # 创建同一块内存的BF16视图（零拷贝）
        param_buffer = torch.tensor(
            grad_buffer.untyped_storage(),
            dtype=bucket_key.dtype,
            device=grad_buffer.device,
        )
```

**内存布局**:

```
Bucket: (bf16, DP)

连续内存缓冲区（grad_buffer）:
┌─────────────────────────────────────────────────────────────────┐
│  param1  │  param3  │  param5  │      padding      │  param2  │  param4  │
│  (DP_0)  │  (DP_0)  │  (DP_0)  │                  │  (DP_1)  │  (DP_1)  │
│          │          │          │                  │          │          │
│◄───────── DP rank 0 负责 ─────────►│                  │◄──────── DP rank 1 负责 ─────────►│
└─────────────────────────────────────────────────────────────────┘

底层存储（共享）:
- fp32_buffer:  上述内存的fp32视图（优化器使用）
- dtype_buffer: 上述内存的bf16视图（模型使用）
```

### 4.3 双视图零拷贝机制

```python
# 分配grad buffer（fp32）
grad_buffer = torch.zeros(num_elements_padded, dtype=torch.float32, device="cuda")

# 创建param buffer（bf16），共享同一块物理内存
param_buffer = torch.tensor(
    grad_buffer.untyped_storage(),  # 共享底层存储
    dtype=bucket_key.dtype,          # bf16/fp16视图
    device=grad_buffer.device,
)
```

**效果**:
- `fp32_buffer[0:1000]`和`dtype_buffer[0:1000]`指向同一块GPU内存
- 优化器操作fp32视图，模型操作bf16视图
- **零拷贝**: 无需在fp32和bf16之间复制数据

### 4.4 梯度累积机制

```python
# steptronoss/optimizer/base_gradient_manager.py:227-238
def _make_param_hook(param):
    def param_hook(*unused):
        if param.grad is not None:
            # 将BF16梯度累加到FP32缓冲区
            param.main_grad.add_(param.grad.data)
            param.grad = None  # 立即释放BF16梯度
    return param_hook
```

**流程**:
1. 反向传播产生`param.grad`（BF16）
2. Hook自动触发，将梯度累加到`main_grad`（FP32视图）
3. 立即释放`param.grad`，只保留FP32的`main_grad`

### 4.5 高效的梯度同步

```python
# steptronoss/optimizer/zero1_gradient_manager.py:187-200
for bucket_key, buffer in self._grad_buffers.items():
    reduce_group = bucket_key.allreduce_group
    group = PM.group_of(reduce_group)
    rank = PM.rank_in(reduce_group)
    local_size = buffer.numel() // world_size

    # ZeRO-1：ReduceScatter替代AllReduce
    torch.distributed.reduce_scatter_tensor(
        output=buffer[rank * local_size : rank * local_size + local_size],
        input=buffer,
        group=group,
    )
```

### 4.6 DP/EDP分离分桶

**分离的必要性**: 在解耦并行架构中，模型有两种参数，使用不同的并行策略：

| 参数类型 | 并行组 | 通信域 | 说明 |
|---------|--------|--------|------|
| **非专家参数** | DP (Data Parallel) | 同Stage内所有GPU | Attention、LayerNorm、Shared Expert、Router |
| **专家参数** | EDP (Expert Data Parallel) | EP组内 | MoE Experts (w1, w2) |

**参数标记**:

```python
# steptronoss/model/common/moe_block.py
class GroupedExperts(nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id: int):
        self.w1 = torch.nn.Parameter(torch.empty(...))
        self.w1.expert_model_parallel = True  # 关键标记
        
        self.w2 = torch.nn.Parameter(torch.empty(...))
        self.w2.expert_model_parallel = True
```

**拓扑感知缩放**（关键差异）:

```python
# steptronoss/optimizer/base_gradient_manager.py:180-200
def _process_expert_parallel_grads(self):
    tp_size = PM.size_of("TP")  # 例如1
    ep_size = PM.size_of("EP")  # 例如8
    
    for bucket_key, gbuf in self._grad_buffers.items():
        if bucket_key.allreduce_group == "EDP":
            # 关键：对EDP桶进行TP/EP缩放
            gbuf.data *= tp_size / ep_size  # 例如1/8
```

### 4.7 通信重叠优化

**异步AllReduce与计算重叠**:

```python
# steptronoss/core/tensor_parallel/layers.py:311-313
# ColumnParallelLinear backward
if ctx.async_grad_allreduce and not ctx.use_moe:
    # 1. 启动异步AllReduce（非阻塞）
    handle = torch.distributed.all_reduce(
        grad_input, group=PM.group_of("TP"), async_op=True
    )
    
    # 2. 立即开始计算grad_weight（与AllReduce并行）
    grad_weight = grad_output.t() @ total_input  # GEMM计算
    
    # 3. 最后等待AllReduce完成
    handle.wait()
```

**环境变量要求**:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**P2P通信重叠**（Pipeline Parallel）:

```python
# VPP调度器中的重叠策略
for k in range(num_microbatches_remaining):
    fwd_waiter()   # 等上一轮的前向通信完成
    
    output_tensor = forward_step_helper(forward_k)
    
    # 启动前向通信（异步），不等完成
    input_tensor, fwd_waiter = p2p_comm.send_forward_recv_forward(
        ..., overlap_p2p_comm=True
    )
    
    bwd_waiter()
    input_tensor_grad = backward_step_helper(backward_k)
    
    # 启动反向通信（异步）
    output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
        ..., overlap_p2p_comm=True
    )
```

---

## 五、核心技术优化之三：Muon优化器与ZeRO-1 Resharding

### 5.1 背景与问题定义

在大规模语言模型训练中，**Muon优化器**与**ZeRO-1**存在根本性的兼容冲突：

| 组件 | 需求 | 冲突点 |
|------|------|--------|
| **Muon优化器** | 需要完整的2D梯度矩阵进行Newton-Schulz正交化 | 无法处理分片梯度 |
| **ZeRO-1** | 通过reduce-scatter将梯度分片到不同DP rank | 每个rank只有1/N的梯度 |

**传统解决方案（Megatron-LM）**:
- 使用All-Reduce替代Reduce-Scatter来恢复完整梯度
- **代价**: 通信量几乎翻倍（~2×）

### 5.2 核心创新

StepTronOSS采用**Rank-Major参数分配策略**，实现：
- **单次Reduce-Scatter**即可让每个rank获得其负责参数的**完整梯度**
- **通信量保持1×**（vs All-Reduce的2×）
- **混合策略**: 仅对专家参数应用此优化，非专家参数使用标准DP All-Reduce

### 5.3 Muon优化器原理

**Muon - MomentUm Orthogonalized by Newton-schulz**

Muon内部运行标准SGD-momentum，然后执行正交化后处理步骤，将每个2D参数的更新替换为最近的正交矩阵。

```python
class Muon(torch.optim.Optimizer):
    """
    Arguments:
        lr: 学习率。更新将有spectral norm为`lr`（0.02是良好默认值）
        momentum: 内部SGD使用的momentum（0.95是良好默认值）
        matched_adamw_rms: Muon设计匹配的AdamW更新RMS（0.2~0.4推荐）
        nesterov: 是否在内部SGD中使用Nesterov-style momentum（推荐）
        ns_steps: Newton-Schulz迭代次数（5可能总是足够的）
        {0, 1}-D params或未被标记为muon的参数将使用AdamW优化
    """
    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        run_ns_in_fp16=True,
        newtonschulz_fn="default",
    ):
```

**Newton-Schulz迭代**:

```python
# steptronoss/optimizer/_helpers.py:66-98
def zeropower(G, steps, run_ns_in_fp16, coeffs):
    """
    Zero-power orthogonalization via Newton-Schulz iteration.
    
    数学公式：
    G₀ = G / ||G||
    Gₖ₊₁ = a·Gₖ + b·Gₖ·Gₖᵀ·Gₖ + c·Gₖ·Gₖᵀ·Gₖ·Gₖᵀ·Gₖ
    
    其中a, b, c是预计算的系数，确保收敛到正交矩阵。
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

**Muon的step实现**:

```python
@torch.no_grad()
def step(self):
    for group in self.param_groups:
        if not group.get("is_muon_param", False):
            continue

        for p in group["params"]:
            # merge_op用于处理TP切分的参数
            merge_op: ReshapeOp = getattr(p, "merge_op", None)
            if merge_op is None:
                merge_op = Identity()

            g: torch.Tensor = p.grad
            assert g is not None

            # 更新muon momentum
            state: dict[str, torch.Tensor] = self.state[p]
            if "muon_buffer" not in state:
                state["muon_buffer"] = torch.zeros_like(g)
            buf = state["muon_buffer"]
            buf.mul_(momentum).add_(g)

            # 准备Newton-Schulz输入
            g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
            g = g.bfloat16() if self.run_ns_in_fp16 else g.float()

            # 关键：通过merge_op恢复完整梯度
            merged_g = merge_op.forward({"grad": g})
            for k in list(merged_g):
                v = merged_g[k]
                adjusted_lr = lr * adjust_ratio_for_muon(matched_adamw_rms, v.shape)
                merged_g[k] = self.newtonschulz_fn(v, steps=ns_steps) * -adjusted_lr

            # 通过merge_op的backward回到分片形式
            update = merge_op.backward(merged_g)["grad"]
            
            # 应用weight decay和update
            p.data.mul_(1 - lr * weight_decay)
            p.data.add_(update)
```

### 5.4 Rank-Major参数分配

**传统ZeRO-1（Megatron-LM）**:
```
参数梯度缓冲区 (4 DP ranks):
┌─────────────────────────────────────────────────────────────────┐
│  param1[0:1/4] │ param1[1/4:2/4] │ param1[2/4:3/4] │ param1[3/4:1] │
│  (DP0)         │  (DP1)          │  (DP2)          │  (DP3)        │
└─────────────────────────────────────────────────────────────────┘
After Reduce-Scatter:
DP0: [param1_sum[0:1/4]]  ← 只有1/4，无法做Newton-Schulz
```

**StepTronOSS Rank-Major**:
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

**贪心分配算法**:

```python
# steptronoss/utils/general.py:129-157
def balanced_list_split(data: list[T], sizes: list[int], split: int) -> list[list[T]]:
    """
    Greedy, order-agnostic split of `data` into `split` buckets.
    
    Items are processed from largest to smallest `size` and placed in the bucket
    whose score is minimal at that step.
    
    关键保证：每个参数完整地分配给某一个bucket (DP rank)
    """
    new_data: list[list[T]] = [[] for i in range(split)]
    new_size = np.zeros(split, dtype=int)
    new_cout = np.zeros(split, dtype=int)
    
    total_sizes = sum(sizes) if number_balance else 0
    
    # 从大到小排序，贪心分配给当前负载最小的rank
    for i in np.argsort(sizes)[::-1]:
        dst = (total_sizes * new_cout + new_size).argmin()
        new_data[dst].append(data[i])  # 整个参数分配给该rank
        new_size[dst] += sizes[i]
        new_cout[dst] += 1
    
    return new_data
```

### 5.5 与并行维度的协同

**并行维度定义**:

| 维度 | 缩写 | 说明 |
|------|------|------|
| Tensor Parallel | TP | 张量并行，切分Attention/FFN |
| Pipeline Parallel | PP | 流水线并行，切分层 |
| Data Parallel | DP | 数据并行，非专家参数 |
| Expert Parallel | EP | 专家并行，切分MoE专家 |
| Expert Data Parallel | EDP | 专家数据并行，专家参数梯度同步 |
| Expert Tensor Parallel | ETP | 专家张量并行 |

**关键：通过merge_op处理TP切分**:

```python
# Muon优化器中的关键代码
merge_op: ReshapeOp = getattr(p, "merge_op", None)
if merge_op is None:
    merge_op = Identity()

# 将分片的梯度gather成完整矩阵
g = g.bfloat16() if self.run_ns_in_fp16 else g.float()
merged_g = merge_op.forward({"grad": g})

# 对每个完整矩阵执行Newton-Schulz
for k in list(merged_g):
    v = merged_g[k]
    adjusted_lr = lr * adjust_ratio_for_muon(matched_adamw_rms, v.shape)
    merged_g[k] = self.newtonschulz_fn(v, steps=ns_steps) * -adjusted_lr

# 通过backward scatter回分片形式
update = merge_op.backward(merged_g)["grad"]
```

### 5.6 性能收益

根据论文数据：
- **端到端迭代时间减少**: ~5%
- **额外内存开销**: < 4GB（来自padding）

---

## 六、核心技术优化之四：算子融合

### 6.1 概述

框架通过以下策略实现Kernel级优化：
- **算子融合**: 将多个小算子融合为单个Kernel，减少Kernel启动开销和内存搬运
- **Triton自定义Kernel**: 使用Triton编写高性能融合算子
- **多后端支持**: 通过`@optimizable`装饰器支持多种实现后端（Triton/CUDA/PyTorch）

### 6.2 Grouped GEMM

**位置**: `steptronoss/model/optimizations/grouped_gemm/`

**功能**: 将多个专家的矩阵乘法分组批量执行，提高GPU利用率

**传统问题**: MoE有多个专家，传统方法是逐个专家做矩阵乘法，这会产生大量小kernel。

**Grouped GEMM解决方案**: 将多个专家的矩阵乘法融合为单个kernel，通过批次处理提高GPU利用率。

**接口定义**:

```python
# steptronoss/model/utils/moe_utils.py:435-452
@optimizable(
    alternatives={
        "nv_grouped_gemm": nv_grouped_gemm,      # CUTLASS实现(H100 Tensor Core)
        "triton_grouped_gemm": triton_grouped_gemm,  # Triton实现
        "function_imple": function_imple_grouped_gemm,  # PyTorch实现
    }
)
def grouped_gemm(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    """
    Args:
        mat_a_flat: [total_tokens, K] 展平的输入
        mat_b: [num_experts, K, N] 或 [num_experts, N, K] 专家权重
        batch_sizes: [num_experts] 每个专家的token数
        trans_b: 是否转置mat_b
    Returns:
        output: [total_tokens, N]
    """
```

**启用方式**:
```python
from steptronoss.utils.optimizable import set_optimization
set_optimization("grouped_gemm", "nv_grouped_gemm")
# 或
set_optimization("grouped_gemm", "triton_grouped_gemm")
```

### 6.3 MoE Scatter/Gather优化

**MoE Scatter**（Token分发）:

```python
# steptronoss/model/utils/moe_utils.py:402-420
@optimizable(
    alternatives={
        "triton": triton_moe_scatter,
    }
)
def moe_scatter(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """将token按专家顺序排列"""
    
# Triton Kernel: _moe_scatter_kernel
# - 并行处理每个(token, top_k)对
# - 支持无效索引(-1)的掩码处理
```

**MoE Weighted Gather**（带权结果收集）:

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
    """按权重收集专家输出"""
    
# Triton Kernel: _mritonMoEWeightedGather
# - 前向: _moe_weighted_gather_kernel
# - 反向: _moe_weighted_gather_grad_in_kernel (使用atomic_add)
```

### 6.4 MoE Routing索引计算

**Histogram**（专家负载统计）:

```python
# steptronoss/model/utils/moe_utils.py:85-129
@optimizable(
    alternatives={
        "triton": triton_histogram,
    }
)
def histogram(top_k_rank: torch.Tensor, expert_num: int) -> torch.Tensor:
    """统计每个专家分配的token数量"""
    
# Triton Kernel: _histogram_kernel
# - 使用tl.atomic_add并行统计
# - 自动处理无效索引(-1)
```

**Index Compute**（索引计算）:

```python
# steptronoss/model/utils/moe_utils.py:132-214
@optimizable(
    alternatives={
        "triton": triton_index_compute,
    }
)
def index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
    """计算稳定的scatter索引，保持原始顺序"""
    
# Triton Kernel: _count_per_block_kernel + _index_compute_kernel
# - 两阶段算法：先分块计数，再计算全局偏移
```

### 6.5 Routed Grouped FFN（端到端融合）

**位置**: `steptronoss/model/optimizations/routed_grouped_ffn/triton.py`

**功能**: 将MoE前向传播的多个操作融合为单个函数

**标准实现流程**:
1. Scatter tokens to experts
2. Grouped GEMM (w1)
3. Activation
4. Grouped GEMM (w2)
5. Gather and weighted sum

**融合实现**: 上述步骤融合为单个Triton kernel

```python
@optimizable(
    alternatives={
        "fused": triton_routed_grouped_ffn_fused,
    }
)
def routed_grouped_ffn(w1, w2, act, x, token_expert_ids, token_weights):
    """端到端MoE FFN计算"""
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

### 6.6 DeepEP融合通信

**位置**: `steptronoss/model/ep_dispatcher/deepep_dispatcher.py`

**功能**: 使用DeepEP的融合Kernel进行专家并行通信

```python
class DeepEPDispatcher:
    def __init__(self, ...):
        self._buffer = Buffer(
            group=self.group,
            num_nvl_bytes=num_nvl_bytes,      # NVLink buffer
            num_rdma_bytes=num_rdma_bytes,    # RDMA buffer
            low_latency_mode=True,
        )
    
    def dispatch(self, hidden_states, token_expert_ids, token_expert_weights):
        """分发token到各EP rank"""
        
    def combine(self, hidden_states):
        """收集各EP rank的结果"""
```

**配置启用**:
```python
from steptronoss.utils.optimizable import set_optimization
set_optimization(TokenDispatcher="deep_ep")
```

### 6.7 SwiGLU激活融合

**位置**: `steptronoss/model/common/feed_forward.py`

**功能**: 将SwiGLU激活与W2投影融合，减少中间结果存储

```python
class FeedForward(torch.nn.Module):
    def __init__(self, cfg: FeedForwardConfig, layer_id: int):
        self.fuse_activation_w2 = cfg.swiglu_recompute_silu_out_proj
        
        self.w1 = tensor_parallel.ColumnParallelLinear(...)
        self.w2 = tensor_parallel.RowParallelLinear(
            ...,
            custom_pre_recompute_function=(
                self.activation if self.fuse_activation_w2 else None
            ),
        )

    def forward(self, x, recompute=False, **kwargs):
        if self.fuse_activation_w2:
            # 融合路径：W1 → (存储激活值) → W2 (在W2内部执行SwiGLU)
            x = self.w1(x)[0]
            output = self.w2(x)[0]  # W2内部执行activation
        else:
            # 分离路径
            x = self.activation(self.w1(x)[0])
            output = self.w2(x)[0]
        return output
```

### 6.8 实现状态汇总

| 优化项 | 状态 | 位置 | 备注 |
|--------|------|------|------|
| FlashAttention | ✅ 完成 | `attention_core.py` | 支持FA2/FA3/SDPA |
| QK Norm + RoPE融合 | ⚠️ 未实现 | - | 配置预留，Kernel待开发 |
| Grouped GEMM | ✅ 完成 | `grouped_gemm/` | 3种后端 |
| MoE Scatter | ✅ 完成 | `moe_scatter/` | Triton Kernel |
| MoE Gather | ✅ 完成 | `moe_gather/` | Triton Kernel |
| MoE Routing | ✅ 完成 | `moe_routing/` | Histogram + Index |
| Routed FFN融合 | ✅ 完成 | `routed_grouped_ffn/` | 端到端融合 |
| DeepEP通信 | ✅ 完成 | `deepep_dispatcher.py` | 需安装DeepEP |
| SwiGLU + W2融合 | ✅ 完成 | `feed_forward.py` | PyTorch实现 |

---

## 七、核心技术优化之五：细粒度重计算

### 7.1 设计哲学

框架提供**模块化、可配置的检查点机制**，支持多级粒度：

1. **Submodule-level**: 切换单个组件（attention、FFN、normalization）
2. **Operation-level**: 操作内部的细粒度控制（SiLU融合、QK-Norm）
3. **Distributed storage**: TP感知激活分布，提高内存效率

### 7.2 配置层次

```python
DecoderLLMConfig.recompute (list[str] | bool)
    ├── "attention"      → Attention模块checkpointing
    ├── "attn_norm"      → Pre-attention LayerNorm checkpointing
    ├── "feed_forward"   → FFN/MoE模块checkpointing
    └── "ffn_norm"       → Pre-FFN LayerNorm checkpointing

AttentionConfig.recompute_qknorm_rope (bool)
    └── QK-Norm + RoPE recomputation

FeedForwardConfig.swiglu_recompute_silu_out_proj (bool)
    └── SiLU activation fusion recomputation
```

### 7.3 CheckpointFunction

**位置**: `steptronoss/core/tensor_parallel/random.py`

自定义autograd函数实现检查点机制：

```python
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, kwargs, *args):
        # 保存RNG状态用于确定性重计算
        ctx.rng_states = _get_all_rng_states()
        
        # 执行前向，不保存中间激活
        with torch.no_grad():
            outputs = run_function(*args, **kwargs)
        
        # 可选：跨TP rank分布激活
        if distribute_saved_activations:
            split_tensor_into_1d_equal_chunks(args[0])
        
        # 只保存输入（不保存中间激活）
        ctx.save_for_backward(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 恢复分布的激活（如需要）
        if ctx.distribute_saved_activations:
            gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
        
        # 恢复RNG状态用于确定性重计算
        _set_all_rng_states(*ctx.rng_states)
        
        # 重计算前向
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
        
        # 计算梯度
        torch.autograd.backward(outputs, grad_outputs)
        return grads
```

### 7.4 TransformerBlock集成

**位置**: `steptronoss/model/decoder_model.py`

基于配置的 conditional checkpointing：

```python
class TransformerBlock(nn.Module):
    def forward(self, x, cu_seqlens=None, max_seq_len=None, position_id=None, **kwargs):
        # Pre-attention normalization with optional checkpointing
        if self.training and "attn_norm" in self.recompute:
            attn_in = checkpoint(self.attention_norm, self.distribute_saved_activations, x)
        else:
            attn_in = self.attention_norm(x)
        
        # Attention with optional checkpointing
        if self.training and "attention" in self.recompute:
            h = x + checkpoint(
                self.attention,
                self.distribute_saved_activations,
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )
        else:
            h = x + self.attention(...)
        
        # Pre-FFN normalization with optional checkpointing
        if self.training and "ffn_norm" in self.recompute:
            ffn_in = checkpoint(self.ffn_norm, self.distribute_saved_activations, h)
        else:
            ffn_in = self.ffn_norm(h)
        
        # FFN/MoE with optional checkpointing (handled internally)
        out = h + self.feed_forward(ffn_in, recompute="feed_forward" in self.recompute)
        return out
```

### 7.5 SiLU激活融合（FFN优化）

**位置**: `steptronoss/model/common/feed_forward.py`

高级优化：将SiLU激活与输出投影融合：

```python
class FeedForward(torch.nn.Module):
    def __init__(self, cfg: FeedForwardConfig, layer_id: int):
        self.fuse_activation_w2 = cfg.swiglu_recompute_silu_out_proj
        
        self.w1 = tensor_parallel.ColumnParallelLinear(...)
        self.w2 = tensor_parallel.RowParallelLinear(
            ...,
            custom_pre_recompute_function=(
                self.activation if self.fuse_activation_w2 else None
            ),
        )

    def _forward(self, x):
        if self.fuse_activation_w2:
            # 存储w1输出（不是SiLU输出）- 节省内存
            x = self.w1(x)[0]
            output = self.w2(x)[0]  # SiLU在backward中通过custom_pre_recompute_function重计算
        else:
            x = self.activation(self.w1(x)[0])
            output = self.w2(x)[0]
        return output
```

**内存节省原理**:

```
标准SwiGLU:
  Forward: x → w1 → split → SiLU(gate) * up → w2 → output
  Store: SiLU(gate) * up (大激活值)

融合SwiGLU:
  Forward: x → w1 → (store this) → [SiLU + mul] → w2 → output
  Store: w1输出（更小，或从存储的输入重计算）
  Backward: 即时重计算SiLU
```

**效果**: ~30% FFN激活内存减少

### 7.6 MoE Block特殊处理

**位置**: `steptronoss/model/common/moe_block.py`

MoE的特殊处理：router不重计算（需要梯度），expert可以重计算：

```python
class MoEBlock(nn.Module):
    def forward(self, x: torch.FloatTensor, recompute: bool = False) -> torch.FloatTensor:
        # Router是NEVER checkpointed - 需要传播梯度来学习路由策略
        logits = self.gate(x)
        token_expert_ids, token_weights, aux_loss = self.forward_router(logits)
        
        # Expert计算CAN be checkpointed
        if PM.size_of("EP") > 1:
            if recompute:
                output = checkpoint(
                    self.forward_experts_ep,
                    self.recompute_dis_activation,
                    x, token_expert_ids, token_weights
                )
            else:
                output = self.forward_experts_ep(x, token_expert_ids, token_weights)
        else:
            if recompute:
                output = checkpoint(
                    self.forward_experts_tp,
                    self.recompute_dis_activation,
                    x, token_expert_ids, token_weights
                )
            else:
                output = self.forward_experts_tp(x, token_expert_ids, token_weights)
        
        return output
```

### 7.7 分布式激活存储

在Tensor Parallelism中，激活值被分布存储以节省内存：

```python
# Forward: split and save only local chunk
if distribute_saved_activations:
    ctx.input_0_shape = args[0].data.shape
    safely_set_viewless_tensor_data(
        args[0],
        split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
    )

# Backward: gather to restore full activation
if ctx.distribute_saved_activations:
    safely_set_viewless_tensor_data(
        inputs[0],
        gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
    )
```

**收益**: 激活内存减少`TP_size`倍每层

### 7.8 逐层配置示例

```python
class MyModelConfig(DecoderLLMConfig):
    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        """每层不同的重计算策略"""
        layers = [...]  # layer ids for this pp rank
        allocation = []
        for i, layer_id in enumerate(layers):
            if layer_id < 10:
                # 早期层：完整checkpointing
                allocation.append({"recompute": ["attention", "attn_norm", "feed_forward", "ffn_norm"]})
            elif layer_id < 40:
                # 中间层：选择性
                allocation.append({"recompute": ["attention", "feed_forward"]})
            else:
                # 晚期层：最小化
                allocation.append({"recompute": []})
        return allocation
```

### 7.9 与业界标准对比

| 特性 | StepTronOSS | Megatron-LM | PyTorch SAC |
|------|-------------|-------------|-------------|
| Submodule-level | ✅ String list | ✅ Predefined modes | ❌ |
| SiLU Fusion | ✅ `custom_pre_recompute_function` | ✅ Similar | ❌ |
| QK-Norm/RoPE | ✅ `recompute_qknorm_rope` | ❌ | ❌ |
| MoE-aware | ✅ Router excluded | ⚠️ Partial | ❌ |
| Distributed storage | ✅ `distribute_saved_activations` | ✅ | ❌ |
| Operator-level | ❌ | ❌ | ✅ Policy-based |
| Auto-tuning | ❌ | ⚠️ Manual config | ❌ |

---

## 八、RLVR后训练流程

### 8.1 整体架构概览

**三模型架构**:

```
┌─────────────────────────────────────────────────────┐
│                   RLVR训练系统                       │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Actor   │  │  Reference   │  │    Critic     │  │
│  │ (策略网络) │  │  (参考模型)   │  │  (价值网络)    │  │
│  │          │  │              │  │               │  │
│  │ 可训练    │  │  权重冻结     │  │  可训练       │  │
│  │ 生成动作  │  │  KL正则化   │  │  估计价值     │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
│       ↑              ↑                  ↑           │
│       │         从Actor初始化      从Actor初始化     │
└─────────────────────────────────────────────────────┘
```

**训练-推理分离的混合引擎**:

```
训练集群                                  推理集群
┌───────────────────────┐              ┌───────────────────────┐
│  PyTorch + Megatron    │   权重同步    │  vLLM Server(s)       │
│                        │ ──────────→ │  TP并行推理           │
│  Actor / Critic /      │  safetensors │                       │
│  Reference 前向反向    │   热部署     │  高吞吐decode         │
└───────────────────────┘              └───────────────────────┘
                                              ↑
                                       ┌──────┴──────┐
                                       │ VLLMRouter   │
                                       │ 负载均衡      │
                                       │ FastAPI      │
                                       └─────────────┘
```

### 8.2 核心数据结构

**EnvTrajectory**: 环境交互轨迹

```python
class EnvTrajectory(DataClass):
    trajectory: torch.LongTensor = None   # prompt + 生成的token ids
    logprobs: torch.Tensor = None         # 生成时的log-probabilities
    is_gen_mask: torch.BoolTensor = None  # True标记生成部分（vs prompt部分）
    raw_reward: float = None              # 环境返回的奖励信号
    meta: dict = None                     # 元信息
    stop_type: str = None                 # 停止原因：length / eos
```

**PPOSample**: PPO训练样本

```python
class PPOSample(EnvTrajectory):
    prompt_id: int = None
    ref_logprobs: torch.Tensor = None     # Reference模型的logprobs
    actor_logprobs: torch.Tensor = None   # Actor模型的logprobs
    advantages: torch.Tensor = None       # GAE计算的advantages
    returns: torch.Tensor = None          # GAE计算的returns
    values: torch.Tensor = None           # Critic估计的values
```

**PackedPPOSamples**: 打包批处理

```python
class PackedPPOSamples(DataClass):
    input_ids: torch.Tensor = None        # 连接后的所有token ids: [1, Total_S]
    cu_seqlens: torch.Tensor = None       # 累积长度（含padding）: [N+1]
    cu_valid_sizes: torch.Tensor = None   # 累积长度（不含padding）: [N+1]
    logprobs: torch.Tensor = None
    ref_logprobs: torch.Tensor = None
    actor_logprobs: torch.Tensor = None
    advantages: torch.Tensor = None
    returns: torch.Tensor = None
    values: torch.Tensor = None
```

### 8.3 PPOTrainer训练流程

```python
def train_step(self):
    # ═══ 阶段1：生成轨迹 ═══
    all_trajs = self.generate_trajectory()
    all_samples = self.adapt_trajs_to_samples(trajs)
    all_samples = all_gather_object(all_samples)

    # ═══ 阶段2：数据处理与打包 ═══
    all_samples = self.filter_samples(all_samples)
    my_samples = self.data_balance_and_pack(all_samples)

    # ═══ 阶段3：前向推理（不更新参数） ═══
    self.get_reference(my_samples)       # Reference前向 → ref_logprobs
    self.get_actor_logprob(my_samples)   # Actor前向 → actor_logprobs
    self.get_values_advantages(my_samples)  # Critic前向 → values → GAE

    # ═══ 阶段4：Critic训练 ═══
    for chunk in chunk_my_samples(my_samples, fix_iters_critic):
        self.critic.forward_backward(chunk, loss_fn=critic_loss_func)
        self.critic.optimizer_step()

    # ═══ 阶段5：Actor训练（跳过warmup期） ═══
    if iteration >= critic_warmup_iters:
        for chunk in chunk_my_samples(my_samples, fix_iters):
            self.actor.forward_backward(chunk, loss_fn=actor_loss_func)
            self.actor.optimizer_step()
```

### 8.4 Actor损失函数（PPO目标）

```python
def actor_loss_func(data: PackedPPOSamples, logits):
    # 1. 从模型logits计算新的logprobs
    logits = scatter_to_balanced_cp_region(logits)
    new_logprobs = -vocab_parallel_cross_entropy(logits / temperature, labels)[0]
    new_logprobs = gather_from_balanced_cp_region(new_logprobs)
    new_logprobs = new_logprobs[is_gen_mask]

    # 2. PPO Clipped Surrogate Loss
    old_logprobs = data.logprobs
    advantages = data.advantages.clamp(adv_min, adv_max)

    ratio = torch.exp(new_logprobs - old_logprobs)
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages
    pg_loss = -torch.min(surrogate1, surrogate2).mean()

    # 3. KL正则化
    ref_logprobs = data.ref_logprobs
    kl_loss = ref_kl_loss_coeff * (new_logprobs - ref_logprobs).mean()
    kl_penalty = ref_kl_penalty_coeff * (old_logprobs - ref_logprobs).mean()

    loss = pg_loss + kl_loss + kl_penalty
    return loss
```

### 8.5 Flow Control异步流控

**三层队列架构**:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   数据源      │     │  pre_gen     │     │  pre_train   │
│ (Dataloader)  │ ──→ │  Queue       │ ──→ │  Queue       │ ──→ 训练
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                         ↑                      ↑
                  _control_worker()      _generation_worker()
```

**三种调度策略**:

| 策略 | 时序 | 特点 |
|------|------|------|
| **on-policy** | `req(0), p0..pN, train, req(1), p0..pN, train, ...` | 每次训练前同步最新权重再生成，数据最新但慢 |
| **one-step-off** | `req(0), p0..pN, train, req(0), p0..pN, train, req(1), ...` | 使用当前权重训练，同时用新权重异步生成下一批 |
| **fully-async** | 持续异步生成，限制max_staleness | 最高吞吐，但数据可能陈旧 |

**权重同步机制**:

```python
def sync_weight(self):
    if self.infer_weight_version != self.train_weight_version:
        self.vllm_cfg.deploy_training_model(self.model)
        self.infer_weight_version = self.train_weight_version

# deploy_training_model内部流程：
# 1. dump_safetensors(hot_path, models) — 将训练模型权重写入快速存储路径
# 2. cli.wait_for_server() — 等待vLLM服务就绪
# 3. cli.reload_weights(hot_path) — 通过HTTP API触发vLLM热加载权重
```

### 8.6 内存优化：模型卸载与重载

**PackedModel三级卸载**:

```python
class PackedModel:
    _offloaded = {
        "params": False,          # 模型参数
        "grad_buffer": False,     # 梯度累积缓冲区
        "optimizer_state": False, # 优化器状态（momentum等）
    }

    def _offload_param(self):     # model.cpu()，释放梯度钩子
    def _backload_param(self):    # model.cuda()，重建梯度钩子

    def _offload_grad_buffer(self):   # 销毁梯度缓冲区（归零）
    def _backload_grad_buffer(self):  # 重建梯度缓冲区

    def _offload_optimizer_state(self):   # 优化器状态 → CPU
    def _backload_optimizer_state(self):  # 优化器状态 → GPU
```

**训练流程中的卸载编排**:

```
┌─ 阶段3: 前向推理 ────────────────────────────────────┐
│  actor.offload_model()         # Actor暂不用         │
│  reference.backload_model()    # 加载Reference       │
│  get_reference(samples)        # 计算ref_logprobs    │
│  reference.offload_model()     # 用完卸载            │
│  actor.backload_model()        # 加载Actor           │
│  get_actor_logprob(samples)    # 计算actor_logprobs  │
│  actor.offload_model()         # 用完卸载            │
│  critic.backload_model()       # 加载Critic          │
│  get_values_advantages(samples) # 计算values + GAE   │
└───────────────────────────────────────────────────────┘
```

**显存预算估算**:

对于参数量为N的BF16模型（单个角色）：

| 组成部分 | 显存 | 说明 |
|---------|------|------|
| 模型参数 | 2N | BF16存储 |
| 梯度累积缓冲区 | 4N | FP32累积 |
| 优化器状态 | 12N | FP32: param copy + momentum1 + momentum2 |
| **合计** | **18N** | 或 **18N / DP_size**（ZeRO-1） |

通过卸载编排，峰值显存约为18N（单个模型训练态），远低于同时加载三个模型的需求。

### 8.7 变长序列打包优化

**打包策略**:

RLVR中每条生成的轨迹长度各不相同。Packing将多条轨迹连接为一个长序列，通过`cu_seqlens`记录边界：

```python
@classmethod
def from_samples(cls, samples: list[PPOSample]) -> PackedPPOSamples:
    # 1. 连接所有labels / trajectories
    packed_data.labels = torch.cat(labels, 0)

    # 2. 计算累积长度
    seqlens = torch.tensor([len(s.trajectory) for s in samples])
    cu_seqlens = torch.cat([zeros(1), torch.cumsum(seqlens, 0)])

    # 3. 对齐填充到TP × CP × 2的倍数
    num_pad = cu_seqlens[-1] % (TP * CP * 2)
    if num_pad != 0:
        num_pad = TP * CP * 2 - num_pad

    # 4. 记录有效长度（padding前）和填充后长度
    packed_data.cu_valid_sizes = cu_seqlens.clone()
    cu_seqlens[-1] += num_pad
    packed_data.cu_seqlens = cu_seqlens
```

**TP × CP × 2对齐约束**:

打包后的总长度必须对齐到`TP × CP × 2`的倍数。这个约束来自两层需求的叠加：

1. **CP Balanced互补分片**: 序列被分成`2 × CP`个chunk，每个CP rank取首尾各一个 → 需要`S % (2 × CP) == 0`
2. **SP（Sequence Parallel）均匀切分**: 切分后的子序列需被TP整除

合在一起: `S % (TP × CP × 2) == 0`

---

## 九、总结与最佳实践

### 9.1 五大核心技术优化总结

| 优化 | 核心效果 | 关键技术 |
|-----|---------|---------|
| **解耦并行** | Attention与MoE使用最适合各自的并行策略 | CP/EP动态切换、Balanced Complementary Sharding、VPP调度 |
| **通信优化** | 高效梯度同步、隐藏通信延迟 | 分桶连续缓冲区、DP/EDP分离、双视图零拷贝、通信重叠 |
| **Muon+ZeRO-1** | 正交梯度下降 + 通信效率 | Rank-Major参数分配、单次Reduce-Scatter获取完整梯度 |
| **算子融合** | 减少Kernel启动开销和内存搬运 | Grouped GEMM、Routed FFN融合、Triton自定义Kernel |
| **细粒度重计算** | 灵活控制激活检查点 | Submodule级别控制、SiLU融合、分布式激活存储 |

### 9.2 推荐配置模板

**Step 3.5 Flash训练配置**:

```python
class Step3p5FlashParallelConfig(ParallelConfig):
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 8
    virtual_pipeline_model_parallel_size = 3
    context_parallel_size = 8
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1

class Step3p5FlashMoEConfig(MoEConfig):
    moe_num_experts = 288
    moe_top_k = 8
    moe_hidden_size = 1280
    share_expert_dim = 1280
    
    # 负载均衡优化
    enable_auxiliary_loss_free_load_balance = True
    router_bias_update_rate = 0
    enable_sigmoid_router = True
    norm_expert_weight = True
    
    # 激活裁剪
    expert_swiglu_limits = {43: 7.0, 44: 7.0}
    shared_expert_swiglu_limit = {44: 16.0}

def configure_optimizable(self):
    from steptronoss.utils.optimizable import set_optimization
    
    set_optimization(
        default="torch_compile",
        AttentionCore="flash-attn",  # 或 "flash-attn-3" (Hopper)
        TokenDispatcher="deep_ep",   # 需要安装DeepEP
        grouped_gemm="nv_grouped_gemm",  # 或 "triton_grouped_gemm"
        routed_grouped_ffn="fused",
        moe_scatter="triton",
        moe_weighted_gather="triton",
    )
```

### 9.3 性能调优建议

| 优化项 | 显存节省 | 速度提升 | 适用场景 |
|--------|---------|---------|---------|
| EP (Expert Parallel) | ★★★ | ★★★ | 必需，所有MoE场景 |
| Grouped GEMM | - | ★★★ | 专家数多、token数多 |
| DeepEP | - | ★★☆ | 多节点、NVLink环境 |
| Aux-Loss-Free | - | ★☆☆ | 追求模型质量 |
| 梯度检查点 | ★★★ | ★☆☆ | 显存紧张 |
| Routed FFN Fused | - | ★★☆ | 小batch、小专家 |
| Muon优化器 | - | ★★☆ | 需要更好的收敛性 |

### 9.4 关键文件索引

| 功能 | 文件路径 |
|------|---------|
| 并行状态管理 | `steptronoss/core/parallel_state.py` |
| MoE主实现 | `steptronoss/model/common/moe_block.py` |
| Muon优化器 | `steptronoss/optimizer/muon.py` |
| ZeRO-1梯度管理 | `steptronoss/optimizer/zero1_gradient_manager.py` |
| 通信缓冲区 | `steptronoss/model/utils/comm_buffer.py` |
| Grouped GEMM | `steptronoss/model/optimizations/grouped_gemm/` |
| RLVR训练 | `steptronoss/core/trainers/ppo_trainer.py` |
| Flow Control | `steptronoss/core/generators/flow_controller.py` |

### 9.5 参考资源

- **论文**: https://arxiv.org/pdf/2602.10604
- **代码仓库**: https://github.com/HkunaMia/SteptronOss
- **文档目录**: https://github.com/HkunaMia/SteptronOss/tree/dev/docs

---

*本文档由SteptronOss代码库文档整合而成，保留所有原始代码示例和技术细节。*
