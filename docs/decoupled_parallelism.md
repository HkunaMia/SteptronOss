# StepTronOSS 解耦并行方案技术文档

## 1. 概述

### 1.1 什么是解耦并行

解耦并行（Decoupled Parallelism）是 StepTronOSS 框架的核心设计理念，指**允许模型的不同模块（Attention 和 MoE）采用独立的并行策略**，为它们分配独立的并行组，并在各自的数据并行组内执行梯度缩减和缩放。

传统的大模型训练框架通常采用"一刀切"的并行策略（如统一的 TP+PP+DP），这在处理混合架构（Dense Attention + Sparse MoE）时会导致通信效率低下。StepTronOSS 的解耦方案允许：

- **Attention 模块**：使用 CP（Context Parallel）处理长序列，避免 TP 的频繁 AllReduce
- **MoE 模块**：使用 EP（Expert Parallel）扩展专家数量，每个 token 只激活少量专家
- **独立的梯度同步组**：注意力参数和专家参数分别在不同的并行组内同步

### 1.2 适用场景

本方案特别适用于以下场景：
- **大规模 MoE 模型**：数百个专家（如 288 个）需要分布式存储
- **长上下文训练**：128K 及以上序列长度
- **多节点集群**：8 节点以上，节点间通过高速网络（IB）连接

## 2. 配置参数详解

### 2.1 整体配置（以 Step-3.5-Flash 为例）

```python
class Step3p5FlashParallelConfig(ParallelConfig):
    tensor_model_parallel_size = 1          # TP = 1，Attention 不切分
    pipeline_model_parallel_size = 8        # PP = 8，8 个流水线阶段
    virtual_pipeline_model_parallel_size = 3 # VPP = 3，每个 Stage 3 个虚拟阶段
    context_parallel_size = 8               # CP = 8，序列并行度
    expert_model_parallel_size = 8          # EP = 8，专家并行度
    expert_tensor_parallel_size = 1         # ETP = 1，专家不切分
```

**硬件需求计算**：
```
Attention MP Size = PP × TP × CP = 8 × 1 × 8 = 64
MoE MP Size = PP × ETP × EP = 8 × 1 × 8 = 64
World Size = 64（8 节点 × 8 GPU）
```

### 2.2 各并行维度含义

| 维度 | 值 | 作用范围 | 说明 |
|:---:|:---:|:---:|:---|
| **TP** | 1 | - | 不使用张量并行，避免 Attention 的 AllReduce |
| **PP** | 8 | 跨 Node | 8 个流水线阶段，每个 Node 一个 Stage |
| **VPP** | 3 | Node 内部 | 每个 Stage 内 3 个虚拟阶段，交错执行减少气泡 |
| **CP** | 8 | Node 内部 | 8 张卡分割序列长度（16K tokens/卡） |
| **EP** | 8 | Node 内部 | 8 张卡分割专家（36 专家/卡） |
| **ETP** | 1 | - | 专家不切分 |

## 3. 模型并行策略

### 3.1 45 层网络的分配算法

模型使用 `list_split` 算法将 45 层分配到 24 个虚拟槽位（PP × VPP = 8 × 3 = 24）：

```python
def list_split(data, split):
    chunk_size = len(data) // split  # 45 // 24 = 1
    extra_data = len(data) % split   # 45 % 24 = 21
    # 前 3 个 chunk 各 1 层，后 21 个 chunk 各 2 层
    ...
```

**分配结果**：
- 3 个槽位：1 层（chunk 0, 1, 2）
- 21 个槽位：2 层（chunk 3-23）

### 3.2 Stage 划分（Node 级别）

```
Stage 0 (Node 0): VP0[0], VP1[13,14], VP2[29,30] = 5 层
Stage 1 (Node 1): VP0[1], VP1[15,16], VP2[31,32] = 5 层
Stage 2 (Node 2): VP0[2], VP1[17,18], VP2[33,34] = 5 层
Stage 3 (Node 3): VP0[3,4], VP1[19,20], VP2[35,36] = 6 层
Stage 4 (Node 4): VP0[5,6], VP1[21,22], VP2[37,38] = 6 层
Stage 5 (Node 5): VP0[7,8], VP1[23,24], VP2[39,40] = 6 层
Stage 6 (Node 6): VP0[9,10], VP1[25,26], VP2[41,42] = 6 层
Stage 7 (Node 7): VP0[11,12], VP1[27,28], VP2[43,44] = 6 层
```

**关键观察**：
- 层号不是连续的，而是**交错分散**在不同 Stage
- Layer 0 在 Node 0，Layer 1 在 Node 1，以此类推
- 每个 Stage 通过 VPP 在本地交错执行

### 3.3 VPP 交错分配

以 Stage 0 (Node 0) 为例：

```
Node 0 的 8 张卡 (GPU 0-7) 都持有以下层：
├── VP0: Layer 0          ← 运行时激活
├── VP1: Layer 13, 14     ← 运行时激活
└── VP2: Layer 29, 30     ← 运行时激活

同一时间只有 1 个 VP 是活跃的，8 张卡通过 CP/EP 并行计算
```

## 4. 通信组详解

### 4.1 CP 组（Context Parallel）

**组构成**：Node 内部的 8 张卡

```
CP Group 0 = [0, 1, 2, 3, 4, 5, 6, 7]     # Node 0
CP Group 1 = [8, 9, 10, 11, 12, 13, 14, 15] # Node 1
...
CP Group 7 = [56, 57, 58, 59, 60, 61, 62, 63] # Node 7
```

**通信操作**：
- **AllGather**：在 Ring Attention 中聚合 key/value
- **AllReduce**：序列并行后的梯度同步

**数据流**：
```
输入序列: [batch, 128K, hidden]
         ↓ scatter_to_balanced_cp_region
GPU 0:  [batch, 0:16K, hidden]
GPU 1:  [batch, 16K:32K, hidden]
...
GPU 7:  [batch, 112K:128K, hidden]

计算后 AllGather 同步
```

### 4.2 EP 组（Expert Parallel）

**组构成**：Node 内部的 8 张卡（与 CP 组相同，但逻辑不同）

```
EP Group 0 = [0, 1, 2, 3, 4, 5, 6, 7]  # Node 0 的 8 张卡
```

**专家分配**（以 Layer 0 为例，共 288 专家）：
```
GPU 0: 专家 0-35      (EP rank 0)
GPU 1: 专家 36-71     (EP rank 1)
GPU 2: 专家 72-107    (EP rank 2)
...
GPU 7: 专家 252-287   (EP rank 7)
```

**通信操作**：
- **All-to-All (Dispatch)**：根据 router 输出，将 token 发送到持有目标专家的 GPU
- **All-to-All (Combine)**：收集专家计算结果，返回原始位置

**通信流程**：
```python
# TokenDispatcher.dispatch()
token_expert_ranks = token_expert_ids // num_local_experts
# 确定每个 token 应该去哪个 EP rank

# All-to-All 发送
distnn.all_to_all(recv_hidden, send_hidden, group=PM.group_of("EP"))
```

### 4.3 PP 组（Pipeline Parallel）

**组构成**：跨 Node 的对应位置 GPU

```
PP Group 0 = [0, 8, 16, 24, 32, 40, 48, 56]   # 各 Node 的第 0 张卡
PP Group 1 = [1, 9, 17, 25, 33, 41, 49, 57]   # 各 Node 的第 1 张卡
...
PP Group 7 = [7, 15, 23, 31, 39, 47, 55, 63]  # 各 Node 的第 7 张卡
```

**通信操作**：
- **P2P Send/Recv**：相邻 Stage 间传递激活值（Forward）和梯度（Backward）

**通信路径**（以 GPU 0 为例）：
```
GPU 0 (Stage 0) → GPU 8 (Stage 1) → GPU 16 (Stage 2) → ... → GPU 56 (Stage 7) → GPU 0
```

**VPP 下的通信**：
每个 micro-batch 在每个 Stage 触发 3 次 P2P 通信（对应 VP0/VP1/VP2）：
```
VP0 计算完成 → P2P Send → 下一 Stage VP0 接收
VP1 计算完成 → P2P Send → 下一 Stage VP1 接收  
VP2 计算完成 → P2P Send → 下一 Stage VP2 接收
```

### 4.4 DP/EDP 组（Data Parallel）

**DP 组**（非专家参数）：
- 组构成：同 Stage 的 8 张卡
- 通信：AllReduce（标准数据并行）
- 包含参数：Attention (wqkv, wo), LayerNorm, Shared Expert, Router

**EDP 组**（专家参数）：
- 组构成：同 Stage 的 8 张卡
- 通信：ReduceScatter（ZeRO-1 优化）
- 包含参数：MoE Experts (w1, w2)
- **特殊处理**：梯度缩放 `gbuf *= tp_size / ep_size = 1/8`

## 5. 并行方案管理模式

StepTronOSS 通过 `ParallelManager`（PM）集中管理所有并行状态，提供统一的 API 接口和动态切换能力。

### 5.1 核心管理器 ParallelManager

PM 是**全局单例**，贯穿整个训练生命周期：

```python
# steptronoss/core/parallel_state.py
PM = ParallelManager()  # 全局单例

class ParallelManager:
    def __init__(self):
        # 当前激活的并行组（运行时状态）
        self.parallels: dict[str, ParallelGroups] = {}
        
        # 所有创建过的 mesh 缓存（key 是配置的字符串表示）
        self.all_parallels: dict[str, dict[str, ParallelGroups]] = {}
        
        # Mesh 切换栈（支持嵌套切换）
        self._stack: list[ParallelConfig] = []
        
        # 当前配置
        self._cur_cfg: ParallelConfig = None
```

**核心 API**：

```python
# 查询接口
PM.size_of("EP")      # 返回 EP 组的大小（如 8）
PM.rank_in("EP")      # 返回当前 rank 在 EP 组内的位置 (0-7)
PM.ranks_of("EP")     # 返回当前 EP 组的所有 ranks [0,1,2,3,4,5,6,7]
PM.group_of("EP")     # 返回 torch.distributed.ProcessGroup
PM.i_am("PP", 0)      # 判断当前 rank 是否是 PP 组的 rank 0
```

### 5.2 Mesh 的概念与表示

**Mesh** 是分布式训练中对 **"并行拓扑结构"** 的抽象，即将 world_size 个 GPU 按照并行维度进行逻辑排列形成的网格。

#### 5.2.1 声明式 Mesh 定义

StepTronOSS 使用 **einops rearrange** 模式定义并行组：

```python
# steptronoss/exp/base_exp.py
class ParallelConfig:
    parallel_definition = {
        "TP": "(p d t) -> (p d) t",     # 标准 TP
        "PP": "(p d t) -> (d t) p",     # PP 跨节点
        "CP": "(p d c t) -> (p d t) c", # 上下文并行
        "EP": "(p edp ep etp) -> (p edp etp) ep",  # 专家并行
        "EDP": "(p edp ep etp) -> (p ep etp) edp", # 专家数据并行
    }
```

**模式解析**（以 `EP: "(p edp ep etp) -> (p edp etp) ep"` 为例）：

```
输入维度: p(PP) × edp(EDP) × ep(EP) × etp(ETP) = 8 × 1 × 8 × 1 = 64
输出: 把 ep 维度独立出来，形成 8 个组，每组 8 个 rank

结果:
- EP Group 0 = [0,1,2,3,4,5,6,7]      # Node 0
- EP Group 1 = [8,9,10,11,12,13,14,15] # Node 1
...
```

对比 `PP: "(p d t) -> (d t) p"`：

```
PP=8, DP=8, TP=1
重排后: 8 个组，每组 8 个 rank
PP Group 0 = [0,8,16,24,32,40,48,56]  # 跨节点，各 Node 第 0 张卡
```

#### 5.2.2 运行时 Mesh 结构

```python
# PM.parallels 存储了所有并行组的 ProcessGroup
{
    "CP": ParallelGroups(
        groups=[PG_0, PG_1, ..., PG_7],  # 8 个 CP 组
        ranks=[[0-7], [8-15], ..., [56-63]],  # 每组的 ranks
        group=PG_current,  # 当前 rank 所属的组
        my_group_rank=0,   # 当前 rank 在组内的位置
    ),
    "PP": ParallelGroups(...),
    "EP": ParallelGroups(...),
    "EDP": ParallelGroups(...),
}
```

### 5.3 通信组的生命周期管理

```python
def _new_group(self, ranks: list[int], **kwargs):
    """全局缓存，相同 ranks 只创建一次 ProcessGroup"""
    key = str(list(ranks)) + str(kwargs.items())
    
    if key not in self._all_groups:
        # 真正创建 NCCL ProcessGroup
        self._all_groups[key] = torch.distributed.new_group(ranks, **kwargs)
    
    return self._all_groups[key]
```

**关键设计**：
- `_all_groups` 是**进程全局缓存**，不因 `set_mesh()` 切换而销毁
- 不同 `parallel_cfg` 如果产生相同的 ranks 分组，会**复用同一个 ProcessGroup**
- ProcessGroup 随进程生命周期存在，程序退出时由 `atexit` 清理

### 5.4 参数并行属性标记

解耦并行的关键是通过属性标记参数属于哪种并行：

```python
# 专家参数标记为 EP 参数
class GroupedExperts(nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id: int):
        self.w1 = torch.nn.Parameter(torch.empty(...))
        self.w1.expert_model_parallel = True  # ← 关键标记
        
        self.w2 = torch.nn.Parameter(torch.empty(...))
        self.w2.expert_model_parallel = True

# Attention 参数（默认，无标记）
class GroupedQueryAttention(nn.Module):
    def __init__(self, ...):
        self.wqkv = nn.Parameter(...)  # 无 expert_model_parallel 标记
        self.wo = nn.Parameter(...)    # 属于标准 DP 组
```

**梯度缓冲区分流**：

```python
def get_bucket_key_from_param_attrs(param: SteptronParameter):
    # 根据参数属性决定其所属的同步组
    if getattr(param, "expert_model_parallel", False):
        allreduce_group = "EDP"   # 专家参数 → EDP 组
    else:
        allreduce_group = "DP"    # 其他参数 → DP 组
    return ParamBucketKey(...)
```

结果生成两套独立的梯度缓冲区：
- **DP Buffer**：Attention, LayerNorm, Shared Expert 参数
- **EDP Buffer**：MoE Expert 参数 (w1, w2)

## 6. 动态 Mesh 切换

动态 Mesh 管理是 StepTronOSS 的高级特性，允许**同一进程在不同阶段使用不同的并行配置**。

### 6.1 设计动机

在实际训练场景中，经常需要临时切换并行策略：

1. **RL 训练**：Actor 和 Critic 使用不同的并行度
2. **训练/评估切换**：训练用完整并行，评估用单卡减少同步开销
3. **多任务学习**：不同任务适应不同的并行拓扑
4. **检查点保存/加载**：临时切换到统一配置进行 IO 操作

### 6.2 实现机制

#### 6.2.1 基础切换：set_mesh

```python
def set_mesh(self, parallel_cfg: ParallelConfig):
    self._cur_cfg = parallel_cfg
    identity = str(self._cur_cfg)  # 配置的字符串标识
    
    if identity not in self.all_parallels:
        # 首次使用此配置：创建所有 ProcessGroup
        self.parallels = {}
        parallel_dict = parallel_cfg.build_parallel()
        
        for name, ranks in parallel_dict.items():
            self.new_parallel(name, ranks)
        
        # 缓存到 all_parallels
        self.all_parallels[identity] = self.parallels
    
    # 切换到缓存的 parallels
    self.parallels = self.all_parallels[identity]
    return self.parallels
```

#### 6.2.2 上下文管理器：use_mesh

```python
@contextmanager
def use_mesh(self, parallel_cfg: ParallelConfig):
    # 1. 保存当前配置到栈
    self._stack.append(self._cur_cfg)
    
    # 2. 切换到新配置
    self.set_mesh(parallel_cfg)
    
    yield  # 执行 with 块内的代码
    
    # 3. 恢复之前的配置
    self.set_mesh(self._stack.pop(-1))
```

**使用示例**：

```python
# 初始状态：训练 mesh (PP=8, CP=8)
PM.set_mesh(train_cfg)

with PM.use_mesh(eval_cfg):  # eval_cfg: PP=1, CP=1
    # 此时：
    # - PM.size_of("PP") 返回 1
    # - PM.size_of("CP") 返回 1
    # - 模型通信使用 eval 的 ProcessGroup
    
    output = model(input)    # 使用单卡并行

# 退出上下文，自动恢复训练配置
# PM.size_of("PP") 恢复为 8
```

### 6.3 缓存与复用

**缓存机制可视化**：

```
第一次调用 set_mesh(train_cfg):
                    ┌─────────────────────┐
  build_parallel()  │  CP: [[0-7],[8-15]..]│
       ↓            │  PP: [[0,8,16..],..] │
  new_parallel()    │  EP: [[0-7],[8-15]..]│
       ↓            └─────────────────────┘
  all_parallels[    "PP=8,CP=8,EP=8,..." 
    "PP=8,CP=8..."  ] = self.parallels
  ] = parallels            ↓
                      self.parallels (当前激活)
                           ↓
                    PM.size_of("CP") → 8

第二次调用 set_mesh(train_cfg):
                    ┌─────────────────────┐
  identity 已存在    │   直接从缓存读取    │
       ↓            │   跳过创建步骤     │
  self.parallels = all_parallels["PP=8,CP=8..."]
                           ↓
                    PM.size_of("CP") → 8
```

### 6.4 嵌套切换支持

```python
# 支持多层嵌套
PM.set_mesh(mesh_a)

with PM.use_mesh(mesh_b):
    # 当前使用 mesh_b
    # _stack = [mesh_a]
    
    with PM.use_mesh(mesh_c):
        # 当前使用 mesh_c
        # _stack = [mesh_a, mesh_b]
        print(PM.size_of("PP"))  # mesh_c 的值
        
    # 自动恢复 mesh_b
    # _stack = [mesh_a]
    print(PM.size_of("PP"))  # mesh_b 的值
    
# 自动恢复 mesh_a
# _stack = []
```

### 6.5 实际应用场景

#### 场景 1：RL 训练中 Actor/Critic 不同并行

```python
class PPOLikeExp(BaseExp):
    actor_parallel_cfg: ParallelConfig = ParallelConfig
    critic_parallel_cfg: ParallelConfig = ParallelConfig
    
    def train_step(self):
        # Actor 前向（可能用更大的 CP）
        with PM.use_mesh(self.actor_parallel_cfg):
            action = actor.generate(state)
            actor_loss = actor.compute_loss(...)
            
        # Critic 评估（可能用更小的并行度）
        with PM.use_mesh(self.critic_parallel_cfg):
            value = critic.evaluate(state)
            critic_loss = critic.compute_loss(...)
```

#### 场景 2：训练与评估切换

```python
class MyExp(BaseExp):
    def train(self):
        # 训练使用完整并行
        PM.set_mesh(self.train_parallel_cfg)
        for batch in train_loader:
            loss = self.train_step(batch)
            
    def evaluate(self):
        # 评估切换到单卡（避免多卡同步开销）
        with PM.use_mesh(self.eval_parallel_cfg):
            # eval_cfg: PP=1, CP=1, TP=1
            for batch in eval_loader:
                output = model.generate(batch)
                metrics.update(output)
        # 自动恢复训练并行配置
```

#### 场景 3：负载均衡计算

```python
# MoE 模型中，有时需要临时切换到纯 DP 模式
def load_balance_regularizer(model):
    # 切换到 EDP=1，所有专家在所有卡上复制
    with PM.use_mesh(no_ep_mesh):
        # 计算负载均衡损失（需要看到所有专家）
        loss = compute_balance_loss(model)
    # 恢复 EP 模式
    return loss
```

### 6.6 完整状态转换图

```
初始状态
    │
    ▼
PM.initialize() ──────► torch.distributed 初始化
    │
    ▼
PM.set_mesh(cfg_a) ───► parallels = {CP: PG_a, PP: PG_a, ...}
    │                      all_parallels["cfg_a"] = parallels
    │
    ▼
代码执行中 ─────────────► PM.size_of("CP") 查询 cfg_a 的配置
    │
    ▼
with PM.use_mesh(cfg_b): ─┐
    │                     │
    ▼                     │
    _stack.append(cfg_a)  │
    set_mesh(cfg_b) ──────┼──► 创建/切换到 cfg_b 的 ProcessGroup
    │                     │     parallels = {CP: PG_b, ...}
    ▼                     │
    执行 with 块内代码 ─────┤     PM.size_of("CP") 查询 cfg_b 的配置
    │                     │
    ▼                     │
    set_mesh(_stack.pop())◄┘     恢复 cfg_a 的配置
    │                              parallels 重新指向 cfg_a 的组
    ▼
继续执行 ────────────────► PM.size_of("CP") 恢复为 cfg_a 的值
```

## 7. 数据流与执行流程

### 7.1 前向传播完整流程

以 Micro-batch 0 在 Node 0 的执行过程为例：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 0 (Node 0) 前向传播流程                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. VP0 阶段 (Layer 0)                                                      │
│     ├── 输入: [batch, 16K, hidden] (CP 分割后)                               │
│     ├── Attention (CP 并行)                                                  │
│     │   └── CP AllGather/Ring 通信                                          │
│     ├── MoE (EP 并行)                                                        │
│     │   ├── Router 计算                                                      │
│     │   ├── EP All-to-All (Dispatch) ← 发送 token 到 Node 内其他 GPU         │
│     │   ├── Expert 计算 (本地 36 专家)                                       │
│     │   └── EP All-to-All (Combine)  ← 收集结果                              │
│     └── PP P2P Send → Stage 1 (Layer 1 在 Node 1)                           │
│                                                                             │
│  2. VP1 阶段 (Layer 13, 14)                                                  │
│     ├── 相同流程（Attention + MoE）                                          │
│     └── PP P2P Send → Stage 1                                               │
│                                                                             │
│  3. VP2 阶段 (Layer 29, 30)                                                  │
│     ├── 相同流程（Attention + MoE）                                          │
│     └── PP P2P Send → Stage 1                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 VPP 调度与流水线交错

```
时间线 (4 个 micro-batches，VPP=3):

Node 0 (Stage 0):
Time:  1     2     3     4     5     6     7     8     9     10    11    12
       │     │     │     │     │     │     │     │     │     │     │     │
MB0:  [VP0] [VP1] [VP2] [VP0] [VP1] [VP2] ...
MB1:        [VP0] [VP1] [VP2] [VP0] [VP1] [VP2] ...
MB2:              [VP0] [VP1] [VP2] [VP0] [VP1] [VP2] ...
MB3:                    [VP0] [VP1] [VP2] [VP0] [VP1] [VP2] ...

说明:
- 同一时间点，不同 micro-batch 使用不同 VP
- 当 MB0 在 VP1 时，MB1 在 VP0，实现流水线填充
- 每个 VP 阶段后都有 PP P2P 通信
```

### 7.3 反向传播与梯度同步

```
反向传播流程:

Stage 7 → Stage 6 → ... → Stage 0 (反向传播)
   ↓
Stage 0 完成本地梯度计算
   ↓
梯度同步:
├── 非专家参数: DP AllReduce (CP 组内)
└── 专家参数:   EDP ReduceScatter (带 TP/EP 缩放)
   ↓
优化器更新 → 下一轮迭代
```

**梯度缩放逻辑**（关键解耦设计）：
```python
def _process_expert_parallel_grads(self):
    tp_size = PM.size_of("TP")  # = 1
    ep_size = PM.size_of("EP")  # = 8
    # 补偿 EP 和 TP 的拓扑差异
    for gbuf in self._grad_buffers:
        if bucket_key.allreduce_group == "EDP":
            gbuf.data *= tp_size / ep_size  # = 1/8
```

## 8. 解耦设计的优势

### 8.1 Attention 模块优化

**传统方案（TP=8）**：
- Attention 的 Q/K/V 权重被切分到 8 张卡
- 每次前向/反向都需要 TP AllReduce
- 长序列时通信量随序列长度线性增长

**解耦方案（TP=1, CP=8）**：
- Attention 权重每张卡完整副本
- 序列切分到 8 张卡并行处理
- 通信只在 CP 组内进行（Node 内部 NVLink）
- **优势**：通信量与序列长度无关，充分利用 NVLink 带宽

### 8.2 MoE 模块优化

**传统方案（与 Attention 同构）**：
- 专家被迫使用 TP，导致额外切分
- 路由和计算复杂度高

**解耦方案（EP=8, ETP=1）**：
- 专家按 ID 范围均匀分布（36 专家/卡）
- 每个 token 只激活 top-8 专家，通过 All-to-All 路由
- **优势**：专家数量可随 Node 数扩展，通信限制在 Node 内部

### 8.3 梯度同步解耦

```
传统方案：所有参数统一在 DP 组内 AllReduce
解耦方案：
├── Attention/Dense 参数: DP AllReduce
└── MoE Expert 参数:     EDP ReduceScatter + 拓扑缩放

优势：
1. 专家参数使用 ZeRO-1 优化，减少显存
2. 拓扑感知缩放（tp_size/ep_size）保证数值一致性
3. 避免 EP 引入的额外梯度缩放因子
```

## 9. 通信量分析

### 9.1 各通信操作量化

假设配置：
- Batch size = 1
- Sequence length = 128K
- Hidden size = 4096
- Experts = 288, Top-k = 8
- Dtype = bfloat16 (2 bytes)

| 通信类型 | 通信组大小 | 单次通信量 | 频率（per layer） | 总通信量（45层） |
|:---|:---:|:---:|:---:|:---:|
| **CP AllGather** | 8 | ~256 MB | 2-4 次/Attention | ~40 GB |
| **EP Dispatch** | 8 | ~32 MB | 1 次/MoE | ~1.4 GB |
| **EP Combine** | 8 | ~32 MB | 1 次/MoE | ~1.4 GB |
| **PP P2P** | 2 | ~128 MB | 3 次/Stage | ~3 GB |
| **DP AllReduce** | 8 | ~50% 参数量 | 1 次/step | ~一半参数 |
| **EDP ReduceScatter** | 8 | ~50% 专家参数量 | 1 次/step | ~一半专家参数 |

### 9.2 通信重叠策略

```
理想重叠（VPP 调度实现）:

时间 →

计算:  [====VP0====][====VP1====][====VP2====]
通信:           [PP0]      [PP1]      [PP2]
                ↑ 与下一个 VP 的计算重叠

实际实现中：
1. PP P2P 与下一个 micro-batch 的 VP 计算重叠
2. EP All-to-All 可以与 Attention 计算部分重叠（如果实现支持）
3. CP 通信在 Ring Attention 中天然与计算交错
```

## 10. 最佳实践与调优建议

### 10.1 配置选择指南

| 场景 | 推荐配置 | 说明 |
|:---|:---|:---|
| 128K 长序列 | CP=8, TP=1 | 避免 TP 的 AllReduce |
| 288+ 专家 | EP=8, ETP=1 | 专家均匀分布，Node 内通信 |
| 8 节点集群 | PP=8, VPP=3 | 每个 Node 一个 Stage |
| 更大规模 | 增加 EP/CP | 保持 TP=1，ETP=1 |

### 10.2 性能调优建议

**1. 内存优化**
```python
# 启用梯度检查点（选择性重计算）
model_cfg.recompute = ["attention", "feed_forward"]  # 重计算这些模块
```

**2. 通信优化**
```python
# 启用 P2P 通信重叠
model_cfg.overlap_p2p_comm = True

# 启用序列并行（如果需要）
model_cfg.tp_cfg.sequence_parallel = False  # CP=8 时通常关闭
```

**3. 负载均衡**
```python
# 启用辅助损失自由负载均衡
moe_cfg.enable_auxiliary_loss_free_load_balance = True
moe_cfg.router_bias_update_rate = 0.001
```

### 10.3 常见问题排查

**问题 1：训练发散**
- 检查 EDP 梯度缩放是否正确：`gbuf *= tp_size / ep_size`
- 验证 CP AllReduce 是否在 Attention 后正确同步

**问题 2：内存溢出**
- 减少 VPP 数量（减少同时存储的激活值）
- 启用 activation checkpointing
- 检查 EP 组的 expert 分配是否均衡

**问题 3：通信瓶颈**
- 监控 EP All-to-All 的耗时（通常是瓶颈）
- 确认 CP 通信只在 Node 内部
- 检查 PP P2P 是否跨节点（应该跨节点）

### 10.4 扩展性建议

**扩展到更多节点（如 16 节点）**：
```python
# 保持 CP=8, EP=8（Node 内部不变）
# 增加 PP=16（更多 Stage）
# 或增加 ETP=2（如果专家需要进一步切分）

新配置：
- PP = 16
- CP = 8
- EP = 8
- ETP = 1
- TP = 1
```

**关键原则**：
- 保持 TP=1，避免 Attention 的 AllReduce
- 保持 ETP=1，简化专家并行
- 优先扩展 PP（流水线）和 EP（专家数）

---

## 附录：通信组速查表

```
64 卡布局（8 Node × 8 GPU）

Node 0: GPU 0-7
Node 1: GPU 8-15
...
Node 7: GPU 56-63

CP/EP/DP/EDP 组（Node 内部）:
  Group 0: [0,1,2,3,4,5,6,7]
  Group 1: [8,9,10,11,12,13,14,15]
  ...

PP 组（跨 Node，对应位置）:
  Group 0: [0,8,16,24,32,40,48,56]  # 各 Node 第 0 张卡
  Group 1: [1,9,17,25,33,41,49,57]  # 各 Node 第 1 张卡
  ...

Stage 分配:
  Stage 0 (PP=0): GPU 0-7   (Node 0)
  Stage 1 (PP=1): GPU 8-15  (Node 1)
  ...
```
