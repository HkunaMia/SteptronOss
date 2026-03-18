# EP=4, ETP=2 混合并行通信流程详解

> 本文通过具体示例 (EP=4, ETP=2, 8 GPUs) 详细讲解 MoE 混合并行的通信过程。

---

## 1. 配置与 GPU 分布

### 1.1 配置参数

```python
parallel_cfg = {
    "expert_model_parallel_size": 4,      # EP = 4
    "expert_tensor_parallel_size": 2,     # ETP = 2
    # 总 GPUs = EP × ETP = 4 × 2 = 8
}

# 其他参数
num_experts = 8        # 8 个专家 E0-E7
top_k = 2              # 每个 token 选择 2 个专家
hidden_size = 4096     # 模型隐藏层维度
ffn_hidden = 11264     # FFN 中间层维度
```

### 1.2 GPU 拓扑结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    8 GPUs 拓扑 (EP=4, ETP=2)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   EP=0          EP=1          EP=2          EP=3               │
│  ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐              │
│  │GPU 0│      │GPU 2│      │GPU 4│      │GPU 6│              │
│  │ETP=0│      │ETP=0│      │ETP=0│      │ETP=0│              │
│  │E0[0]│      │E2[0]│      │E4[0]│      │E6[0]│              │
│  │E1[0]│      │E3[0]│      │E5[0]│      │E7[0]│              │
│  └──┬──┘      └──┬──┘      └──┬──┘      └──┬──┘              │
│     │            │            │            │                  │
│  ETP All-Reduce  ETP All-Reduce  ETP All-Reduce  ETP All-Reduce │
│     │            │            │            │                  │
│  ┌──┴──┐      ┌──┴──┐      ┌──┴──┐      ┌──┴──┐              │
│  │GPU 1│      │GPU 3│      │GPU 5│      │GPU 7│              │
│  │ETP=1│      │ETP=1│      │ETP=1│      │ETP=1│              │
│  │E0[1]│      │E2[1]│      │E4[1]│      │E6[1]│              │
│  │E1[1]│      │E3[1]│      │E5[1]│      │E7[1]│              │
│  └─────┘      └─────┘      └─────┘      └─────┘              │
│                                                                 │
│  All-to-All 通信发生在 EP 组之间 (GPU 0,2,4,6 或 GPU 1,3,5,7)   │
│  All-Reduce 通信发生在 ETP 组内部 (GPU 0-1, GPU 2-3, ...)       │
└─────────────────────────────────────────────────────────────────┘

参数说明:
- E0[0]: 专家 E0 的参数切片 0 (前 1/2)
- E0[1]: 专家 E0 的参数切片 1 (后 1/2)
- w1 形状: [num_local_experts, ffn_hidden*2/ETP, hidden_size]
- w2 形状: [num_local_experts, hidden_size, ffn_hidden/ETP]
```

---

## 2. 完整通信流程

以一个 batch 的前向传播为例，假设 batch=32, seq_len=1024，总 tokens = 32,768。

### 2.1 Step 0: 输入数据分布

```
所有 8 个 GPU 都有相同的输入:
- x: [32768, 4096] (bf16, ~256 MB)
- 每个 GPU 计算相同的 Router 输出
```

### 2.2 Step 1: Router 计算

```python
# 每个 GPU 独立计算 (数据并行)
logits = gate(x)  # [32768, 8]
token_expert_ids, token_weights = topk(logits, k=2)

# 假设 Token 0 (T0) 被路由到专家 E0 和 E2
# 假设 Token 1 (T1) 被路由到专家 E1 和 E3
# ...
```

**输出**:
- `token_expert_ids`: [32768, 2] (int64) - 每个 token 的 top-2 专家 ID
- `token_weights`: [32768, 2] (bf16) - 对应的权重

---

### 2.3 Step 2: Token Dispatch (All-to-All across EP)

这是最关键的一步，将 token 发送到目标专家所在的 EP 组。

#### 2.3.1 Dispatch 前的准备

```python
# 对于 GPU 0 (EP=0, ETP=0):
# 它负责专家 E0 和 E1

# 遍历所有 token，找出需要发给 EP=0,1,2,3 的 token
tokens_for_ep0 = []  # 目标专家 E0/E1 的 token
tokens_for_ep1 = []  # 目标专家 E2/E3 的 token
tokens_for_ep2 = []  # 目标专家 E4/E5 的 token
tokens_for_ep3 = []  # 目标专家 E6/E7 的 token

for token_id, (e1, e2) in enumerate(token_expert_ids):
    # Token 可能被路由到多个专家
    for expert_id in [e1, e2]:
        target_ep = expert_id // (num_experts // EP)  # EP 组
        if target_ep == 0:
            tokens_for_ep0.append((token_id, expert_id))
        elif target_ep == 1:
            tokens_for_ep1.append((token_id, expert_id))
        # ...
```

#### 2.3.2 All-to-All 通信

```
通信矩阵 (以 GPU 0,2,4,6 为例，ETP=0 组):

              发送到
           EP0    EP1    EP2    EP3
         ┌──────┬──────┬──────┬──────┐
   EP0   │  -   │ T→E2 │ T→E4 │ T→E6 │  GPU 0
         ├──────┼──────┼──────┼──────┤
   EP1   │ T→E0 │  -   │ T→E4 │ T→E6 │  GPU 2
来自     ├──────┼──────┼──────┼──────┤
   EP2   │ T→E0 │ T→E2 │  -   │ T→E6 │  GPU 4
         ├──────┼──────┼──────┼──────┤
   EP3   │ T→E0 │ T→E2 │ T→E4 │  -   │  GPU 6
         └──────┴──────┴──────┴──────┘

具体过程:
1. GPU 0 将目标为 E2,E3 的 token 发送给 GPU 2
2. GPU 0 将目标为 E4,E5 的 token 发送给 GPU 4
3. GPU 0 将目标为 E6,E7 的 token 发送给 GPU 6
4. 同时接收来自其他 GPU 的目标为 E0,E1 的 token

注意: GPU 1,3,5,7 执行相同的 All-to-All (ETP=1 组)
```

#### 2.3.3 通信后状态

```
以 GPU 0 (EP=0, ETP=0) 为例:

Dispatch 前:
- 本地 token: 32,768 个 (全部)

Dispatch 后:
- 本地 token: ~8,192 个 (目标专家为 E0/E1 的 token)
- 每个 token 可能包含 2 个条目 (因为 top_k=2)
- 实际计算量: 8,192 × 2 = 16,384 个 expert-computations

形状变化:
- x: [32768, 4096] → [~16384, 4096] (bf16)
- token_expert_ids: [32768, 2] → [~16384] (int64, 已展开)
- token_weights: [32768, 2] → [~16384] (bf16)
```

**通信量计算**:
```
每个 token 有 top_k=2 个专家选择
每个 token 的通信量 = 2 × hidden_size × sizeof(bf16) = 2 × 4096 × 2 = 16 KB

总通信量 (per GPU):
- 发送: 32768 tokens × 16 KB = 512 MB
- 接收: 32768 tokens × 16 KB = 512 MB
- 总计: ~1 GB All-to-All 通信

注意: 实际通信量取决于负载均衡情况
如果完美均衡，每个 EP 组处理 1/EP 的 tokens
```

---

### 2.4 Step 3: Expert 计算 (含 ETP All-Reduce)

现在 token 已经到达正确的 EP 组，需要进行 Expert 计算。由于 ETP=2，需要 All-Reduce。

#### 2.4.1 Grouped GEMM 计算

```python
# GPU 0 和 GPU 1 都有相同的输入 token (因为 ETP 组内广播)
# 但它们拥有同一专家的不同参数切片

# GPU 0 (ETP=0):
# w1: [2, 11264, 4096] → 实际存储 [2, 11264*2/2, 4096] = [2, 11264, 4096] 的前半
w1_gpu0 = w1[:, :11264, :]  # [2, 11264, 4096]
w2_gpu0 = w2[:, :, :11264]  # [2, 4096, 11264]

# GPU 1 (ETP=1):
# w1: [2, 11264, 4096] → 实际存储后半
w1_gpu1 = w1[:, 11264:, :]  # [2, 11264, 4096]
w2_gpu1 = w2[:, :, 11264:]  # [2, 4096, 11264]
```

#### 2.4.2 计算流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Expert 计算流程 (SwiGLU)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU 0 (ETP=0)              GPU 1 (ETP=1)                       │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ Input:       │          │ Input:       │                    │
│  │ [16384,4096] │          │ [16384,4096] │ (相同输入)         │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         ▼                         ▼                             │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ w1[:,:11264] │          │ w1[:,11264:] │                    │
│  │ @ x.T        │          │ @ x.T        │                    │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         ▼                         ▼                             │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ out1_part    │          │ out1_part    │                    │
│  │ [16384,11264]│          │ [16384,11264]│                    │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         └──────────┬──────────────┘                             │
│                    │                                            │
│         All-Reduce (ETP Group: GPU 0,1)                         │
│         out1 = out1_gpu0 + out1_gpu1                            │
│                    │                                            │
│         ┌──────────┴──────────────┐                             │
│         ▼                         ▼                             │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ Activation   │          │ Activation   │                    │
│  │ (SwiGLU)     │          │ (SwiGLU)     │                    │
│  │ [16384,11264]│          │ [16384,11264]│ (相同结果)         │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         ▼                         ▼                             │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ w2[:,:11264] │          │ w2[:,11264:] │                    │
│  │ @ act.T      │          │ @ act.T      │                    │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         ▼                         ▼                             │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ out2_part    │          │ out2_part    │                    │
│  │ [16384,4096] │          │ [16384,4096] │                    │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         └──────────┬──────────────┘                             │
│                    │                                            │
│         All-Reduce (ETP Group: GPU 0,1)                         │
│         out2 = out2_gpu0 + out2_gpu1                            │
│                    │                                            │
│         ┌──────────┴──────────────┐                             │
│         ▼                         ▼                             │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │ Final Output │          │ Final Output │                    │
│  │ [16384,4096] │          │ [16384,4096] │ (相同结果)         │
│  └──────────────┘          └──────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.4.3 通信量计算

```
每次 All-Reduce 通信量:
- 数据大小: 16384 tokens × 4096 hidden × 2 bytes (bf16) = 128 MB
- Ring All-Reduce 算法: 2×(N-1)/N × data = 2×(2-1)/2 × 128 MB = 128 MB

每层 MoE 有 2 次 All-Reduce:
1. w1 输出 All-Reduce: 128 MB
2. w2 输出 All-Reduce: 128 MB
总计: 256 MB (per ETP group)
```

---

### 2.5 Step 4: Token Combine (All-to-All across EP)

计算完成后，需要将结果返回到原来的 GPU。

#### 2.5.1 Combine 过程

```
这是 Dispatch 的逆过程:

GPU 0 (EP=0) 发送:
- 原本来自 GPU 2 (EP=1) 的 token 结果 → 返回给 GPU 2
- 原本来自 GPU 4 (EP=2) 的 token 结果 → 返回给 GPU 4
- 原本来自 GPU 6 (EP=3) 的 token 结果 → 返回给 GPU 6
- 保留原本属于自己的 token 结果

All-to-All 通信矩阵 (逆):
              发送到
           EP0    EP1    EP2    EP3
         ┌──────┬──────┬──────┬──────┐
   EP0   │  -   │ ◄──  │ ◄──  │ ◄──  │  GPU 0
         ├──────┼──────┼──────┼──────┤
   EP1   │ ◄──  │  -   │ ◄──  │ ◄──  │  GPU 2
         ├──────┼──────┼──────┼──────┤
   EP2   │ ◄──  │ ◄──  │  -   │ ◄──  │  GPU 4
         ├──────┼──────┼──────┼──────┤
   EP3   │ ◄──  │ ◄──  │ ◄──  │  -   │  GPU 6
         └──────┴──────┴──────┴──────┘
```

#### 2.5.2 结果还原

```python
# GPU 0 接收来自其他 EP 组的结果
# 使用 index_add 还原原始顺序

output = torch.zeros(32768, 4096, dtype=torch.bfloat16)

# 对于每个返回的 token
for token_id, expert_output in received_tokens:
    # 按 token_id 累加 (可能有多个专家的贡献)
    output[token_id] += expert_output * token_weight
```

**通信量**: 同 Dispatch (~1 GB)

---

## 3. 完整通信总结

### 3.1 单次 MoE 层前向传播

```
EP=4, ETP=2, batch=32, seq=1024

┌────────────────────────────────────────────────────────────────┐
│                        通信流程总结                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Step 2: Dispatch (All-to-All, EP 维度)                         │
│  ├─ 发送方: 每个 GPU 将 token 路由到目标 EP 组                  │
│  ├─ 接收方: 每个 EP 组接收目标专家对应的 token                   │
│  ├─ 通信量: ~1 GB (per GPU)                                     │
│  └─ 涉及 GPU: GPU 0,2,4,6 (ETP=0) 和 GPU 1,3,5,7 (ETP=1)       │
│                                                                │
│  Step 3: Expert Compute                                         │
│  ├─ Local: Grouped GEMM 计算                                   │
│  ├─ ETP All-Reduce #1 (w1 后)                                   │
│  │   ├─ 组内: GPU (0,1), (2,3), (4,5), (6,7)                   │
│  │   ├─ 通信量: 128 MB per group                               │
│  │   └─ 算法: Ring All-Reduce                                  │
│  │                                                              │
│  ├─ Local: SwiGLU Activation                                    │
│  ├─ ETP All-Reduce #2 (w2 后)                                   │
│  │   ├─ 组内: GPU (0,1), (2,3), (4,5), (6,7)                   │
│  │   └─ 通信量: 128 MB per group                               │
│  └─ 总计: 256 MB per GPU                                       │
│                                                                │
│  Step 4: Combine (All-to-All, EP 维度)                          │
│  ├─ 发送方: 每个 EP 组将计算结果返回给源 GPU                    │
│  ├─ 接收方: 每个 GPU 收集所有专家的贡献                          │
│  ├─ 通信量: ~1 GB (per GPU)                                     │
│  └─ 涉及 GPU: GPU 0,2,4,6 和 GPU 1,3,5,7                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  总通信量 (per GPU per layer):                                  │
│  ├─ All-to-All (Dispatch):     ~1024 MB                         │
│  ├─ All-Reduce (ETP #1):       ~128 MB                          │
│  ├─ All-Reduce (ETP #2):       ~128 MB                          │
│  ├─ All-to-All (Combine):      ~1024 MB                         │
│  └─ 总计:                      ~2304 MB (~2.25 GB)              │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 与纯 EP=8 (无 ETP) 的对比

```
配置对比:
┌─────────────────┬──────────────────┬──────────────────┐
│     配置        │    EP=8, ETP=1   │    EP=4, ETP=2   │
├─────────────────┼──────────────────┼──────────────────┤
│ EP 组数         │       8          │       4          │
│ 每 EP 专家数    │     8/8=1        │     8/4=2        │
│ ETP 大小        │       1          │       2          │
│ 每卡专家参数    │    100%          │     50%          │
├─────────────────┼──────────────────┼──────────────────┤
│ All-to-All      │    ~2 GB         │    ~1 GB         │
│   (通信量)      │   (更多 EP 组)    │   (更少 EP 组)    │
├─────────────────┼──────────────────┼──────────────────┤
│ ETP All-Reduce  │       0          │    ~256 MB       │
│   (通信量)      │                  │                  │
├─────────────────┼──────────────────┼──────────────────┤
│ 总通信量        │    ~2 GB         │   ~2.25 GB       │
├─────────────────┼──────────────────┼──────────────────┤
│ 显存节省        │     一般         │      更好        │
│ (每卡专家参数)  │                  │                  │
└─────────────────┴──────────────────┴──────────────────┘

结论:
- ETP 增加了 All-Reduce 开销
- 但允许更大的专家模型 (专家参数量可以 ×ETP)
- 当单专家太大时使用 ETP，否则优先增大 EP
```

---

## 4. 代码实现关键点

### 4.1 Dispatch 实现

```python
# steptronoss/model/ep_dispatcher/token_dispatcher.py

def dispatch(self, hidden_states, token_expert_ids, token_expert_weights):
    # 1. 计算每个 token 应该发送到哪个 EP rank
    token_expert_ranks = token_expert_ids // self.num_local_experts
    
    # 2. 按目标 rank 分组
    for rank in range(self.world_size):  # EP world size
        this_rank = (token_expert_ranks == rank).any(dim=1)
        send_hidden.append(hidden_states[this_rank])
        send_indices.append(token_expert_ids[this_rank])
        send_probs.append(token_expert_weights[this_rank])
    
    # 3. All-to-All 通信
    recv_hidden = distnn.all_to_all(send_hidden, send_hidden, group=self.group)
    recv_indices = distnn.all_to_all(send_indices, send_indices, group=self.group)
    recv_probs = distnn.all_to_all(send_probs, send_probs, group=self.group)
    
    return recv_hidden, recv_indices, recv_probs
```

### 4.2 ETP All-Reduce 实现

```python
# steptronoss/model/common/moe_block.py:168

def forward(self, x: torch.FloatTensor, token_expert_ids, token_weights):
    # Grouped GEMM 计算
    x = routed_grouped_ffn(
        self.w1, self.w2, self.activation,
        x, token_expert_ids, token_weights
    )
    
    # 关键: 在 ETP 组内 all-reduce
    # group="ETP" 表示只在 ETP 组内通信 (如 GPU 0,1)
    x = reduce_from_tensor_model_parallel_region(x, group="ETP")
    return x
```

### 4.3 Combine 实现

```python
# steptronoss/model/ep_dispatcher/token_dispatcher.py

def combine(self, hidden_states):
    # 使用 dispatch 时保存的通信记录
    send_sizes, recv_sizes, restore_S, token_ids = self.comm_record
    
    # All-to-All 反向通信
    recv_hidden = distnn.all_to_all(send_hidden, send_hidden, group=self.group)
    
    # 按 token_id 还原并累加
    output = hidden_states.new_zeros((restore_S, hidden_states.size(1)))
    output.index_add_(0, recv_token_ids, recv_hidden)
    return output
```

---

## 5. 可视化总结

```
完整数据流图:

Input [32768, 4096] on all GPUs
        │
        ▼
┌───────────────┐
│   Router      │ (All GPUs 相同)
│  [32768, 8]   │
└───────┬───────┘
        │
        ▼
Top-k Selection [32768, 2]
        │
        ▼
┌───────────────┐
│   Dispatch    │ All-to-All (EP=4)
│  [~16384, 2]  │ GPU 0◄──►GPU 2,4,6
└───────┬───────┘
        │
        ▼
Expert Input [~16384, 4096] per EP group
        │
        ├──► GPU 0 (ETP=0) ──┐
        │   w1[:,:11264]     │
        │                    │ ETP All-Reduce
        └──► GPU 1 (ETP=1) ──┤ (GPU 0,1)
            w1[:,11264:]     │
                               │
        ◄──────────────────────┘
        │
        ▼
Activation [~16384, 11264]
        │
        ├──► GPU 0 (ETP=0) ──┐
        │   w2[:,:11264]     │
        │                    │ ETP All-Reduce
        └──► GPU 1 (ETP=1) ──┤ (GPU 0,1)
            w2[:,11264:]     │
                               │
        ◄──────────────────────┘
        │
        ▼
Expert Output [~16384, 4096]
        │
        ▼
┌───────────────┐
│   Combine     │ All-to-All (EP=4)
│  [32768, 4096]│ GPU 0◄──►GPU 2,4,6
└───────┬───────┘
        │
        ▼
Final Output [32768, 4096] on all GPUs
        │
        ▼
Add to Residual
```

---

> 文档版本: 2025-03
> 详细解释了 EP=4, ETP=2 的完整通信流程
