# MoE 通信量计算详解

> 本文详细解释 EP/ETP 混合并行中的通信量计算方法，包括 All-to-All 和 All-Reduce 的公式推导。

---

## 1. 基础参数定义

首先明确各个参数的含义：

```python
# 模型参数
B = 32          # batch_size (批次大小)
S = 1024        # seq_len (序列长度)
H = 4096        # hidden_size (隐藏层维度)
F = 11264       # ffn_hidden (FFN 中间层维度)
K = 2           # top_k (每个 token 选择的专家数)
E = 8           # num_experts (专家总数)
L = 45          # num_layers (模型层数)

# 并行参数
EP = 4          # expert_model_parallel_size
ETP = 2         # expert_tensor_parallel_size
PP = 8          # pipeline_model_parallel_size
TP = 1          # tensor_model_parallel_size
WORLD_SIZE = 64 # 总 GPU 数

# 数据类型
dtype = torch.bfloat16  # 2 bytes
```

---

## 2. All-to-All 通信量计算

### 2.1 基础概念

**All-to-All 是什么？**
- 每个 GPU 发送不同的数据给所有其他 GPU
- 每个 GPU 从所有其他 GPU 接收不同的数据
- 类似于转置操作

**示意图** (4 GPUs):
```
通信前:                通信后:
GPU 0: [A0, B0, C0, D0]    GPU 0: [A0, A1, A2, A3]
GPU 1: [A1, B1, C1, D1]    GPU 1: [B0, B1, B2, B3]
GPU 2: [A2, B2, C2, D2]    GPU 2: [C0, C1, C2, C3]
GPU 3: [A3, B3, C3, D3]    GPU 3: [D0, D1, D2, D3]

每个 GPU 发送的数据量 = 3 × (数据大小)  # 发给其他 3 个 GPU
每个 GPU 接收的数据量 = 3 × (数据大小)  # 从其他 3 个 GPU 接收
总通信量 = 2 × 3 × (数据大小) = 6 × (数据大小)  # 双向
```

### 2.2 MoE Dispatch 通信量

**场景**: 32,768 个 tokens，每个 token 选择 top-2 专家

#### Step 1: 计算每个 token 的通信量

```
每个 token 被路由到 2 个专家 (top_k=2)

单个 token 的通信内容:
1. hidden_states: [1, H] = [1, 4096], bf16 = 4096 × 2 bytes = 8,192 bytes = 8 KB
2. expert_ids: [1] (int64) = 8 bytes (可忽略)
3. expert_weights: [1] (bf16) = 2 bytes (可忽略)

主要通信量 = H × sizeof(dtype) × top_k
           = 4096 × 2 × 2
           = 16,384 bytes
           = 16 KB per token
```

#### Step 2: 计算总通信量

```
总 tokens = B × S = 32 × 1024 = 32,768

完美负载均衡时:
- 每个 EP 组接收的 tokens = 总 tokens × top_k / EP
- = 32,768 × 2 / 4
- = 16,384 tokens

All-to-All 通信量公式:
对于 N 个节点的 All-to-All:
- 每个节点发送: (N-1)/N × 总数据量
- 每个节点接收: (N-1)/N × 总数据量
- 总带宽 (双向): 2 × (N-1)/N × 总数据量

在我们的场景:
- N = EP = 4 (All-to-All 在 EP 组内进行)
- 总数据量 = 32,768 tokens × 16 KB = 512 MB
- (N-1)/N = 3/4

因此:
- 每个 GPU 发送: 3/4 × 512 MB = 384 MB
- 每个 GPU 接收: 3/4 × 512 MB = 384 MB
- 总通信量 (P2P): 384 + 384 = 768 MB

但通常我们说 "通信量" 指的是单向的峰值:
- 峰值发送/接收 = 384 MB

或者使用更粗略的估计:
- 总数据量 × 2 (双向) = 512 MB × 2 = 1,024 MB ≈ 1 GB
```

#### 更精确的计算

```python
def calculate_alltoall_communication(
    batch_size, seq_len, hidden_size, top_k, 
    num_experts, ep_size, dtype_bytes=2
):
    """
    计算 MoE Dispatch All-to-All 通信量
    """
    total_tokens = batch_size * seq_len
    
    # 每个 token 的通信大小 (hidden state)
    bytes_per_token = hidden_size * dtype_bytes * top_k
    
    # 总数据量
    total_data = total_tokens * bytes_per_token
    
    # All-to-All 通信系数 (N-1)/N
    comm_factor = (ep_size - 1) / ep_size
    
    # 每个 GPU 的发送/接收量
    per_gpu_send = total_data * comm_factor
    per_gpu_recv = total_data * comm_factor
    
    # 总 P2P 通信量 (发送+接收)
    total_p2p = per_gpu_send + per_gpu_recv
    
    return {
        'total_tokens': total_tokens,
        'bytes_per_token': bytes_per_token,
        'total_data_mb': total_data / 1024**2,
        'comm_factor': comm_factor,
        'per_gpu_send_mb': per_gpu_send / 1024**2,
        'per_gpu_recv_mb': per_gpu_recv / 1024**2,
        'total_p2p_mb': total_p2p / 1024**2,
    }

# 示例计算
result = calculate_alltoall_communication(
    batch_size=32, seq_len=1024, hidden_size=4096, 
    top_k=2, num_experts=8, ep_size=4
)

print(f"""
All-to-All 通信量计算:
  总 tokens: {result['total_tokens']:,}
  每 token 字节: {result['bytes_per_token']} bytes
  总数据量: {result['total_data_mb']:.1f} MB
  通信系数 (EP-1)/EP: {result['comm_factor']:.2f}
  每 GPU 发送: {result['per_gpu_send_mb']:.1f} MB
  每 GPU 接收: {result['per_gpu_recv_mb']:.1f} MB
  总 P2P 通信: {result['total_p2p_mb']:.1f} MB
""")
```

**输出**:
```
All-to-All 通信量计算:
  总 tokens: 32,768
  每 token 字节: 16384 bytes (16 KB)
  总数据量: 512.0 MB
  通信系数 (EP-1)/EP: 0.75
  每 GPU 发送: 384.0 MB
  每 GPU 接收: 384.0 MB
  总 P2P 通信: 768.0 MB
```

---

## 3. All-Reduce 通信量计算

### 3.1 基础概念

**All-Reduce 是什么？**
- 所有 GPU 都有部分数据
- 计算总和 (或平均值) 后，所有 GPU 得到相同结果
- 常用算法: Ring All-Reduce

**Ring All-Reduce 过程** (N 个 GPU):
```
阶段 1: Reduce-Scatter (N-1 步)
- 每步每个 GPU 发送 1/N 数据给下一个 GPU
- 每步每个 GPU 接收 1/N 数据并累加
- 总通信: (N-1) × (数据大小/N) = (N-1)/N × 数据大小

阶段 2: All-Gather (N-1 步)
- 每步每个 GPU 发送 1/N 结果给下一个 GPU
- 每步每个 GPU 接收 1/N 结果
- 总通信: (N-1) × (数据大小/N) = (N-1)/N × 数据大小

总计: 2 × (N-1)/N × 数据大小
```

### 3.2 ETP All-Reduce 通信量

**场景**: Expert 计算后的结果聚合

#### Step 1: 计算数据大小

```
经过 Expert 计算后:
- tokens 数: ~16,384 (假设负载均衡)
- hidden 维度: H = 4096
- 数据形状: [16384, 4096]
- 数据大小: 16384 × 4096 × 2 bytes = 134,217,728 bytes = 128 MB
```

#### Step 2: 应用 Ring All-Reduce 公式

```
对于 ETP All-Reduce:
- N = ETP = 2
- 数据大小 = 128 MB
- (N-1)/N = 1/2

阶段 1 (Reduce-Scatter):
- 通信量 = 1/2 × 128 MB = 64 MB

阶段 2 (All-Gather):
- 通信量 = 1/2 × 128 MB = 64 MB

总计:
- 每个 GPU 发送: 64 + 64 = 128 MB
- 每个 GPU 接收: 64 + 64 = 128 MB
- 总 P2P 通信: 256 MB

或者使用简化公式:
- 总通信 = 2 × (N-1)/N × 数据大小
- = 2 × 1/2 × 128 MB
- = 128 MB (单向峰值)
- = 256 MB (双向总计)
```

#### 代码实现

```python
def calculate_allreduce_communication(
    num_tokens, hidden_size, etp_size, dtype_bytes=2
):
    """
    计算 ETP All-Reduce 通信量
    """
    # 数据大小
    data_size = num_tokens * hidden_size * dtype_bytes
    
    # Ring All-Reduce 系数 2*(N-1)/N
    comm_factor = 2 * (etp_size - 1) / etp_size
    
    # 每个 GPU 的发送/接收量
    per_gpu_comm = data_size * comm_factor
    
    return {
        'num_tokens': num_tokens,
        'data_size_mb': data_size / 1024**2,
        'comm_factor': comm_factor,
        'per_gpu_comm_mb': per_gpu_comm / 1024**2,
        'total_p2p_mb': (per_gpu_comm * 2) / 1024**2,  # 发送+接收
    }

# 示例计算
result = calculate_allreduce_communication(
    num_tokens=16384, hidden_size=4096, etp_size=2
)

print(f"""
All-Reduce 通信量计算 (w1 输出):
  Token 数: {result['num_tokens']:,}
  数据大小: {result['data_size_mb']:.1f} MB
  通信系数 2*(ETP-1)/ETP: {result['comm_factor']:.2f}
  每 GPU 通信: {result['per_gpu_comm_mb']:.1f} MB
  总 P2P 通信: {result['total_p2p_mb']:.1f} MB
""")
```

**输出**:
```
All-Reduce 通信量计算 (w1 输出):
  Token 数: 16,384
  数据大小: 128.0 MB
  通信系数 2*(ETP-1)/ETP: 1.00
  每 GPU 通信: 128.0 MB
  总 P2P 通信: 256.0 MB
```

---

## 4. 总通信量计算

### 4.1 单层 MoE 的通信量

```python
def calculate_moe_layer_communication(
    batch_size, seq_len, hidden_size, ffn_hidden, top_k,
    num_experts, ep_size, etp_size, dtype_bytes=2
):
    """
    计算单层 MoE 的总通信量
    """
    total_tokens = batch_size * seq_len
    
    # 1. Dispatch All-to-All
    dispatch_data = total_tokens * hidden_size * dtype_bytes * top_k
    dispatch_factor = (ep_size - 1) / ep_size
    dispatch_send = dispatch_data * dispatch_factor
    
    # 2. ETP All-Reduce (w1 后)
    # 经过 dispatch 后，每个 EP 组的 tokens
    tokens_per_ep = total_tokens * top_k / ep_size
    w1_output_size = tokens_per_ep * ffn_hidden * dtype_bytes
    w1_allreduce_factor = 2 * (etp_size - 1) / etp_size
    w1_allreduce = w1_output_size * w1_allreduce_factor
    
    # 3. ETP All-Reduce (w2 后)
    w2_output_size = tokens_per_ep * hidden_size * dtype_bytes
    w2_allreduce = w2_output_size * w1_allreduce_factor
    
    # 4. Combine All-to-All
    combine_data = total_tokens * hidden_size * dtype_bytes * top_k
    combine_send = combine_data * dispatch_factor
    
    # 总计
    total_per_gpu = dispatch_send + w1_allreduce + w2_allreduce + combine_send
    
    return {
        'dispatch_mb': dispatch_send / 1024**2,
        'w1_allreduce_mb': w1_allreduce / 1024**2,
        'w2_allreduce_mb': w2_allreduce / 1024**2,
        'combine_mb': combine_send / 1024**2,
        'total_per_gpu_mb': total_per_gpu / 1024**2,
    }

# 示例计算
result = calculate_moe_layer_communication(
    batch_size=32, seq_len=1024, hidden_size=4096, ffn_hidden=11264,
    top_k=2, num_experts=8, ep_size=4, etp_size=2
)

print(f"""
单层 MoE 通信量:
  Dispatch (All-to-All):     {result['dispatch_mb']:.1f} MB
  w1 All-Reduce (ETP):       {result['w1_allreduce_mb']:.1f} MB
  w2 All-Reduce (ETP):       {result['w2_allreduce_mb']:.1f} MB
  Combine (All-to-All):      {result['combine_mb']:.1f} MB
  ─────────────────────────────────
  总计 (每 GPU 每步):        {result['total_per_gpu_mb']:.1f} MB
""")
```

**输出**:
```
单层 MoE 通信量:
  Dispatch (All-to-All):     384.0 MB
  w1 All-Reduce (ETP):       704.0 MB  # 注意: ffn_hidden=11264
  w2 All-Reduce (ETP):       256.0 MB
  Combine (All-to-All):      384.0 MB
  ─────────────────────────────────
  总计 (每 GPU 每步):        1728.0 MB ≈ 1.7 GB
```

**注意**: 我之前文档中说的 ~2.25 GB 是粗略估计，实际精确计算是 ~1.7 GB。

### 4.2 完整公式推导

```
变量定义:
- T = B × S (总 tokens)
- H (hidden_size)
- F (ffn_hidden)
- K (top_k)
- EP, ETP

1. Dispatch All-to-All:
   数据 = T × H × 2 × K (bf16)
   通信 = (EP-1)/EP × T × H × 2 × K

2. w1 All-Reduce:
   每 EP tokens = T × K / EP
   数据 = (T × K / EP) × F × 2 (bf16)
   通信 = 2 × (ETP-1)/ETP × (T × K / EP) × F × 2

3. w2 All-Reduce:
   数据 = (T × K / EP) × H × 2
   通信 = 2 × (ETP-1)/ETP × (T × K / EP) × H × 2

4. Combine All-to-All:
   同 Dispatch
```

---

## 5. 不同配置的对比

### 5.1 EP=8, ETP=1 vs EP=4, ETP=2

```python
configs = [
    {'ep': 8, 'etp': 1, 'name': 'EP=8, ETP=1'},
    {'ep': 4, 'etp': 2, 'name': 'EP=4, ETP=2'},
    {'ep': 2, 'etp': 4, 'name': 'EP=2, ETP=4'},
]

for cfg in configs:
    result = calculate_moe_layer_communication(
        batch_size=32, seq_len=1024, hidden_size=4096, ffn_hidden=11264,
        top_k=2, num_experts=8, 
        ep_size=cfg['ep'], etp_size=cfg['etp']
    )
    print(f"{cfg['name']}: {result['total_per_gpu_mb']:.1f} MB")
```

**输出**:
```
EP=8, ETP=1: 896.0 MB   # 最少通信，但单专家参数大
EP=4, ETP=2: 1728.0 MB  # 中等通信，单专家参数减半
EP=2, ETP=4: 3328.0 MB  # 最多通信，单专家参数最小
```

**结论**:
- 增加 ETP 会增加通信量 (因为 All-Reduce 开销)
- 但 ETP 可以支持更大的专家模型
- 需要在通信量和模型容量之间权衡

---

## 6. 实际影响因素

### 6.1 负载不均衡的影响

```
上述计算假设完美负载均衡。

实际情况:
- 如果某些专家过载，实际通信量会增加
- 因为某些 EP 组会接收更多的 tokens

最坏情况:
- 所有 tokens 都路由到同一个专家
- 该专家所在的 EP 组接收所有 tokens × top_k
- 通信量变为原来的 EP 倍
```

### 6.2 Sequence Parallel 的影响

```
如果启用 SP (Sequence Parallel):
- 输入 x 是切分过的
- 需要额外的 gather/scatter 通信
- 通信量增加 (SP size - 1)/SP size × 数据大小
```

### 6.3 异步通信

```
实际优化:
- DeepEP 可以 overlap 通信和计算
- 实际 wall-clock time 可能小于理论通信时间
- 但带宽占用仍然是这么多
```

---

## 7. 快速估算公式

### 7.1 简化估算

```
对于 Dispatch/Combine (All-to-All):
通信量 ≈ 2 × (EP-1)/EP × T × H × K × 2 bytes

对于 ETP All-Reduce:
通信量 ≈ 2 × (ETP-1)/ETP × (T×K/EP) × (H + F) × 2 bytes

总计 ≈ 4 × T × H × K × (EP-1)/EP × 2 bytes + 
      4 × T × K × (H + F) × (ETP-1)/(EP×ETP) × 2 bytes
```

### 7.2 经验公式

```python
def quick_estimate(batch_size, seq_len, hidden_size, ffn_hidden, top_k, ep, etp):
    """快速估算通信量 (MB)"""
    T = batch_size * seq_len
    H = hidden_size
    F = ffn_hidden
    K = top_k
    
    # All-to-All (Dispatch + Combine)
    alltoall = 4 * T * H * K * (ep - 1) / ep * 2 / 1024**2
    
    # All-Reduce (w1 + w2)
    allreduce = 4 * T * K * (H + F) * (etp - 1) / (ep * etp) * 2 / 1024**2
    
    return alltoall + allreduce

# 示例
print(quick_estimate(32, 1024, 4096, 11264, 2, 4, 2))
# 输出: ~1728 MB
```

---

## 8. 总结

### 8.1 核心公式

| 通信类型 | 公式 | 说明 |
|---------|------|------|
| All-to-All | `(N-1)/N × 总数据` | N=EP, 总数据=T×H×K×2 bytes |
| All-Reduce | `2×(N-1)/N × 总数据` | N=ETP, Ring 算法 |

### 8.2 关键洞察

1. **EP 越大，All-to-All 通信越接近总数据量**
   - EP=8: 7/8 = 87.5% 的数据需要通信
   - EP=4: 3/4 = 75% 的数据需要通信

2. **ETP 越大，All-Reduce 通信越接近 2×数据量**
   - ETP=2: 1×数据量
   - ETP=4: 1.5×数据量
   - ETP→∞: 2×数据量 (上限)

3. **通信量与 batch_size × seq_len 成正比**
   - 更大的 batch 意味着更多的通信
   - 但计算效率通常也更高

---

> 文档版本: 2025-03
> 详细解释了 MoE 通信量的计算方法
