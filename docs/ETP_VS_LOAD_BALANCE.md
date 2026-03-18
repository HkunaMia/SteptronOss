# ETP 与负载均衡的关系

> 本文澄清 ETP (Expert Tensor Parallel) 与 MoE 负载均衡的关系，解释为什么 ETP 不能缓解负载不均。

---

## 1. 核心结论

### 1.1 ETP 与负载均衡无关

| 维度 | 作用 | 是否影响负载均衡 |
|------|------|-----------------|
| **EP** (Expert Parallel) | 将不同专家分布到不同 GPU | ✅ 直接影响负载均衡 |
| **ETP** (Expert Tensor Parallel) | 将单个专家切分到多个 GPU | ❌ 不影响负载均衡 |
| **Router** | 决定 token 分配给哪些专家 | ✅ 直接影响负载均衡 |
| **Load Balance Strategy** | Aux-Loss-Free / Capacity | ✅ 直接影响负载均衡 |

### 1.2 通俗解释

```
类比: 医院分诊系统

EP (Expert Parallel):
- 相当于将不同科室分配到不同大楼
- 内科在 A 楼，外科在 B 楼，儿科在 C 楼...
- 如果所有人都去内科，A 楼就会爆满（负载不均）

ETP (Expert Tensor Parallel):
- 相当于在一个科室内增加医生数量
- 内科有 3 个医生（ETP=3），可以并行处理病人
- 但如果病人都挂内科号，这 3 个医生都会很忙
- ETP 只是让处理更快，但不改变"大家都去内科"的事实

Router (分诊台):
- 决定病人去哪个科室
- 好的分诊策略可以平衡各科室的负载
```

---

## 2. 技术细节分析

### 2.1 ETP 的工作方式

```python
# ETP 的核心代码 (moe_block.py:158)
def forward(self, x, token_expert_ids, token_weights):
    # 1. 所有 ETP GPU 接收相同的输入
    # GPU 0 (ETP=0) 和 GPU 1 (ETP=1) 的 x 完全相同
    
    # 2. 各自计算部分结果
    x = routed_grouped_ffn(self.w1, self.w2, ...)  # 局部计算
    
    # 3. All-Reduce 聚合结果
    x = reduce_from_tensor_model_parallel_region(x, group="ETP")
    return x
```

**关键观察**:
- ETP 组内的所有 GPU 接收 **完全相同的 token 集合**
- 只是各自拥有专家参数的 **不同切片**
- 通过 All-Reduce 聚合得到完整结果

### 2.2 负载均衡体现在哪里

```
场景: 专家 E0 过载，专家 E2 空闲

EP=4, ETP=2 配置:
- EP=0 (GPUs 0,1): 负责 E0, E1
- EP=1 (GPUs 2,3): 负责 E2, E3
- EP=2 (GPUs 4,5): 负责 E4, E5
- EP=3 (GPUs 6,7): 负责 E6, E7

负载情况:
┌──────────────────────────────────────────────────────────┐
│                    Token 分布                             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  专家 E0 (EP=0)        专家 E2 (EP=1)                     │
│  ┌─────────────┐      ┌─────────────┐                    │
│  │ GPU 0 (ETP=0)│      │ GPU 2 (ETP=0)│                    │
│  │ ▓▓▓▓▓▓▓▓▓▓ │      │ ░░░░░░░░░░ │  <- 空闲            │
│  │ 1000 tokens │      │ 100 tokens  │                    │
│  ├─────────────┤      ├─────────────┤                    │
│  │ GPU 1 (ETP=1)│      │ GPU 3 (ETP=1)│                    │
│  │ ▓▓▓▓▓▓▓▓▓▓ │      │ ░░░░░░░░░░ │  <- 空闲            │
│  │ 1000 tokens │      │ 100 tokens  │                    │
│  └─────────────┘      └─────────────┘                    │
│                                                           │
│  注意: GPU 0 和 GPU 1 处理相同的 1000 个 token！          │
│       只是各自计算部分参数 (w1[:,:half] vs w1[:,half:])  │
│                                                           │
└──────────────────────────────────────────────────────────┘

问题: 
- E0 有 1000 个 token，E2 只有 100 个 token
- EP=0 组 (GPUs 0,1) 很忙
- EP=1 组 (GPUs 2,3) 很闲
- 这就是负载不均！

ETP 的作用:
- 让 GPU 0 和 GPU 1 并行处理这 1000 个 token
- 但无法减少这 1000 个 token 的数量
- 也无法增加 E2 的 100 个 token 数量
```

---

## 3. 真正的负载均衡方案

### 3.1 负载均衡发生在哪个层次

```
┌──────────────────────────────────────────────────────────┐
│                   负载均衡的层次                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. Router 层 (最关键)                                    │
│     ├─ Aux-Loss-Free Load Balancing                      │
│     ├─ Sigmoid Router                                    │
│     └─ Router Bias Update                                │
│     作用: 让 token 更均匀地分配给不同专家                  │
│                                                           │
│  2. Capacity Factor 层                                    │
│     ├─ 限制每个专家的最大 token 数                        │
│     ├─ 超出 capacity 的 token 被丢弃或重路由              │
│     └─ 缓解极端过载情况                                   │
│                                                           │
│  3. EP 层                                                 │
│     ├─ 通过调整 EP 大小改变专家分布粒度                   │
│     ├─ EP 越大，每 GPU 负责的专家越少                     │
│     └─ 但无法解决单个专家过载问题                         │
│                                                           │
│  ❌ ETP 层                                                │
│     ├─ 只影响计算并行度                                   │
│     └─ 不改变 token 分配                                  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Aux-Loss-Free Load Balancing

```python
# 这才是真正缓解负载不均的机制 (moe_block.py:235)

def forward_router(self, logits):
    gate_prob = F.softmax(logits, dim=1)
    
    if self.cfg.enable_auxiliary_loss_free_load_balance:
        # 使用 bias 调整路由选择
        # bias 会根据历史负载动态调整
        biased_prob = gate_prob + self.router_balance_bias.unsqueeze(0)
        topk_expert_ids = biased_prob.topk(self.moe_top_k).indices
        
        # 但权重仍使用原始 prob (保证模型质量)
        token_weights = gate_prob.gather(1, topk_expert_ids)
```

**工作原理**:
- 如果专家 E0 过载，`router_balance_bias[0]` 会降低
- 这样 token 就更不容易选择 E0
- 实现负载的动态平衡

### 3.3 Capacity Factor

```python
# 虽然代码中没有直接显示，但通常 MoE 会实现 capacity 限制

# 概念:
max_tokens_per_expert = capacity_factor * (total_tokens / num_experts)

# 如果某专家接收到的 token 超过 capacity:
# - 方案 1: 丢弃多余 token (Drop)
# - 方案 2: 重路由到次要专家 (Reroute)
# - 方案 3: 阻塞等待 (Block)
```

---

## 4. ETP 的真正作用

### 4.1 ETP 解决什么问题

```
ETP 解决的是 "单专家太大" 的问题，而不是 "负载不均" 的问题。

场景 1: 单专家参数太大
- 专家 hidden_size = 16384
- 专家参数量 = hidden_size × ffn_hidden × 2 = 数 GB
- 单卡放不下 → 使用 ETP 切分

场景 2: 计算并行度
- ETP 组内的 GPU 可以并行计算同一个专家
- 减少单个专家的延迟
- 但通信开销 (All-Reduce) 会增加
```

### 4.2 ETP 与 EP 的配合

```
┌──────────────────────────────────────────────────────────┐
│              EP 与 ETP 的分工                             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  EP 负责:                                                 │
│  ├─ 将不同专家分布到不同 GPU                             │
│  ├─ 通过增加 EP 减少每 GPU 的专家数                       │
│  ├─ 通过 Dispatch/Combine 实现 token 路由                 │
│  └─ 间接影响负载均衡 (更多 EP = 更细粒度的专家分布)       │
│                                                           │
│  ETP 负责:                                                │
│  ├─ 将单个专家切分到多个 GPU                             │
│  ├─ 支持超大专家模型                                     │
│  ├─ 提供计算并行度                                       │
│  └─ ❌ 不参与负载均衡                                    │
│                                                           │
│  关系:                                                   │
│  ├─ EP 解决 "专家间" 的分布问题                          │
│  ├─ ETP 解决 "专家内" 的切分问题                         │
│  └─ 两者正交，可以独立配置                               │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 5. 负载均衡的评估指标

### 5.1 关键指标

```python
# steptronoss/model/common/moe_block.py:365

GlobalMetrics.peak_to_avg_ratio.add(
    (token_expert_ids != -1).sum() / raw_token_count / self.cfg.moe_top_k,
    subname=f"layer{self.moe_layer_id}"
)

# 这个指标计算:
# - 实际处理的 token 数 / 期望的 token 数
# - 期望 = total_tokens × top_k / num_experts (完美均衡时)
# - 越接近 1，说明负载越均衡

# 其他指标 (moe_block.py:294-336)
GlobalMetrics.moe_minp      # 最小路由概率
GlobalMetrics.moe_sumk      # top-k 概率和
GlobalMetrics.moe_std       # 路由分布标准差
GlobalMetrics.expert_coef   # 专家负载系数 (max/sum)
```

### 5.2 指标解读

```
完美均衡情况:
- peak_to_avg_ratio = 1.0
- expert_coef = 1.0 / num_experts (所有专家负载相同)
- moe_std = 0 (所有专家接收相同数量的 token)

实际情况下:
- peak_to_avg_ratio = 1.2 ~ 2.0 (20%~100% 的不均衡)
- 如果 > 2.0，说明负载严重不均，需要优化 router

ETP 对这些指标的影响:
- 无影响
- 因为 ETP 不改变 token 的分配
```

---

## 6. 最佳实践

### 6.1 如何选择 EP 和 ETP

```python
# 决策树:

if 单专家可以放入单卡显存:
    # 不需要 ETP
    ETP = 1
    EP = 根据专家总数和 GPU 总数决定
else:
    # 需要 ETP 切分专家
    ETP = ceil(单专家参数量 / 单卡显存容量)
    EP = (总 GPU 数) / ETP

# 负载均衡优化 (与 ETP 无关):
if 负载不均严重:
    启用 enable_auxiliary_loss_free_load_balance = True
    调整 router_bias_update_rate
    考虑使用 capacity_factor 限制
```

### 6.2 配置示例

```python
# 场景: 256 专家，每个专家 2GB，总 GPU 64

# 方案 1: 只用 EP (如果单专家能放下)
EP = 64, ETP = 1
- 每 GPU 负责 256/64 = 4 个专家
- 专家不切分
- 负载均衡靠 Router

# 方案 2: EP + ETP (如果单专家太大)
EP = 32, ETP = 2
- 每 GPU 负责 256/32 = 8 个专家的一半参数
- 专家切分为 2 份
- 负载均衡仍然靠 Router，ETP 只解决显存问题

# 方案 3: 更多 ETP (如果单专家非常大)
EP = 16, ETP = 4
- 每 GPU 负责 256/16 = 16 个专家的四分之一参数
- 专家切分为 4 份
- 负载均衡仍然靠 Router
```

---

## 7. 常见误区

### 误区 1: "增加 ETP 可以缓解负载不均"

❌ 错误理解:
- "ETP=2 时，两个 GPU 一起处理专家，所以不会过载"

✅ 正确理解:
- ETP 组内的 GPU 处理 **相同的 token**
- 如果某个专家过载，整个 ETP 组都会过载
- ETP 只是加速计算，不减少工作量

### 误区 2: "EP 越大负载越均衡"

⚠️ 部分正确:
- EP 越大，每 GPU 负责的专家越少
- 但单个专家仍然可能过载
- 真正的均衡需要 Router 策略的配合

### 误区 3: "负载不均只影响性能，不影响模型质量"

❌ 错误:
- 负载不均会导致:
  1. 某些专家训练不足 (接收 token 太少)
  2. 某些专家过拟合 (接收 token 太多)
  3. GPU 利用率不均衡 (部分 GPU 空闲)
- 需要使用 Aux-Loss-Free 等策略保证均衡

---

## 8. 总结

### 8.1 核心观点

1. **ETP 与负载均衡无关**
   - ETP 只解决 "专家太大放不下" 的问题
   - 不改变 token 的分配策略

2. **负载均衡靠 Router 和 EP**
   - Router: Aux-Loss-Free, Sigmoid, Bias Update
   - EP: 细粒度的专家分布
   - Capacity Factor: 硬性限制

3. **ETP 和 EP 是正交的**
   - 可以独立配置
   - 解决不同的问题

### 8.2 快速判断

```
问题: "我的 MoE 模型负载不均，应该增加 ETP 吗？"

回答:
- 如果问题是 "某些 GPU 太忙，某些 GPU 太闲" → 调整 Router 策略，不是 ETP
- 如果问题是 "单专家太大，单卡放不下" → 增加 ETP
- 如果问题是 "专家计算太慢" → 增加 ETP 或使用 Grouped GEMM
```

---

> 文档版本: 2025-03
> 澄清了 ETP 与负载均衡的关系
