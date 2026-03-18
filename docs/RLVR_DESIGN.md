# SteptronOss RLVR 训练流程设计与优化详解

本文档系统梳理 SteptronOss 框架在 RLVR（Reinforcement Learning with Verifiable Rewards / PPO）训练流程中的架构设计和优化手段。

---

## 目录

- [SteptronOss RLVR 训练流程设计与优化详解](#steptronoss-rlvr-训练流程设计与优化详解)
  - [目录](#目录)
  - [一、整体架构概览](#一整体架构概览)
    - [1.1 三模型架构](#11-三模型架构)
    - [1.2 训练-推理分离的混合引擎](#12-训练-推理分离的混合引擎)
    - [1.3 数据流总览](#13-数据流总览)
  - [二、核心数据结构](#二核心数据结构)
    - [2.1 EnvTrajectory：环境交互轨迹](#21-envtrajectory环境交互轨迹)
    - [2.2 PPOSample：PPO 训练样本](#22-pposampleppo-训练样本)
    - [2.3 PackedPPOSamples：打包批处理](#23-packedpposamples打包批处理)
  - [三、训练流程详解](#三训练流程详解)
    - [3.1 PPOTrainer.train\_step() 全流程](#31-ppotrainertrain_step-全流程)
    - [3.2 Actor 损失函数（PPO 目标）](#32-actor-损失函数ppo-目标)
    - [3.3 Critic 损失函数](#33-critic-损失函数)
    - [3.4 GAE 计算](#34-gae-计算)
    - [3.5 Critic Warmup 机制](#35-critic-warmup-机制)
  - [四、Flow Control 异步流控](#四flow-control-异步流控)
    - [4.1 三层队列架构](#41-三层队列架构)
    - [4.2 三种调度策略](#42-三种调度策略)
    - [4.3 权重同步机制](#43-权重同步机制)
    - [4.4 持久化队列与检查点恢复](#44-持久化队列与检查点恢复)
  - [五、异步生成控制器](#五异步生成控制器)
    - [5.1 多进程生成架构](#51-多进程生成架构)
    - [5.2 Trainable 抽象与奖励计算](#52-trainable-抽象与奖励计算)
  - [六、vLLM 推理服务架构](#六vllm-推理服务架构)
    - [6.1 Router 负载均衡](#61-router-负载均衡)
    - [6.2 权重热部署](#62-权重热部署)
  - [七、内存优化：模型卸载与重载](#七内存优化模型卸载与重载)
    - [7.1 PackedModel 三级卸载](#71-packedmodel-三级卸载)
    - [7.2 训练流程中的卸载编排](#72-训练流程中的卸载编排)
    - [7.3 显存预算估算](#73-显存预算估算)
  - [八、变长序列打包优化](#八变长序列打包优化)
    - [8.1 打包策略](#81-打包策略)
    - [8.2 数据均衡分配](#82-数据均衡分配)
    - [8.3 TP × CP × 2 对齐约束](#83-tp--cp--2-对齐约束)
  - [九、并行策略在 RLVR 中的适配](#九并行策略在-rlvr-中的适配)
    - [9.1 Context Parallel 与 GAE 的兼容](#91-context-parallel-与-gae-的兼容)
    - [9.2 分布式 Loss 计算](#92-分布式-loss-计算)
    - [9.3 数据源 Rank 选择](#93-数据源-rank-选择)
  - [十、指标与监控系统](#十指标与监控系统)
    - [10.1 指标分类](#101-指标分类)
    - [10.2 离策略诊断](#102-离策略诊断)
  - [十一、检查点与容错](#十一检查点与容错)
    - [11.1 多角色检查点](#111-多角色检查点)
    - [11.2 自动恢复](#112-自动恢复)
    - [11.3 样本保存与重放](#113-样本保存与重放)
  - [十二、总结对照表](#十二总结对照表)

---

## 一、整体架构概览

### 1.1 三模型架构

RLVR 训练涉及三个模型角色，各有不同的生命周期和更新策略：

```
┌─────────────────────────────────────────────────────┐
│                   RLVR 训练系统                       │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Actor   │  │  Reference   │  │    Critic     │  │
│  │ (策略网络) │  │  (参考模型)   │  │  (价值网络)    │  │
│  │          │  │              │  │               │  │
│  │ 可训练    │  │  权重冻结     │  │  可训练       │  │
│  │ 生成动作  │  │  KL 正则化   │  │  估计价值     │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
│       ↑              ↑                  ↑           │
│       │         从 Actor 初始化      从 Actor 初始化  │
└─────────────────────────────────────────────────────┘
```

**代码位置**：`steptronoss/exp/rl.py:196-202`（PPOCheckpointCfg）

```python
class PPOCheckpointCfg(Config):
    actor: RoleCheckpointConfig    # Actor 检查点配置
    critic: RoleCheckpointConfig   # Critic 检查点配置
    reference: RoleCheckpointConfig  # Reference 检查点配置
```

### 1.2 训练-推理分离的混合引擎

```
训练集群                                  推理集群
┌───────────────────────┐              ┌───────────────────────┐
│  PyTorch + Megatron    │   权重同步    │  vLLM Server(s)       │
│                        │ ──────────→ │  TP 并行推理           │
│  Actor / Critic /      │  safetensors │                       │
│  Reference 前向反向    │   热部署     │  高吞吐 decode         │
└───────────────────────┘              └───────────────────────┘
                                              ↑
                                       ┌──────┴──────┐
                                       │ VLLMRouter   │
                                       │ 负载均衡      │
                                       │ FastAPI      │
                                       └─────────────┘
```

训练引擎使用 PyTorch + Megatron 并行框架进行梯度更新，推理引擎使用 vLLM 进行高吞吐的 token 生成。两者通过权重文件热部署机制同步。

### 1.3 数据流总览

```
数据源 (Dataloader)
  ↓
GenableItem / TrainableItem
  ↓  ← FlowController 调度
异步生成 (GenerationController + vLLM)
  ↓
EnvTrajectory [trajectory, logprobs, is_gen_mask, raw_reward]
  ↓  ← adapt_trajs_to_samples()
PPOSample [+ ref_logprobs, actor_logprobs, advantages, returns, values]
  ↓  ← data_balance_and_pack()
PackedPPOSamples [打包后的高效批处理]
  ↓
Reference 前向 → ref_logprobs
Actor 前向 → actor_logprobs
Critic 前向 → values → GAE → advantages, returns
  ↓
Critic 训练 (MSE loss on values vs returns)
Actor 训练 (PPO clipped surrogate loss)
```

---

## 二、核心数据结构

### 2.1 EnvTrajectory：环境交互轨迹

**代码位置**：`steptronoss/exp/rl.py:44-74`

```python
class EnvTrajectory(DataClass):
    trajectory: torch.LongTensor = None   # prompt + 生成的 token ids
    logprobs: torch.Tensor = None         # 生成时的 log-probabilities
    is_gen_mask: torch.BoolTensor = None  # True 标记生成部分（vs prompt 部分）
    raw_reward: float = None              # 环境返回的奖励信号
    meta: dict = None                     # 元信息（如 prompt_text, gt 等）
    stop_type: str = None                 # 停止原因：length / eos
```

`can_be_trained` 属性验证轨迹完整性（trajectory 和 logprobs 均不为 None 且长度一致）。

### 2.2 PPOSample：PPO 训练样本

**代码位置**：`steptronoss/exp/rl.py:77-92`

```python
class PPOSample(EnvTrajectory):
    prompt_id: int = None                 # 来自哪个 prompt
    ref_logprobs: torch.Tensor = None     # Reference 模型的 logprobs
    actor_logprobs: torch.Tensor = None   # Actor 模型的 logprobs
    advantages: torch.Tensor = None       # GAE 计算的 advantages
    returns: torch.Tensor = None          # GAE 计算的 returns
    values: torch.Tensor = None           # Critic 估计的 values
    input_ids: torch.LongTensor = None
    labels: torch.LongTensor = None
```

### 2.3 PackedPPOSamples：打包批处理

**代码位置**：`steptronoss/exp/rl.py:94-182`

```python
class PackedPPOSamples(DataClass):
    input_ids: torch.Tensor = None        # 连接后的所有 token ids: [1, Total_S]
    labels: torch.Tensor = None           # shift-left 后的 labels: [1, Total_S]
    cu_seqlens: torch.Tensor = None       # 累积长度（含 padding）: [N+1]
    cu_valid_sizes: torch.Tensor = None   # 累积长度（不含 padding）: [N+1]
    max_seq_len: torch.Tensor = None      # 最大单条序列长度

    logprobs: torch.Tensor = None         # 生成时的 logprobs（仅 response 部分）
    ref_logprobs: torch.Tensor = None     # Reference logprobs
    actor_logprobs: torch.Tensor = None   # Actor logprobs
    advantages: torch.Tensor = None       # Advantages
    returns: torch.Tensor = None          # Returns
    values: torch.Tensor = None           # Values

    is_gen_mask: torch.Tensor = None      # 生成部分掩码
    samples: list[PPOSample] = None       # 原始样本引用（用于 unpack）
```

`from_samples()` 类方法负责打包逻辑，详见[第八节](#八变长序列打包优化)。

---

## 三、训练流程详解

### 3.1 PPOTrainer.train_step() 全流程

**代码位置**：`steptronoss/core/trainers/ppo_trainer.py:439-581`

```python
def train_step(self):
    # ═══ 阶段 1：生成轨迹 ═══
    all_trajs = self.generate_trajectory()           # FlowController 调度生成
    all_samples = self.adapt_trajs_to_samples(trajs)  # 转为 PPOSample，张量移至 CUDA
    all_samples = all_gather_object(all_samples)      # 广播到所有 DP rank（保证一致）

    # ═══ 阶段 2：数据处理与打包 ═══
    all_samples = self.filter_samples(all_samples)    # 过滤无效样本
    my_samples = self.data_balance_and_pack(all_samples)  # 分组 → 均衡分配 → 打包

    # ═══ 阶段 3：前向推理（不更新参数） ═══
    self.get_reference(my_samples)       # Reference 前向 → ref_logprobs
    self.get_actor_logprob(my_samples)   # Actor 前向 → actor_logprobs
    self.get_values_advantages(my_samples)  # Critic 前向 → values → GAE

    # ═══ 阶段 4：Critic 训练 ═══
    for chunk in chunk_my_samples(my_samples, fix_iters_critic):
        self.critic.forward_backward(chunk, loss_fn=critic_loss_func)
        self.critic.optimizer_step()

    # ═══ 阶段 5：Actor 训练（跳过 warmup 期） ═══
    if iteration >= critic_warmup_iters:
        for chunk in chunk_my_samples(my_samples, fix_iters):
            self.actor.forward_backward(chunk, loss_fn=actor_loss_func)
            self.actor.optimizer_step()
```

### 3.2 Actor 损失函数（PPO 目标）

**代码位置**：`playground/rlvr/qwen3_1p5b_rlvr_math.py:255-291`

```python
def actor_loss_func(data: PackedPPOSamples, logits):
    # 1. 从模型 logits 计算新的 logprobs
    logits = scatter_to_balanced_cp_region(logits)   # CP 切分
    new_logprobs = -vocab_parallel_cross_entropy(     # TP 分布式交叉熵
        logits / temperature, labels
    )[0]
    new_logprobs = gather_from_balanced_cp_region(new_logprobs)
    new_logprobs = new_logprobs[is_gen_mask]          # 只取 response 部分

    # 2. PPO Clipped Surrogate Loss
    old_logprobs = data.logprobs                      # 采样时的 logprobs
    advantages = data.advantages.clamp(adv_min, adv_max)

    ratio = torch.exp(new_logprobs - old_logprobs)    # 重要性采样比
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages
    pg_loss = -torch.min(surrogate1, surrogate2).mean()

    # 3. KL 正则化（可选）
    ref_logprobs = data.ref_logprobs
    kl_loss = ref_kl_loss_coeff * (new_logprobs - ref_logprobs).mean()      # 反向 KL
    kl_penalty = ref_kl_penalty_coeff * (old_logprobs - ref_logprobs).mean()  # 策略梯度 KL

    loss = pg_loss + kl_loss + kl_penalty
    return loss
```

### 3.3 Critic 损失函数

**代码位置**：`playground/rlvr/qwen3_1p5b_rlvr_math.py:293-309`

```python
def critic_loss_func(data: PackedPPOSamples, outputs):
    values = outputs[:, 0, 0]   # Critic 输出标量值: (S_local,)

    # CP > 1 时需要重建完整序列
    if CP > 1:
        values = gather_from_balanced_cp_region(values)

    # MSE Loss（仅在 response 部分）
    loss = F.mse_loss(values[is_gen_mask], data.returns)
    return loss
```

### 3.4 GAE 计算

**代码位置**：`steptronoss/utils/rl_utils.py:180-192`

```python
def compute_gae(values, rewards, lambd=0.95, gamma=1.0):
    """
    Generalized Advantage Estimation (GAE)

    values:  (B, T) 或 (T,)
    rewards: (B, T) 或 (T,)
    """
    advantages_reversed = []
    lastgaelam = 0
    T = values.shape[-1]

    for t in reversed(range(T)):
        nextvalues = values[..., t + 1] if t < T - 1 else 0
        delta = rewards[..., t] + gamma * nextvalues - values[..., t]  # TD 残差
        lastgaelam = delta + gamma * lambd * lastgaelam                 # 递推
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=-1)
    returns = advantages + values
    return advantages, returns
```

**奖励设置**：每条轨迹只在最后一个 token 处给奖励，其余为 0：

```python
rewards = torch.zeros_like(values)
rewards[-1] = sample.raw_reward  # 终端奖励
```

### 3.5 Critic Warmup 机制

**代码位置**：`steptronoss/exp/rl.py:519`

```python
class PPOLikeTrainerConfig(TrainerConfig):
    critic_warmup_iters: int = 100  # 前 100 步只训练 Critic，不更新 Actor
```

**原理**：训练初期 Critic 的价值估计不准确，如果此时就用这些不准的 advantage 来更新 Actor，会导致策略更新方向偏差。Warmup 期间让 Critic 先学到合理的价值函数基线，再开始 Actor 的策略优化。

---

## 四、Flow Control 异步流控

### 4.1 三层队列架构

**代码位置**：`steptronoss/core/generators/flow_controller.py:50-211`

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   数据源      │     │  pre_gen     │     │  pre_train   │
│ (Dataloader)  │ ──→ │  Queue       │ ──→ │  Queue       │ ──→ 训练
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                         ↑                      ↑
                  _control_worker()      _generation_worker()
                  (调度生成任务)          (执行异步生成)
```

**两个后台线程**：

- **`_control_worker()`**：根据策略调度 prompt 到 `pre_gen` 队列
  - 发送 `Signal("need_version", v)` 通知生成器需要的权重版本
  - 逐个发送 `prompt_per_iter` 个 prompt
  - 发送 `Signal("train")` 标记一轮生成结束

- **`_generation_worker()`**：从 `pre_gen` 取任务，提交到 `GenerationController`
  - 收到 `Signal("need_version", v)` 时等待权重同步完成
  - 收到 `Signal("train")` 时等待所有挂起的生成任务完成
  - 生成结果通过回调放入 `pre_train` 队列

### 4.2 三种调度策略

**代码位置**：`steptronoss/exp/rl.py:452-483`（FlowControllerConfig）

```python
class FlowControllerConfig(Config):
    async_strategy: str = "on-policy"  # "on-policy" | "one-step-off" | "fully-async"
    prompt_per_iter: int = 64          # 每轮生成的 prompt 数
    max_untrained_prompts: int = 128   # fully-async: 最大待训练样本数
    max_staleness: int = 2             # fully-async: 最大陈旧步数
```

| 策略 | 时序 | 特点 |
|------|------|------|
| **on-policy** | `req(0), p0..pN, train, req(1), p0..pN, train, ...` | 每次训练前同步最新权重再生成，数据最新但慢 |
| **one-step-off** | `req(0), p0..pN, train, req(0), p0..pN, train, req(1), ...` | 使用当前权重训练，同时用新权重异步生成下一批 |
| **fully-async** | 持续异步生成，限制 `max_staleness` | 最高吞吐，但数据可能陈旧 |

**代码逻辑**（`_control_worker`，`flow_controller.py:116-137`）：

```python
# on-policy: 每轮请求新版本
scheduled_weight_version = train_weight_version + 1
flow["pre_gen"].put(Signal("need_version", scheduled_weight_version))
for i in range(prompt_per_iter):
    flow["pre_gen"].put(source.pop())
flow["pre_gen"].put(Signal("train"))

# one-step-off: 只在偶数轮请求新版本（延迟一步）
if strategy == "one-step-off":
    scheduled_weight_version = train_weight_version  # 不 +1
```

### 4.3 权重同步机制

**代码位置**：`steptronoss/core/generators/flow_controller.py:102-109`

```python
def sync_weight(self):
    if self.infer_weight_version != self.train_weight_version:
        self.vllm_cfg.deploy_training_model(self.model)
        self.infer_weight_version = self.train_weight_version
```

`deploy_training_model()` 内部流程：
1. `dump_safetensors(hot_path, models)` — 将训练模型权重写入快速存储路径（如 `/oss/tmp/`）
2. `cli.wait_for_server()` — 等待 vLLM 服务就绪
3. `cli.reload_weights(hot_path)` — 通过 HTTP API 触发 vLLM 热加载权重

**版本追踪**：`infer_weight_version` 和 `train_weight_version` 分别追踪推理端和训练端的权重版本，避免重复同步。

### 4.4 持久化队列与检查点恢复

**代码位置**：`steptronoss/utils/rl_utils.py:213-404`

```python
class PersistentQueue:
    """支持 ack/rollback 的队列，用于训练流水线的容错"""

    def pop(self):       # 阻塞弹出
    def get(self):       # 弹出并分配 UUID，存入 pending
    def ack(self, id):   # 确认处理完成，移出 pending
    def state_dict(self):  # 序列化：deque + reverse(pending)
    def load_state_dict(self, d):  # 恢复

class PersistentFlow:
    """多阶段流水线，由多个命名的 PersistentQueue 组成"""
    # 例如: {"source": PQ, "pre_gen": PQ, "pre_train": PQ}
    # 支持原子化的多队列操作（线程锁保护）
```

当训练中断恢复时，`PersistentFlow` 的 `state_dict` 保存了所有队列的完整状态（包括 pending 中未确认的任务），确保不丢失数据也不重复处理。

---

## 五、异步生成控制器

### 5.1 多进程生成架构

**代码位置**：`steptronoss/generation/async_generation.py:115-205`

```
                    主进程
┌──────────────────────────────────┐
│  submit_with_callback(item, cb)  │
│       │                          │
│       ▼                          │
│  ┌─────────┐    ┌─────────────┐  │
│  │ mp.Queue │    │ 回调线程     │  │
│  │ (input)  │    │ 消费结果队列  │  │
│  └────┬────┘    │ 执行 callback │  │
│       │         └──────┬──────┘  │
└───────│────────────────│─────────┘
        │                │
        ▼                ▲
┌───────────────────────────────┐
│     Worker 进程池 (N 个)        │
│                               │
│  ┌──────────────────────────┐ │
│  │ SingleGenerationController│ │
│  │ asyncio 事件循环            │ │
│  │ → work_on_item(item)      │ │
│  │ → aiohttp POST vLLM      │ │
│  └──────────────────────────┘ │
│                               │
│  结果 → mp.Queue (output)     │
└───────────────────────────────┘
```

**关键设计**：
- 每个 Worker 进程运行独立的 `asyncio` 事件循环，支持高并发 HTTP 请求
- 主进程的**回调线程**消费结果队列，在主进程上下文中执行回调（保证线程安全）
- `submit_with_callback()` 分配唯一 `task_id`，通过 `callback_map` 路由结果

### 5.2 Trainable 抽象与奖励计算

**代码位置**：`playground/rlvr/simple_trainable.py:9-116`

```python
class SimpleTrainable(TrainableItem):
    async def generate_for_train(self) -> list[EnvTrajectory]:
        # 1. 调用 vLLM 生成
        response = await aiohttp.post("/v1/completions", json={
            "model": model_name,
            "prompt": prompt_ids,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        })

        # 2. 解析生成结果
        decoded_text = tokenizer.decode(decode_ids)
        predicted = _extract_boxed(decoded_text)  # 提取 \boxed{...} 答案

        # 3. 计算奖励
        is_correct = (predicted == ground_truth)
        raw_reward = 0.0 if truncated else float(is_correct)

        # 4. 构造轨迹
        return [EnvTrajectory(
            trajectory = prompt_ids + decode_ids,
            logprobs = token_logprobs,
            is_gen_mask = [False]*len(prompt) + [True]*len(decode),
            raw_reward = raw_reward,
        )]
```

`TrainableItem` 是可扩展的抽象接口。用户只需实现 `generate_for_train()` 方法，定义如何调用推理服务、解析输出、计算奖励，即可接入 RLVR 训练流程。

---

## 六、vLLM 推理服务架构

### 6.1 Router 负载均衡

**代码位置**：`steptronoss/generation/vllm/vllm_router.py`

```
TrainableItem (aiohttp)
       │
       ▼
┌─────────────────┐
│   VLLMRouter     │
│   (FastAPI)      │
│                  │
│  /register       │ ← vLLM Server 注册
│  /unregister     │ ← vLLM Server 注销
│  /get_info       │ ← 查询服务状态
│  /v1/completions │ ← 代理请求（Round-Robin）
│  /reload_weights │ ← 广播权重重载
└────────┬─────────┘
         │ Round-Robin
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ vLLM 0 │ │ vLLM 1 │  ... (多个 TP 并行实例)
└────────┘ └────────┘
```

### 6.2 权重热部署

**代码位置**：`steptronoss/generation/vllm/vllm_controller.py`

权重热部署流程避免了重启 vLLM 服务的开销：

```
训练完一步
  ↓
dump_safetensors(hot_path, model)    # 写入快速存储（如 NVMe 或 tmpfs）
  ↓
cli.wait_for_server()                 # 等待 vLLM 就绪
  ↓
cli.reload_weights(hot_path)          # HTTP API 触发热加载
  ↓
vLLM 加载新权重（不中断服务）
```

**关键优势**：
- 使用 safetensors 格式，支持零拷贝内存映射加载
- 热路径部署，无需重启推理服务
- 版本号追踪，避免重复同步

---

## 七、内存优化：模型卸载与重载

### 7.1 PackedModel 三级卸载

**代码位置**：`steptronoss/core/trainers/packed_model.py:23-304`

PPO 训练同时需要 Actor、Critic、Reference 三个模型，GPU 显存紧张。`PackedModel` 提供三个独立的卸载维度：

```python
class PackedModel:
    _offloaded = {
        "params": False,          # 模型参数
        "grad_buffer": False,     # 梯度累积缓冲区
        "optimizer_state": False, # 优化器状态（momentum 等）
    }

    # 参数卸载/回载
    def _offload_param(self):     # model.cpu()，释放梯度钩子
    def _backload_param(self):    # model.cuda()，重建梯度钩子

    # 梯度缓冲区卸载/回载
    def _offload_grad_buffer(self):   # 销毁梯度缓冲区（归零）
    def _backload_grad_buffer(self):  # 重建梯度缓冲区

    # 优化器状态卸载/回载
    def _offload_optimizer_state(self):   # 优化器状态 → CPU
    def _backload_optimizer_state(self):  # 优化器状态 → GPU

    # 组合操作
    def offload_model(self):   # 卸载参数 + 梯度缓冲区
    def offload_state(self):   # 卸载优化器状态
    def backload_model(self):  # 回载参数 + 梯度缓冲区
    def backload_state(self):  # 回载优化器状态

    # 上下文管理器
    @contextmanager
    def on_gpu(self):          # 临时加载到 GPU
```

### 7.2 训练流程中的卸载编排

`train_step()` 中的典型卸载/回载序列：

```
┌─ 阶段 3: 前向推理 ────────────────────────────────────┐
│                                                       │
│  actor.offload_model()         # Actor 暂不用         │
│                                                       │
│  reference.backload_model()    # 加载 Reference       │
│  get_reference(samples)        # 计算 ref_logprobs    │
│  reference.offload_model()     # 用完卸载             │
│                                                       │
│  actor.backload_model()        # 加载 Actor           │
│  get_actor_logprob(samples)    # 计算 actor_logprobs  │
│  actor.offload_model()         # 用完卸载             │
│                                                       │
│  critic.backload_model()       # 加载 Critic          │
│  get_values_advantages(samples) # 计算 values + GAE   │
│                                                       │
└───────────────────────────────────────────────────────┘

┌─ 阶段 4: Critic 训练 ────────────────────────────────┐
│  critic 已在 GPU 上                                   │
│  forward_backward(critic_loss)                        │
│  optimizer_step()                                     │
│  critic.offload_state()        # 训练完卸载优化器状态  │
│  critic.offload_model()        # 卸载 Critic 参数     │
└───────────────────────────────────────────────────────┘

┌─ 阶段 5: Actor 训练 ─────────────────────────────────┐
│  actor.backload_model()        # 重新加载 Actor       │
│  actor.backload_state()        # 加载优化器状态        │
│  forward_backward(actor_loss)                         │
│  optimizer_step()                                     │
└───────────────────────────────────────────────────────┘
```

**进阶优化**：`forward_backward()` 支持 `offload_opt_while_forward` 参数，在前向传播期间临时卸载优化器状态（仅 momentum），进一步降低峰值显存：

```python
self.actor.forward_backward(
    data_list=samples,
    loss_fn=actor_loss_func,
    offload_opt_while_forward="momentum",  # 前向时卸载 momentum
)
```

### 7.3 显存预算估算

对于参数量为 N 的 BF16 模型（单个角色）：

| 组成部分 | 显存 | 说明 |
|---------|------|------|
| 模型参数 | 2N | BF16 存储 |
| 梯度累积缓冲区 | 4N | FP32 累积 |
| 优化器状态 | 12N | FP32: param copy + momentum1 + momentum2 |
| **合计** | **18N** | 或 **18N / DP_size**（ZeRO-1） |

三个模型的总需求为 54N（不考虑 ZeRO 和激活）。通过卸载编排，峰值显存约为 18N（单个模型训练态），远低于同时加载三个模型的需求。

---

## 八、变长序列打包优化

### 8.1 打包策略

**代码位置**：`steptronoss/exp/rl.py:131-182`（`PackedPPOSamples.from_samples()`）

RLVR 中每条生成的轨迹长度各不相同。Packing 将多条轨迹连接为一个长序列，通过 `cu_seqlens` 记录边界：

```python
@classmethod
def from_samples(cls, samples: list[PPOSample]) -> PackedPPOSamples:
    # 1. 连接所有 labels / trajectories
    packed_data.labels = torch.cat(labels, 0)

    # 2. 计算累积长度
    seqlens = torch.tensor([len(s.trajectory) for s in samples])
    cu_seqlens = torch.cat([zeros(1), torch.cumsum(seqlens, 0)])

    # 3. 对齐填充
    num_pad = cu_seqlens[-1] % (TP * CP * 2)
    if num_pad != 0:
        num_pad = TP * CP * 2 - num_pad

    # 4. 记录有效长度（padding 前）和填充后长度
    packed_data.cu_valid_sizes = cu_seqlens.clone()    # 真实长度
    cu_seqlens[-1] += num_pad                          # 填充后长度
    packed_data.cu_seqlens = cu_seqlens
```

**结果示例**（3 条轨迹，长度分别为 100, 150, 80）：

```
cu_valid_sizes = [0, 100, 250, 330]      # 真实边界
cu_seqlens     = [0, 100, 250, 336]      # 填充到 TP*CP*2 的倍数（假设 =16）
input_ids      = [tok0...tok329, pad*6]  # 尾部 6 个 padding token
```

### 8.2 数据均衡分配

**代码位置**：`steptronoss/core/trainers/ppo_trainer.py:246-292`

```python
def data_balance_and_pack(self, all_samples):
    # 1. 按 max_seq_len 分组（确保同组样本长度接近）
    grouped = _group_samples(all_samples, max_seq_len)

    # 2. 跨 DP rank 均衡分配
    my_groups = balanced_list_split(grouped, dp_size)[dp_rank]

    # 3. 每组打包为 PackedPPOSamples
    return [PackedPPOSamples.from_samples(group) for group in my_groups]
```

`_group_samples()` 贪心地将样本按序填入固定大小的 chunk（不超过 `max_seq_len`），确保每个打包后的序列长度接近，提高 GPU 利用率。

### 8.3 TP × CP × 2 对齐约束

打包后的总长度必须对齐到 `TP × CP × 2` 的倍数。这个约束来自两层需求的叠加：

1. **CP Balanced 互补分片**：序列被分成 `2 × CP` 个 chunk，每个 CP rank 取首尾各一个 → 需要 `S % (2 × CP) == 0`
2. **SP（Sequence Parallel）均匀切分**：切分后的子序列需被 TP 整除

合在一起：`S % (TP × CP × 2) == 0`

对应 `scatter_to_balanced_cp_region` 中的断言（`context_parallel.py:132-134`）：

```python
assert S % (2 * cp_size * tp_size) == 0,
    f"size ({S}) should be divisible by 2 * context parallel size ({cp_size}) x sequence parallel size({tp_size})"
```

下游 loss 计算通过 `cu_valid_sizes` 和 `is_gen_mask` 忽略 padding 部分。

---

## 九、并行策略在 RLVR 中的适配

### 9.1 Context Parallel 与 GAE 的兼容

**代码位置**：`steptronoss/core/trainers/ppo_trainer.py:376-386`

GAE 需要完整序列的 values 进行逆序递推，但 CP 模式下每个 rank 只有部分序列。解决方法是在 GAE 计算前**聚合完整 values**：

```python
def get_values_advantages(self, my_samples):
    # Critic 前向得到 local values
    values = critic_forward(samples)  # shape: (S_local,)

    # CP > 1 时聚合完整序列
    cp_size = PM.size_of("CP")
    if cp_size > 1:
        values = gather_from_balanced_cp_region(values, dim=0)
        # shape: (S_local,) → (S_cp,) → (S,)  AllGather + 互补重排

    # 现在 values 是完整序列，可以正确计算 GAE
    advantages, returns = compute_gae(values, rewards, lambda=0.95, gamma=1.0)
```

### 9.2 分布式 Loss 计算

RLVR 中提取 logprobs 需要兼顾 TP 和 CP 两个并行维度：

```python
def get_reference(samples):
    def hacked_loss_fn(samples, logits):
        # 1. CP 切分 logits（本地操作，无通信）
        logits = scatter_to_balanced_cp_region(logits)

        # 2. TP 分布式交叉熵（vocab 维度切分在 TP 组内）
        ref_logprobs = -vocab_parallel_cross_entropy(logits, labels)[0]

        # 3. CP 聚合 logprobs（AllGather + 互补重排）
        ref_logprobs = gather_from_balanced_cp_region(ref_logprobs)

        # 4. 只保留 response 部分
        samples.ref_logprobs = ref_logprobs[is_gen_mask]
        return 0  # 不计算实际 loss
```

这种 **"Hacked Loss"** 模式巧妙地利用模型前向传播的 logits 提取所需指标，而不进行实际的反向传播。

### 9.3 数据源 Rank 选择

**代码位置**：`steptronoss/exp/rl.py:595-634`

```python
def is_data_source(self):
    """只在特定 rank 上构建数据加载器"""
    return (
        (PM.i_am("PP", 0) or PM.i_am("PP", -1))  # PP 首/尾 stage
        and PM.i_am("TP", 0)   # TP 组内第 0 号
        and PM.i_am("CP", 0)   # CP 组内第 0 号
    )
```

数据在 PP0（或 PP-1，取决于是否需要 labels）上加载，然后广播到其他 PP stage。TP 和 CP 组内通过 broadcast 同步，避免每个 rank 都构建独立的 dataloader。

---

## 十、指标与监控系统

### 10.1 指标分类

**代码位置**：`steptronoss/exp/rl.py:204-424`（PPOMetricConfig）

PPOMetricConfig 定义了 60+ 个监控指标，覆盖训练的各个维度：

| 类别 | 指标示例 | 用途 |
|------|---------|------|
| **奖励** | `reward_mean/max/min` | 策略质量评估 |
| **正确率** | `correctness`（细分 `all_accept/all_fail/some_accept/some_reward`） | 任务完成情况 |
| **生成质量** | `rollout_avg_tokens`, `rollout_avg_think_tokens`, `rollout_avg_solution_tokens` | 生成长度分析（区分 `</think>` 前后） |
| **重复检测** | `repeatness_score`, `repeatness_rate` | 检测模型退化（重复输出） |
| **PPO 诊断** | `ppo_clip_count`, `ppo_value_clip_count` | 裁剪比例监控 |
| **Advantage 统计** | `advantage_mean/std/max/min` | 基线质量诊断（均值应接近 0） |
| **生成性能** | `gen_rollout_latency_s`, `gen_rollout_throughput_tps` | E2E 延迟和吞吐 |
| **截断率** | `truncated_rate` | 达到 max_length 被截断的比例 |
| **文本样本** | `training_generation_text(20)`, `eval_generation_text(10)` | 定性分析 |

### 10.2 离策略诊断

```python
# 重要性采样比
ratio_diff = π_new / π_old - 1                    # 过大说明策略偏移严重
ratio_diff_quantile = [p01, p05, p10, p90, p95, p99]  # 分位数分布

# KL 散度
kl_with_ref_dist = KL(π_new || π_ref)             # 与参考策略的偏离
kl_with_sample_dist = KL(π_new || π_sample)        # 与采样策略的偏离

# logprob 差异
sampling_logprob_diff = log π_new - log π_old      # 新旧策略的 logprob 差
sampling_logprob_diff_quantile = [p01...p99]
```

这些指标帮助诊断训练是否偏离安全区域：如果 `ratio_diff` 过大或 `kl_with_ref_dist` 增长过快，可能需要降低学习率或增大 KL 惩罚系数。

---

## 十一、检查点与容错

### 11.1 多角色检查点

**代码位置**：`steptronoss/exp/rl.py:196-202`

```python
class PPOCheckpointCfg(Config):
    actor: RoleCheckpointConfig      # Actor 的 load_path, load_safetensors, strict_load
    critic: RoleCheckpointConfig     # Critic 的 load_path（可从 Actor 初始化）
    reference: RoleCheckpointConfig  # Reference 的 load_path（通常与 Actor 相同）
```

每个角色可以独立配置检查点来源，支持：
- Actor 和 Reference 加载同一个预训练权重
- Critic 从 Actor 初始化（共享 backbone），输出层不同
- 各角色从不同的训练阶段恢复

### 11.2 自动恢复

```python
def set_autoresume(self):
    """从 latest_ckpt 文件自动恢复训练"""
    if os.path.exists(latest_ckpt_file):
        latest_ckpt = read(latest_ckpt_file)
        cfg.actor.load_path = join(latest_ckpt, "actor")
        cfg.critic.load_path = join(latest_ckpt, "critic")
        cfg.reference.load_path = join(latest_ckpt, "reference")
        # 恢复所有加载选项
        cfg.actor.load_option.all()
        cfg.critic.load_option.all()
        cfg.reference.load_option.all()
```

### 11.3 样本保存与重放

**代码位置**：`steptronoss/utils/rl_utils.py:95-177`

```python
class RaggedPPOSampleDumper:
    """将 PackedPPOSamples 解包并保存为可序列化格式"""

    def _unpack_samples(self, packed_data, dump_keys):
        # 通过 cu_seqlens 索引还原每条样本
        # 通过 is_gen_mask 提取 response 部分
        # 恢复 ref_logprobs, actor_logprobs, advantages, returns, values
        return unpacked_samples

    def __call__(self, samples_list, dump_keys) -> list[dict]:
        # 解包所有 packed samples → 转为 dict → 序列化保存
```

保存的样本可用于：
- 离线分析（奖励分布、advantage 分布等）
- 重放训练（调试或消融实验）
- 可视化生成质量

---

## 十二、总结对照表

| 设计/优化 | 类别 | 核心效果 | 代码位置 |
|-----------|------|---------|---------|
| **训练-推理分离** | 架构 | 训练用 Megatron，推理用 vLLM，各取所长 | `vllm_controller.py` |
| **Flow Control** | 调度 | on-policy / one-step-off / fully-async 三种策略 | `flow_controller.py` |
| **多进程异步生成** | 调度 | 生成不阻塞训练主线程，多进程并发 HTTP 请求 | `async_generation.py` |
| **权重热部署** | 架构 | safetensors → vLLM reload，无需重启服务 | `vllm_controller.py` |
| **三级 CPU Offload** | 显存 | 参数/梯度缓冲区/优化器状态独立卸载 | `packed_model.py` |
| **变长序列 Packing** | 计算效率 | 多条轨迹拼接，cu_seqlens 记录边界，减少 padding 浪费 | `rl.py:from_samples()` |
| **数据均衡分配** | 计算效率 | 跨 DP rank 均匀分配样本，避免负载倾斜 | `ppo_trainer.py` |
| **TP × CP × 2 对齐** | 正确性 | 满足 SP 均匀切分 + CP 互补分片的联合约束 | `rl.py:158-160` |
| **CP 下 GAE 聚合** | 正确性 | AllGather values 后再做逆序递推 | `ppo_trainer.py:376-386` |
| **Hacked Loss** | 设计模式 | 利用前向 logits 提取 logprobs，无需反向传播 | `ppo_trainer.py:295-325` |
| **Critic Warmup** | 训练稳定性 | 前 N 步只训练 Critic，让价值估计先收敛 | `rl.py:519` |
| **持久化队列** | 容错 | PersistentQueue 支持 ack/rollback 和检查点恢复 | `rl_utils.py:213-404` |
| **多角色检查点** | 容错 | Actor/Critic/Reference 独立保存和恢复 | `rl.py:196-202` |
| **100+ 监控指标** | 可观测性 | 奖励、正确率、KL、clip 率、生成延迟等全面覆盖 | `rl.py:204-424` |
| **Trainable 抽象** | 扩展性 | 用户只需实现 generate_for_train()，即可接入新任务 | `simple_trainable.py` |
