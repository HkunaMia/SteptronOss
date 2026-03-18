# StepTronOSS 预训练流程优化详解

> 本文档详细介绍 StepTronOSS 框架在 LLM 预训练流程中的各项优化手段。

---

## 1. 数据加载与预处理优化

### 1.1 DataRecipe - 多领域数据混合采样

**文件**: `steptronoss/data/recipe.py`

**核心功能**:
- 支持多领域数据（如 wiki、code、chat）的混合训练
- 通过 `epochs` 配置控制各领域数据的采样比例
- 支持数据子采样 (`subsample_rate`)

```python
@dataclass
class DataRecipe:
    domains: dict[str, list[DataSourceFile]]  # 多领域数据源
    epochs: dict[str, float]                  # 各领域采样轮数

# 使用示例
recipe = DataRecipe(
    domains={
        "wiki": [DataSourceFile("/data/wiki.jsonl", subsample_rate=0.8)],
        "code": [DataSourceFile("/data/code.jsonl", subsample_rate=1.0)],
    },
    epochs={"wiki": 2.0, "code": 3.0}  # code 数据采样更多轮
)
```

**优化点**:
- **加权随机采样**: 根据 `epochs` 自动计算采样权重
- **数据子采样**: 支持对大数据集进行比例采样，减少 I/O

### 1.2 CompiledDataset - 数据预编译加速

**文件**: `steptronoss/data/datasets/compile_dataset.py`

**核心功能**:
- 将原始数据集预编译为优化的二进制格式
- 预计算样本元数据（如长度），避免运行时重复计算

```python
# 编译数据集
compile_dataset(
    dataset,
    save_path="/data/compiled/",
    sample_meta_extractor=lambda x: len(x['tokens'])  # 预提取样本长度
)

# 使用编译后的数据集
compiled_ds = CompiledDataset("/data/compiled/")
```

**优化点**:
- **启动加速**: 避免训练启动时遍历整个数据集计算长度
- **内存优化**: 按需加载样本元数据，减少内存占用

### 1.3 MixedPackedDataloader - 序列打包

**文件**: `steptronoss/data/dataloader/packed_dataloader.py`

**核心功能**:
- 将多个短序列打包到同一个 `max_length` 序列中
- 支持两种策略: `sequential` 和 `random` 采样

```python
dataloader = MixedPackedDataloader(
    datasets=[ds1, ds2, ds3],
    epochs=[2.0, 1.0, 3.0],           # 各领域采样比例
    max_length=32768,                  # 打包后的序列长度
    oversize_policy="extend",          # 超长序列处理: extend 或 drop
    dataset_sampling="random",         # 采样策略
)
```

**打包算法** (Non-truncation Packing):
```
原始序列: [A:1000 tokens], [B:800 tokens], [C:1500 tokens], ...
打包结果: [A+B+C:3300 tokens], [D+E+F:...], ...

优化效果:
- 减少 Padding: 从 30-40% 降至 <5%
- 提高 GPU 利用率: 更少的无效计算
- 支持变长训练: 同一 batch 内序列长度不同
```

**优化点**:
- **两级采样**:
  1. 领域间采样: 按 `epochs` 权重随机选择领域
  2. 领域内采样: 顺序或随机选择样本
- **高效打包**: 使用非截断算法，最大化序列利用率

---

## 2. 训练核心优化

### 2.1 梯度累积与微批次

**文件**: `steptronoss/exp/ntp.py:78`

```python
class NTPTrainerConfig(TrainerConfig):
    micro_batch_size: int    # 单卡前向的批次大小
    global_batch_size: int   # 全局有效批次大小
```

**梯度累积计算**:
```python
grad_accumulation_steps = global_batch_size // (dp_size * micro_batch_size)
```

**优化点**:
- **显存友好**: 小 `micro_batch_size` 减少激活值内存
- **大 batch 训练**: 通过累积实现大 batch 效果
- **流水线并行**: 与 PP 配合，隐藏通信延迟

### 2.2 上下文并行 (Context Parallel)

**文件**: `steptronoss/core/context_parallel/`

**核心功能**:
- 将长序列切分到多个 GPU 上并行计算
- 支持 Ring Attention 变体

```python
# 配置
parallel_cfg.context_parallel_size = 8

# 训练长序列
trainer_cfg.global_seq_length = 128 * 1024  # 128K
```

**优化点**:
- **扩展序列长度**: 支持 128K+ 长序列训练
- **显存节省**: 每卡只存储部分序列的激活值
- **通信优化**: 使用高效的 All-Gather/Reduce-Scatter

### 2.3 学习率调度策略

**文件**: `steptronoss/exp/lr_schedulers.py`

支持多种调度策略:

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **Constant** | 恒定学习率 + warmup | 短周期实验 |
| **Linear** | 线性衰减 | 标准预训练 |
| **Cosine** | 余弦退火 | 大模型训练 |

```python
class CosineSchedulerConfig(SchedulerConfig):
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_schedule: float = 2000  # warmup 步数
    total_schedule: float = 100000 # 总调度步数
    scheduler_unit: str = "iter"   # iter/sample/token
```

**优化点**:
- **灵活 warmup**: 支持按迭代/样本/token 计数
- **最小学习率裁剪**: 避免学习率过小导致训练停滞

---

## 3. 模型计算优化

### 3.1 Flash Attention

**文件**: `steptronoss/model/common/attention_core.py`

```python
# 启用 Flash Attention
set_optimization(AttentionCore="flash-attn")

# 支持变长序列
flash_attn_varlen_func(q, k, v, cu_seqlens, max_seqlen)
```

**优化效果**:
- **显存节省**: 从 O(N²) 降至 O(N)
- **计算加速**: 2-4x Attention 速度提升
- **支持长序列**: 配合 CP 支持 128K+ 序列

### 3.2 梯度检查点 (Gradient Checkpointing)

**文件**: `steptronoss/exp/base_exp.py:44`

```python
model_cfg.recompute = ["attention", "feed_forward"]
# 或
model_cfg.recompute = True  # 启用全部
```

**策略**:
- **细粒度控制**: 可选择性重计算特定层
- **显存换计算**: 牺牲 ~20% 计算换取 50%+ 显存节省

### 3.3 torch.compile

**文件**: `steptronoss/utils/optimizable.py`

```python
def configure_optimizable(self):
    set_optimization(default="torch_compile")
```

**优化效果**:
- **图优化**: 自动算子融合
- **减少 Python 开销**: 特别是对于小模型

---

## 4. 检查点与恢复优化

### 4.1 异步保存 (Async Checkpoint)

**文件**: `steptronoss/checkpointing/local_checkpoint.py`

```python
checkpoint_cfg.async_dump = True  # 异步保存，不阻塞训练
checkpoint_cfg.save_interval = 1000  # 每 1000 步保存
```

**优化点**:
- **非阻塞**: 保存检查点时不暂停训练
- **独立线程**: 在后台线程执行 I/O

### 4.2 分布式检查点 (Distributed Checkpoint)

**文件**: `steptronoss/checkpointing/`

```python
checkpoint_cfg.use_distributed_optimizer = True
checkpoint_cfg.reshard_optimizer_state = True  # 支持 DP 变化时恢复
```

**优化点**:
- **并行保存**: 每卡保存自己的状态
- **自动 Reshape**: 支持不同并行配置间的检查点转换

### 4.3 灵活加载选项

```python
# 只加载模型权重
checkpoint_cfg.load_option.none(but=["model"])

# 不加载优化器状态
checkpoint_cfg.load_option.all(but=["optimizer"])

# 从 HuggingFace 格式加载
checkpoint_cfg.load_safetensors = "/path/to/hf/model"
```

---

## 5. 数据并行优化

### 5.1 Zero1 Gradient Manager

**文件**: `steptronoss/optimizer/zero1_gradient_manager.py`

```python
optimizer_cfg.use_distributed_optimizer = True
```

**优化点**:
- **ZeRO-1**: 优化器状态分片，节省显存
- **梯度复用**: 复用 DDP 梯度缓冲区

### 5.2 梯度累积融合

**文件**: `steptronoss/exp/base_exp.py:366`

```python
tp_cfg.gradient_accumulation_fusion = True  # 需要 APEX
```

**优化点**:
- **融合 kernel**: 梯度累积与 All-Reduce 融合
- **减少内存搬运**

---

## 6. 性能监控与调试

### 6.1 内存追踪 (CMT)

**文件**: `steptronoss/utils/memory_tracker.py`

```bash
export MEM_DIAGNOSE=1  # 启用内存追踪
```

```python
from steptronoss.utils.memory_tracker import CMT

with CMT.record("forward_pass"):
    output = model(input)
```

**输出**:
```
[CMT] after_build_model: 12.5 GB
[CMT] after_forward_backward: 18.2 GB
[CMT] after_optimizer_step: 15.1 GB
```

### 6.2 计时器 (Timers)

**文件**: `steptronoss/timers.py`

```python
with self.timers.record("forward-backward", log_level=1):
    pp_scheduler.run(grad_accumulation_steps)
```

**输出**:
```
[Timer] forward-backward: 245.3 ms
[Timer] optimizer-step: 12.5 ms
[Timer] data-loading: 5.2 ms
```

### 6.3 全局指标 (GlobalMetrics)

**文件**: `steptronoss/utils/metrics.py`

```python
# 自动聚合跨并行维度的指标
GlobalMetrics.lm_loss.add(loss)        # 跨 DP/EP 平均
GlobalMetrics.acc.add(acc)             # 跨 TP 求和
GlobalMetrics.moe_aux_loss.add(aux_loss)  # 跨 PP 求和
```

---

## 7. 预训练最佳实践

### 7.1 推荐配置 (7B 模型)

```python
class Exp(PretrainExp):
    def __init__(self):
        super().__init__()
        
        # 1. 数据配置
        self.data_cfg.max_length = 4096
        self.data_cfg.dataset_sampling = "random"
        
        # 2. 训练配置
        self.trainer_cfg.global_batch_size = 2048
        self.trainer_cfg.micro_batch_size = 2
        self.trainer_cfg.global_seq_length = 4096
        
        # 3. 并行配置
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 2
        self.model_cfg.parallel_cfg.pipeline_model_parallel_size = 4
        self.model_cfg.parallel_cfg.context_parallel_size = 1
        
        # 4. 优化配置
        self.model_cfg.recompute = ["attention", "feed_forward"]
        self.optimizer_cfg.use_distributed_optimizer = True
        
        # 5. 学习率
        self.scheduler_cfg.lr = 3e-4
        self.scheduler_cfg.min_lr = 3e-5
        self.scheduler_cfg.warmup_schedule = 2000
        
        # 6. 检查点
        self.checkpoint_cfg.async_dump = True
        self.checkpoint_cfg.save_interval = 1000
    
    def configure_optimizable(self):
        set_optimization(
            default="torch_compile",
            AttentionCore="flash-attn",
        )
```

### 7.2 长序列训练配置 (128K)

```python
class LongContextExp(PretrainExp):
    def __init__(self):
        super().__init__()
        
        # 关键: 使用 Context Parallel
        self.model_cfg.parallel_cfg.context_parallel_size = 8
        self.trainer_cfg.global_seq_length = 128 * 1024
        
        # 启用 Flash Attention
        self.model_cfg.use_flash_attn = True
        
        # 梯度检查点必备
        self.model_cfg.recompute = True
```

### 7.3 MoE 模型配置

```python
class MoEPretrainExp(PretrainExp):
    def __init__(self):
        super().__init__()
        
        # MoE 配置
        self.model_cfg.moe_num_experts = 64
        self.model_cfg.moe_top_k = 2
        
        # EP 并行
        self.model_cfg.parallel_cfg.expert_model_parallel_size = 8
        self.model_cfg.parallel_cfg.expert_tensor_parallel_size = 1
        
        # 负载均衡
        self.model_cfg.enable_auxiliary_loss_free_load_balance = True
        
        # Grouped GEMM 加速
        self.configure_optimizable = lambda: set_optimization(
            grouped_gemm="nv_grouped_gemm"
        )
```

---

## 8. 性能调优检查清单

### 8.1 启动前检查

- [ ] 使用 `cfshow` 验证配置
- [ ] 确保 `sanity_check()` 通过
- [ ] 数据集已编译 (`CompiledDataset`)
- [ ] 检查 GPU 拓扑和 NVLink

### 8.2 训练中监控

- [ ] GPU 利用率 (>90%)
- [ ] 显存占用 (避免 OOM)
- [ ] 数据加载延迟 (<10% 总时间)
- [ ] 梯度范数 (检查爆炸/消失)

### 8.3 优化启用检查

- [ ] Flash Attention 已启用
- [ ] 梯度检查点配置合理
- [ ] 异步检查点已开启
- [ ] torch.compile 已启用 (如适用)

---

## 9. 关键文件索引

| 功能 | 文件 |
|------|------|
| 数据配方 | `steptronoss/data/recipe.py` |
| 数据编译 | `steptronoss/data/datasets/compile_dataset.py` |
| 打包加载 | `steptronoss/data/dataloader/packed_dataloader.py` |
| 预训练配置 | `steptronoss/exp/ntp.py` |
| 学习率调度 | `steptronoss/exp/lr_schedulers.py` |
| 检查点配置 | `steptronoss/exp/checkpointing.py` |
| 上下文并行 | `steptronoss/core/context_parallel/` |
| Attention 优化 | `steptronoss/model/common/attention_core.py` |
| 内存追踪 | `steptronoss/utils/memory_tracker.py` |

---

> 文档版本: 2025-03
> 基于 StepTronOSS 预训练实现整理
