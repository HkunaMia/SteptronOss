# StepTronOSS 框架指南

> 本文档总结了 StepTronOSS 训练框架的核心结构、训练流程和优化技术。

---

## 1. 框架概述

**StepTronOSS** 是一个轻量级的大规模语言模型训练框架，专注于：
- 模块化、配置驱动的实验设计
- 可复现的训练工作流（SFT、RLVR、Pretrain）
- 多任务编排与灵活启动工具
- 可扩展的数据/优化器/模型栈

### 1.1 与 TorchTitan 的对比

| 特性 | StepTronOSS | TorchTitan |
|------|-------------|------------|
| 配置方式 | Python 类 (`configurize`) | TOML 文件 |
| 并行策略 | 自定义 TP/PP/DP/CP/EP | PyTorch DTensor/DeviceMesh |
| 工作流支持 | SFT/RLVR/Pretrain 原生支持 | 主要聚焦预训练 |
| 推理部署 | 内置 vLLM 集成 | 训练为主 |
| 数据系统 | DataRecipe + 编译 | HuggingFace datasets |

---

## 2. 代码结构

```
steptronoss/                    # 核心包
├── core/                       # 并行和训练基础设施
│   ├── parallel_state.py       # 全局 PM (ParallelManager) 管理 TP/PP/DP/CP/EP
│   ├── trainers/               # Trainer 实现
│   │   ├── base_trainer.py     # 基础 Trainer 接口
│   │   ├── lm_trainer.py       # 预训练/SFT Trainer (DecoderPretrainTrainer)
│   │   └── ppo_trainer.py      # RLVR Trainer
│   ├── pipeline_parallel/      # PP/VPP 调度器
│   ├── tensor_parallel/        # TP 工具 (Megatron-style)
│   └── context_parallel/       # CP 实现
│
├── model/                      # 模型架构
│   ├── qwen_dense.py           # Qwen 稠密模型
│   ├── step3p5.py              # Step3.5 MoE 模型
│   ├── decoder_model.py        # 基础 Decoder 抽象
│   ├── common/                 # 共享组件
│   │   ├── attention_core.py   # Flash Attention 实现
│   │   ├── moe_block.py        # MoE 块实现
│   │   ├── feed_forward.py     # FFN (SwiGLU)
│   │   └── grouped_query_attention.py  # GQA
│   ├── ep_dispatcher/          # Expert Parallel 分发器 (DeepEP)
│   └── optimizations/          # Triton 优化内核
│
├── exp/                        # 实验配置
│   ├── base_exp.py             # BaseExp, TrainerConfig 等核心配置
│   ├── ntp.py                  # 预训练实验 (PretrainExp)
│   ├── sft.py                  # SFT 实验
│   ├── rl.py                   # RLVR/PPO 实验
│   ├── optimizer.py            # 优化器配置 (Adam, Muon)
│   ├── lr_schedulers.py        # 学习率调度器
│   └── checkpointing.py        # 检查点配置
│
├── data/                       # 数据加载和处理
│   ├── recipe.py               # DataRecipe 数据编译
│   ├── dataloader/
│   │   └── packed_dataloader.py  # 序列打包
│   └── packing/                # 打包算法
│
├── optimizer/                  # 优化器和梯度管理
│   ├── gradient_manager.py
│   ├── zero1_gradient_manager.py  # ZeRO-1 实现
│   └── muon.py                 # Muon 优化器
│
├── checkpointing/              # 检查点保存/加载/变形
│   ├── local_checkpoint.py
│   ├── reshape_ops.py          # 检查点 reshape 操作
│   └── hf_checkpoint.py
│
└── utils/                      # 工具
    ├── optimizable.py          # @optimizable 装饰器
    ├── memory_tracker.py       # CMT 内存追踪
    └── metrics.py              # 指标系统

playground/                     # 实验配置
├── pretrain/                   # 预训练实验
├── sft/                        # SFT 实验
├── rlvr/                       # RLVR 实验
└── tools/                      # 数据编译工具
```

---

## 3. 配置系统

### 3.1 核心概念

StepTronOSS 使用 `configurize` 库实现声明式配置：

```python
from configurize import Config, Ref

class MyConfig(Config):
    param_a: int                    # 必填字段
    param_b: float = 1.0           # 默认值
    nested_cfg: SubConfig = SubConfig  # 子配置
    
    # 跨节点引用
    derived_value: int = Ref("..parent_value")
    
    def build(self):
        """构建运行时对象"""
        return MyObj(cfg=self)
    
    def sanity_check(self):
        """验证配置"""
        super().sanity_check()
        assert self.param_b > 0
```

### 3.2 Ref 引用机制

```python
class Step3p5FlashMoEConfig(MoEConfig):
    def __init__(self):
        # "..." 表示向上2级引用
        self.tp_cfg = Ref("...tp_cfg")           # -> Step3p5FlashModelConfig.tp_cfg
        self.hidden_size = Ref("...hidden_size") # -> Step3p5FlashModelConfig.hidden_size
        
        # ".." 表示向上1级引用
        self.activation = Ref("..activation")    # -> MoEFeedForwardConfig.activation
```

### 3.3 配置验证工具

```bash
# 查看完整配置树
uv run cfshow playground/sft/qwen3/qwen3_1p7b_sft_step3_data.py

# 查看特定子配置
uv run cfshow playground/sft/qwen3/qwen3_1p7b_sft_step3_data.py -k model_cfg

# 命令行覆盖配置
uv run tools/mp_run.py your_exp.py trainer_cfg.lr=1e-4
```

---

## 4. 训练流程

### 4.1 实验入口

以 SFT 为例 (`playground/sft/qwen3/qwen3_1p7b_sft_step3_data.py`):

```python
class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig
    data_cfg = Recipe0311QwenCompiledSFTDataConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 128 * 1024
        self.scheduler_cfg.lr = 1e-5
        self.checkpoint_cfg.load_safetensors = "/path/to/model/"

if __name__ == "__main__":
    Exp().train()
```

### 4.2 训练流程图

```
Exp.train()
    │
    ├── update_from_args()      # 解析命令行参数
    ├── sanity_check()          # 配置验证
    ├── trainer = DecoderPretrainTrainer(exp)
    ├── configure_optimizable() # 设置优化选项
    └── trainer.train()
            │
            ├── before_train()
            │   ├── PM.initialize()          # 初始化分布式
            │   ├── PM.set_mesh()            # 设置并行策略
            │   ├── setup_model()            # 构建模型
            │   ├── build_gradient_manager() # 构建优化器
            │   ├── build_dataloader()       # 构建数据加载器
            │   └── load_checkpoint()        # 加载检查点
            │
            ├── train_loop()
            │   └── while iter < max_iters:
            │       ├── train_step()
            │       │   ├── zero_grad()
            │       │   ├── pp_scheduler.run()  # 前向+反向
            │       │   ├── grad_manager.step() # 参数更新
            │       │   └── scheduler.step()    # 更新学习率
            │       ├── logging()
            │       └── save_checkpoint()
            │
            └── after_train()
                └── save_final_checkpoint()
```

### 4.3 核心训练步骤

```python
# steptronoss/core/trainers/lm_trainer.py
def train_step(self):
    # 1. 清零梯度
    self.grad_manager.zero_grad()
    
    # 2. 计算梯度累积步数
    grad_accumulation_steps = (
        global_batch_size / dp_size / micro_batch_size
    )
    
    # 3. 配置流水线调度器
    pp_scheduler.configure(models, data_iterators, loss_fn, ...)
    
    # 4. 前向+反向传播 (由 PP Scheduler 管理)
    pp_scheduler.run(grad_accumulation_steps)
    
    # 5. 参数更新
    update_successful, grad_norm, _ = self.grad_manager.step()
    
    # 6. 更新学习率
    self.opt_param_scheduler.step(increment=...)
    
    # 7. 更新 MoE 路由偏置
    MoEBlock.update_router_balance_bias_per_gbs(self.models)
```

### 4.4 启动方式

```bash
# 单节点单任务 (SFT)
uv run torchrun playground/sft/your_exp.py

# 多任务编排 (RLVR)
export STEPTRON_MEET_DIR=/path/to/shared
uv run tools/mp_run.py playground/rlvr/your_exp.py

# 生成多节点启动脚本
uv run tools/build_scripts.py your_exp.py /mnt/entrypoints/
```

---

## 5. 训练优化项

### 5.1 并行优化

| 优化项 | 配置/位置 | 说明 |
|--------|-----------|------|
| **Tensor Parallel (TP)** | `parallel_cfg.tensor_model_parallel_size` | Megatron-style TP |
| **Pipeline Parallel (PP)** | `parallel_cfg.pipeline_model_parallel_size` | 1F1B 调度 |
| **Virtual PP (VPP)** | `parallel_cfg.virtual_pipeline_model_parallel_size` | 同一 GPU 多 stage |
| **Context Parallel (CP)** | `parallel_cfg.context_parallel_size` | 长序列切分 |
| **Expert Parallel (EP)** | `parallel_cfg.expert_model_parallel_size` | MoE 专家并行 |
| **Async TP AllReduce** | `tp_cfg.async_tensor_model_parallel_allreduce` | 异步通信重叠 |

### 5.2 内存优化

| 优化项 | 配置/位置 | 说明 |
|--------|-----------|------|
| **梯度检查点** | `model_cfg.recompute` | 细粒度控制: `["attention", "feed_forward"]` |
| **Pipeline Activation Offload** | `model_cfg.pipeline_activation_cpu_offload` | PP 激活值 offload 到 CPU |
| **Optimizer State Offload** | `trainer_cfg.offload_optimizer_state` | 优化器状态 CPU  offload |
| **Float16Module** | `lm_trainer.py:352` | 自动混合精度管理 |
| **Zero1 Gradient Manager** | `zero1_gradient_manager.py` | ZeRO-1 数据并行 |

### 5.3 计算优化

| 优化项 | 配置/位置 | 说明 |
|--------|-----------|------|
| **Flash Attention** | `set_optimization(AttentionCore="flash-attn")` | 高效 Attention 计算 |
| **Grouped GEMM** | `set_optimization(grouped_gemm="nv_grouped_gemm")` | MoE 专家计算融合 |
| **torch.compile** | `set_optimization(default="torch_compile")` | 图编译优化 |
| **SwiGLU Clipping** | `moe_cfg.expert_swiglu_limits` | 激活值裁剪防梯度爆炸 |
| **梯度累加融合** | `tp_cfg.gradient_accumulation_fusion` | APEX 融合优化 |

### 5.4 通信优化

| 优化项 | 配置/位置 | 说明 |
|--------|-----------|------|
| **DeepEP** | `model/ep_dispatcher/deepep_dispatcher.py` | MoE dispatch/combine 优化 |
| **Async Checkpoint** | `checkpoint_cfg.async_dump` | 异步保存检查点 |

### 5.5 数据优化

| 优化项 | 配置/位置 | 说明 |
|--------|-----------|------|
| **Sequence Packing** | `packed_dataloader.py` | 多序列打包减少 padding |
| **DataRecipe 编译** | `recipe.py` | 预编译数据索引加速启动 |

### 5.6 MoE 特有优化

| 优化项 | 配置/位置 | 说明 |
|--------|-----------|------|
| **Aux-Loss-Free Load Balance** | `moe_block.py:236` | 无辅助损失负载均衡 |
| **Router Bias FP32** | `moe_block.py:423` | 保持 router bias FP32 精度 |
| **Expert 特定 Swiglu Limit** | `moe_cfg.expert_swiglu_limits` | 层特定激活裁剪 |

---

## 6. 快速启用优化清单

```python
class Exp(SFTExp):
    def __init__(self):
        super().__init__()
        
        # 1. 启用 Flash Attention
        self.model_cfg.use_flash_attn = True
        
        # 2. 启用梯度检查点 (省显存，增耗时)
        self.model_cfg.recompute = ["attention", "feed_forward"]
        
        # 3. 启用 VPP 提高 GPU 利用率
        self.model_cfg.parallel_cfg.virtual_pipeline_model_parallel_size = 3
        
        # 4. Optimizer 状态 offload (省显存)
        self.trainer_cfg.offload_optimizer_state = True
        
        # 5. 启用 Async TP AllReduce
        self.model_cfg.tp_cfg.async_tensor_model_parallel_allreduce = True
        
        # 6. Async Checkpoint
        self.checkpoint_cfg.async_dump = True
        
    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization
        
        set_optimization(
            default="torch_compile",
            AttentionCore="flash-attn",
            grouped_gemm="nv_grouped_gemm",
        )
```

---

## 7. 调试工具

### 7.1 内存追踪 (CMT)

```bash
export MEM_DIAGNOSE=1
```

```python
from steptronoss.utils.memory_tracker import CMT

with CMT.record("my_operation"):
    # 你的代码
    pass
```

### 7.2 性能计时器

```python
with self.timers.record("forward-backward", log_level=1):
    pp_scheduler.run(grad_accumulation_steps)
```

### 7.3 配置检查

```bash
# 查看配置树
uv run cfshow your_exp.py

# 类型检查
uv run mypy your_exp.py
```

---

## 8. 关键文件速查

| 功能 | 文件 |
|------|------|
| 并行状态管理 | `steptronoss/core/parallel_state.py` |
| 训练主流程 | `steptronoss/core/trainers/lm_trainer.py` |
| 流水线调度 | `steptronoss/core/pipeline_parallel/schedules.py` |
| MoE 实现 | `steptronoss/model/common/moe_block.py` |
| Attention 核心 | `steptronoss/model/common/attention_core.py` |
| 梯度管理 | `steptronoss/optimizer/zero1_gradient_manager.py` |
| 检查点 | `steptronoss/checkpointing/local_checkpoint.py` |
| 优化选项 | `steptronoss/utils/optimizable.py` |

---

> 文档版本: 2025-03
> 基于 StepTronOSS 代码库整理
