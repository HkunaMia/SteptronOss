# StepTronOSS 框架内部分享

> 面向技术团队的 StepTronOSS 框架介绍，涵盖核心价值、技术亮点与最佳实践。

---

## 1. 框架定位

### 1.1 一句话定义

**StepTronOSS** 是 StepFun 开源的轻量级大模型训练框架，支持从预训练到 RLVR 的完整工作流，以配置驱动和模块化设计为核心特色。

### 1.2 核心价值

| 维度 | 价值 |
|------|------|
| **轻量** | 纯 PyTorch 实现，最小依赖，易于理解和修改 |
| **模块化** | 配置驱动实验，组件可插拔，快速迭代 |
| **生产级** | 支持 TP/PP/DP/CP/EP 5D 并行，开箱即用 |
| **完整流** | 预训练、SFT、RLVR 统一框架，无缝切换 |

---

## 2. 技术架构亮点

### 2.1 配置系统（configurize）

```python
# 声明式配置，类级定义 + 实例级赋值
class Exp(SFTExp):
    model_cfg = Qwen3Config        # 类级声明
    data_cfg = SFTDataConfig
    
    def __init__(self):
        self.model_cfg.hidden_size = 4096    # 实例级配置
        self.trainer_cfg.global_batch_size = 32
```

**优势**:
- 配置可序列化、可验证 (`cfshow` 工具)
- 支持跨配置引用 (`Ref("...tp_cfg")`)
- 热更新：命令行覆盖配置参数

### 2.2 5D 并行架构

```
┌─────────────────────────────────────────────────────────────┐
│                    5D 并行支持                               │
├─────────────────────────────────────────────────────────────┤
│  TP (Tensor Parallel)      │ Attention 层切分               │
│  PP (Pipeline Parallel)    │ 层间流水线                    │
│  DP (Data Parallel)        │ ZeRO-1 数据并行               │
│  CP (Context Parallel)     │ 长序列切分 (128K+)            │
│  EP (Expert Parallel)      │ MoE 专家并行                  │
└─────────────────────────────────────────────────────────────┘
```

**特色**:
- EP + TP 混合：支持超大专家模型
- VPP (Virtual Pipeline)：提升 GPU 利用率
- DeepEP 集成：高效 MoE 通信

### 2.3 MoE 优化全家桶

| 优化项 | 说明 | 收益 |
|--------|------|------|
| **Aux-Loss-Free** | 无辅助损失负载均衡 | 模型质量 + 训练稳定 |
| **Grouped GEMM** | 专家计算融合 | 30-50% 计算加速 |
| **DeepEP** | 融合通信内核 | 通信延迟降低 |
| **Sigmoid Router** | 替代 Softmax | 路由稳定性 |

### 2.4 数据流水线

```
原始数据 → DataRecipe → 编译 → Packed Dataloader → 训练
                ↓            ↓              ↓
           多领域混合    元数据预计算    序列打包
           加权采样      启动加速        减少 Padding
```

**关键特性**:
- **DataRecipe**: 多领域数据混合，灵活配置采样比例
- **CompiledDataset**: 预编译加速启动
- **Packed Dataloader**: 序列打包，Padding 率 < 5%

---

## 3. 快速上手

### 3.1 环境准备

```bash
# 安装依赖
uv sync
apt install -y redis-server

# 可选优化
uv pip install flash-attn --no-build-isolation
```

### 3.2 第一个实验

```python
# playground/sft/my_first_exp.py
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.pretrain.qwen3.qwen3_1p7b import Qwen3_1p7BConfig

class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig
    
    def __init__(self):
        super().__init__()
        self.trainer_cfg.global_batch_size = 32
        self.scheduler_cfg.lr = 1e-5

if __name__ == "__main__":
    Exp().train()
```

```bash
# 验证配置
uv run cfshow my_first_exp.py

# 启动训练
uv run torchrun my_first_exp.py
```

### 3.3 常用命令

```bash
# 配置检查
uv run cfshow exp.py -k model_cfg

# 多任务 RLVR
uv run tools/mp_run.py exp.py

# 生成多节点脚本
uv run tools/build_scripts.py exp.py /mnt/entrypoints/
```

---

## 4. 与开源框架对比

### 4.1 对比概览

| 特性 | StepTronOSS | Megatron-LM | DeepSpeed | TorchTitan |
|------|-------------|-------------|-----------|------------|
| **定位** | 轻量全栈 | 大规模预训练 | 训练加速 | PyTorch 原生 |
| **代码量** | 中等 (~2w 行) | 大 (~5w+ 行) | 大 (~3w+ 行) | 中等 (~1w 行) |
| **配置方式** | Python 类 | YAML/JSON | JSON | TOML |
| **并行** | 5D (TP/PP/DP/CP/EP) | 4D (TP/PP/DP/EP) | ZeRO + PP | 4D (TP/PP/DP/CP) |
| **MoE** | 原生支持 | 支持 | 支持 | 开发中 |
| **RLVR** | 原生支持 | 需扩展 | 需扩展 | 不支持 |
| **Serving** | vLLM 集成 | 不支持 | 不支持 | 不支持 |

### 4.2 详细对比

#### vs Megatron-LM

| 方面 | StepTronOSS | Megatron-LM |
|------|-------------|-------------|
| **易用性** | ✅ 配置驱动，快速上手 | 配置复杂，学习曲线陡 |
| **灵活性** | ✅ 模块化，易于修改 | 高度耦合，侵入式修改 |
| **性能** | 接近 | 优化充分，略优 |
| **MoE** | ✅ Aux-Loss-Free, DeepEP | 传统 Aux Loss |
| **社区** | 发展中 | NVIDIA 背书，生态大 |

**适用场景**:
- StepTronOSS: 快速实验迭代，定制化需求
- Megatron-LM: 超大规模集群，追求极致性能

#### vs DeepSpeed

| 方面 | StepTronOSS | DeepSpeed |
|------|-------------|-----------|
| **ZeRO** | ZeRO-1 | ZeRO-1/2/3/Infinity |
| **并行** | 完整 5D | 侧重 DP + PP |
| **易用性** | ✅ 原生集成 | 需理解 ZeRO 配置 |
| **MoE** | ✅ 完整支持 | 需额外配置 |
| **压缩** | 基础 | ✅ 量化、剪枝 |

**适用场景**:
- StepTronOSS: 标准大规模训练，MoE 模型
- DeepSpeed: 显存极度受限，需要 ZeRO-Offload

#### vs TorchTitan

| 方面 | StepTronOSS | TorchTitan |
|------|-------------|------------|
| **配置** | Python 类 | TOML |
| **工作流** | ✅ 预训练/SFT/RLVR | 主要是预训练 |
| **数据** | DataRecipe + Packing | HuggingFace |
| **MoE** | ✅ 完整支持 | 开发中 |
| **Serving** | ✅ vLLM 集成 | 不支持 |

**适用场景**:
- StepTronOSS: 完整工作流，MoE，生产部署
- TorchTitan: 纯 PyTorch 研究，快速原型

---

## 5. 性能数据

### 5.1 预训练性能 (参考)

| 模型 | 配置 | 吞吐量 | MFU |
|------|------|--------|-----|
| Qwen3-8B | TP=2, PP=4 | ~4000 tokens/s | ~45% |
| Step3.5 | EP=8, TP=1 | ~2500 tokens/s | ~40% |

### 5.2 优化效果

| 优化 | 收益 |
|------|------|
| Flash Attention | 2-4x Attention 加速 |
| Grouped GEMM | 30-50% MoE 计算加速 |
| 序列打包 | Padding 从 30% → 5% |
| 异步检查点 | 保存时间从 60s → 5s |

---

## 6. 最佳实践

### 6.1 配置选择指南

```
模型大小 < 7B:
  TP=1, PP=1, DP=8
  不需要 gradient checkpointing

7B < 模型大小 < 30B:
  TP=2, PP=4, DP=8
  recompute=["attention", "feed_forward"]

30B < 模型大小 < 100B:
  TP=4, PP=8, DP=8
  recompute=True
  offload_optimizer_state=True

MoE 模型:
  EP=8, TP=1 (小专家)
  EP=4, TP=2 (大专家)
  enable_auxiliary_loss_free_load_balance=True
```

### 6.2 调试技巧

```bash
# 内存诊断
export MEM_DIAGNOSE=1

# 查看配置树
uv run cfshow exp.py -k model_cfg

# 检查并行配置
python -c "from steptronoss.core.parallel_state import PM; PM.initialize()"
```

---

## 7. 路线图与待办

### 7.1 短期 (3个月)

- [ ] Eval 框架集成
- [ ] RLVR 完整实现
- [ ] Triton Kernel 优化
- [ ] 更多预训练模型配置

### 7.2 中期 (6个月)

- [ ] 自动并行策略搜索
- [ ] 模型压缩 (量化、剪枝)
- [ ] 多模态支持
- [ ] 更完善的文档和教程

---

## 8. 总结

### 8.1 什么时候选择 StepTronOSS

✅ **适合**:
- 需要快速迭代实验
- MoE 模型训练
- 完整工作流 (预训练 → SFT → RLVR)
- 需要定制化修改
- 生产部署 (vLLM 集成)

❌ **不适合**:
- 超大规模集群 (>1024 GPUs) 追求极致性能
- 需要 ZeRO-Infinity 等极端显存优化
- 纯研究，不需要完整工作流

### 8.2 核心优势回顾

1. **轻量**: 纯 PyTorch，易于理解和修改
2. **完整**: 覆盖训练全流程
3. **高效**: MoE 优化、序列打包、异步检查点
4. **灵活**: 配置驱动，模块化设计

---

## 9. 参考资源

- 框架指南: `docs/STEPTRONOSS_GUIDE.md`
- MoE 优化: `docs/MOE_OPTIMIZATIONS.md`
- 并行策略: `docs/EP_TP_HYBRID_PARALLEL.md`
- 预训练优化: `docs/PRETRAIN_OPTIMIZATIONS.md`

---

> 分享日期: 2025-03
> 基于 StepTronOSS 最新代码整理
