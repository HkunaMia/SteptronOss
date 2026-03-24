# Step 3.5 Flash 与 SteptronOss 训练框架技术分享

**演讲时长: 约50分钟**

---

## 开场 (约2分钟)

各位好，今天我为大家介绍Step 3.5 Flash模型及其背后的训练框架SteptronOss。

Step 3.5 Flash是由阶跃星辰发布的大语言模型，采用了MoE架构，总参数196B，激活参数仅11B。这意味着它拥有强大的表达能力，但推理成本却只相当于一个11B的稠密模型。

我将从六个方面进行介绍：模型架构、训练框架、以及四个核心优化——解耦并行、通信与Muon优化、算子融合与重计算、以及RLVR与预训练流程。

---

## 第一部分: 模型架构概览 (约5分钟)

Step 3.5 Flash的核心架构特点可以概括为三点：MoE架构实现高效推理、Hybrid Attention平衡长上下文与计算成本、以及创新的负载均衡策略。

### 模型规格
- 196B总参数 / 11B激活参数
- 45层Transformer结构
- 每层64个注意力头，8个KV头（GQA设计）
- 隐藏层维度4096，head维度128
- 128K上下文窗口
- 128,896词表大小

### Hybrid Attention (S3F1)

Step 3.5 Flash最独特的架构设计是混合注意力机制。我们采用了S3F1的设计——3层Sliding Window Attention配合1层Full Attention交替排列。

为什么要这样设计？Full Attention可以捕捉长距离依赖，支持128K的上下文窗口，但计算复杂度是序列长度的平方。而Sliding Window Attention只关注局部512个token，复杂度大幅降低。S3F1在保持长上下文能力的同时，显著降低了训练和推理成本。

模型还引入了Head-wise Gated Attention，这是一种对注意力头的门控机制，让模型可以自适应地选择关注哪些注意力头。

### MoE设计创新

在MoE设计上，Step 3.5 Flash使用了64个专家，每个token激活8个专家。

这里有一个重要的创新——Aux-Loss-Free负载均衡。传统的MoE训练需要辅助损失来强制负载均衡，但这会影响模型质量。我们的方案是为每个专家维护一个偏置项，根据历史负载动态调整，选择时用概率加偏置，但权重仍用原始概率。这样既保证了负载均衡，又不损害模型效果。

此外，我们采用了Sigmoid Router替代Softmax，配合归一化实现更稳定的路由。还针对特定层设置了SwiGLU激活裁剪，防止数值溢出。Router Bias保持FP32精度，确保路由稳定性。

### 多Token预测 (MTP-3)

模型还采用了MTP-3，即一次预测3个未来token，这提高了训练效率，也为推理加速提供了可能。

总结一下，Step 3.5 Flash通过MoE实现高效推理，通过Hybrid Attention平衡长上下文与计算成本，通过创新的负载均衡策略保证训练稳定性和模型质量。

---

## 第二部分: 训练框架架构 (约5分钟)

接下来我为大家介绍SteptronOss训练框架的架构设计。

### 框架定位

SteptronOss是一个轻量级的大规模语言模型训练框架。相比于TorchTitan这样的全功能框架，SteptronOss更专注于大规模训练的核心需求，同时保持模块化和可配置性。

让我们看看两者的对比：
- TorchTitan使用TOML文件进行配置，SteptronOss采用Python类配置，给了我们更大的灵活性
- SteptronOss实现了自定义的TP、PP、DP、CP、EP策略，而TorchTitan主要依赖PyTorch的DTensor
- SteptronOss原生支持SFT、RLVR、Pretrain多种工作流，而TorchTitan主要聚焦预训练

### 代码结构

SteptronOss的代码结构非常清晰：
- `core/`目录包含并行和训练基础设施，包括parallel_state管理、各种trainer实现、流水线并行调度器等
- `model/`目录包含模型架构，从Qwen稠密模型到Step 3.5 MoE模型都有实现
- `exp/`目录是实验配置的核心，使用configurize库实现声明式配置
- `data/`目录处理数据加载
- `optimizer/`目录管理优化器和梯度
- `checkpointing/`目录处理检查点保存/加载

### configurize配置系统

configurize配置系统是SteptronOss的一大特色。它允许我们用Python类来声明配置，而不是写死板的配置文件。更重要的是Ref引用机制——通过"..."可以向上引用2级配置，".."引用1级。这让复杂的嵌套配置变得简洁。

举个例子，MoE配置可以引用上层的hidden_size，而不需要重复定义。这种引用关系在配置验证时会自动解析。

### 训练流程

训练流程从Exp.train()开始，经过配置解析和验证，创建对应的Trainer，然后进入训练循环。关键步骤包括：
1. 初始化并行管理器PM
2. 构建模型
3. 构建优化器
4. 加载检查点
5. 核心train_loop

每个train_step包括：清零梯度、配置流水线调度器、执行前向反向传播、参数更新、学习率调整，以及MoE路由偏置更新。

框架还提供了丰富的优化选项：Flash Attention、梯度检查点、VPP、异步TP AllReduce、异步检查点，都可以通过简单配置启用。

---

## 第三部分: 解耦并行方案 (约10分钟)

现在进入核心优化部分。首先介绍解耦并行方案，这是SteptronOss最具创新性的设计之一。

### 为什么需要解耦并行

传统大模型训练框架通常采用"一刀切"的并行策略，比如统一的TP+PP+DP。但在处理混合架构——比如Dense Attention加Sparse MoE时，这种统一策略会导致通信效率低下。

为什么？因为Attention和MoE有截然不同的并行需求：
- Attention计算密集，适合用TP张量并行，但TP会带来频繁的AllReduce通信
- MoE的特点是稀疏激活，每个token只访问少数专家，更适合用EP专家并行

如果我们强行用同一套并行策略，必然有一方是低效的。

### 解耦并行核心思想

SteptronOss的解耦并行方案允许不同模块采用独立的并行策略：
- Attention模块使用CP上下文并行处理长序列，避免TP的频繁AllReduce
- MoE模块使用EP专家并行扩展专家数量
- Attention参数和专家参数分别在独立的并行组内同步梯度

### Step 3.5 Flash并行配置

以Step 3.5 Flash为例，配置是：
- TP=1：Attention不切分，完全依赖CP处理长序列
- PP=8：8个流水线阶段，每个节点一个Stage
- CP=8：8张卡分割序列长度，128K序列每张卡处理16K tokens
- EP=8：8张卡分割288个专家，每卡36个专家
- ETP=1：专家不切分

这里的关键是CP和EP组的动态切换。在Attention计算时，使用CP组进行Ring Attention通信；切换到MoE计算时，使用EP组进行All-to-All的token分发。这就是解耦并行的精髓。

### CP的Balanced Complementary Sharding

CP的实现采用了Balanced Complementary Sharding策略。128K序列被切成16个8K的chunk，8个CP rank每个取首尾各一个chunk。这样设计是为了负载均衡——每个rank处理的计算量是相同的。

对比Ring Attention，Balanced Complementary Sharding的优势在于通信量更少。Ring Attention需要每个rank和相邻rank多次通信，而Balanced方案只需要AllGather和Reduce-Scatter，在节点内部通过NVLink可以高效完成。

### Mesh动态切换

SteptronOss的ParallelManager提供了switch_mesh上下文管理器，可以在运行时动态切换当前活跃的并行组。这让我们可以在同一层内先以CP模式计算Attention，再切换到EP模式计算MoE，全程自动处理通信组的切换和RNG状态的保存恢复。

示例代码：
```python
with PM.switch_mesh("CP"):
    output = attention(x)
with PM.switch_mesh("EP"):
    output = moe(x)
```

### 45层分配算法

45层网络被分配到24个虚拟槽位（PP×VPP=8×3=24）：
- 前3个槽位各1层
- 后21个槽位各2层

层号交错分散在不同Node，Node内通过CP/EP并行计算。

总结一下，解耦并行通过允许Attention和MoE使用最适合各自的并行策略，并通过动态Mesh切换实现无缝协作，显著提升了混合架构模型的训练效率。

---

## 第四部分: 通信优化与Muon方案 (约10分钟)

接下来介绍通信优化和Muon优化器方案，这是论文中提到的第2和第3个核心优化。

### 通信优化

大规模分布式训练的通信开销往往是性能瓶颈，SteptronOss从多个维度进行了优化。

**梯度缓冲区设计**：传统Megatron使用参数分桶方式，每个参数有自己的梯度缓冲区，导致内存不连续，通信效率低。SteptronOss采用了分桶连续缓冲区，将梯度按数据类型分桶存储。这样通信时可以发起更大的数据传输，显著提升带宽利用率。

**DP与EDP分离**：EDP是Expert Data Parallel，专门用于MoE专家参数的数据并行。通过将普通DP和EDP分离，我们可以避免不必要的通信。Attention参数只在普通DP组内同步，专家参数只在EDP组内同步，互不干扰。

**通信与计算重叠**：SteptronOss在梯度计算的同时就启动AllReduce，通过双缓冲机制让通信和计算流水线化。这隐藏了通信延迟，让GPU计算单元始终处于忙碌状态。

**双视图零拷贝技术**：这个设计借鉴了PyTorch分布式数据并行的经验，将参数和梯度缓冲区统一寻址。更新参数时不需要显式拷贝，直接操作底层内存，避免了额外的内存搬运。

### Muon优化器与ZeRO-1 Resharding

Muon是一种正交梯度下降的优化器，相比Adam有更好的收敛性。它的核心是对梯度进行正交化处理，使用Newton-Schulz迭代算法。这个算法可以在少量迭代内将梯度矩阵正交化，计算开销可控。

但Muon与ZeRO-1的结合面临一个挑战：内存布局。ZeRO-1需要将优化器状态分片到不同DP rank，而Muon的正交化需要特定的矩阵布局。

SteptronOss的解决方案是**Rank-Major参数分配**。传统是按参数分片，每个rank持有部分参数的完整状态。而Rank-Major是按矩阵行分片，每个rank持有所有参数的部分行。这种布局天然适合Muon的正交化计算。

实现上，SteptronOss采用了**两阶段更新策略**：
1. 第一阶段，各rank本地更新自己持有的行
2. 第二阶段，通过分布式gather/scatter操作同步更新结果

这样既保持了ZeRO-1的内存效率，又满足了Muon的计算需求。

此外，SteptronOss还实现了拓扑感知缩放。针对H800的NVLink拓扑，优化了通信路由，让高带宽链路承载更多流量，进一步提升了通信效率。

---

## 第五部分: 算子融合与细粒度重计算 (约10分钟)

接下来介绍论文的第4和第5个核心优化：算子融合和细粒度重计算。

### 算子融合

现代GPU的瓶颈往往不是计算，而是内存带宽和kernel启动开销。算子融合将多个小算子合并为单个kernel，减少内存搬运和kernel调度开销。

SteptronOss使用Triton编写了大量的融合算子，通过`@optimizable`装饰器支持多后端。这是框架的一大特色——同一功能可以有多种实现：PyTorch原生、Triton、CUDA CUTLASS，用户可以根据硬件选择最优方案。

**Grouped GEMM**：MoE有多个专家，传统方法是逐个专家做矩阵乘法，这会产生大量小kernel。Grouped GEMM将多个专家的矩阵乘法融合为单个kernel，通过批次处理提高GPU利用率。

框架提供了三种Grouped GEMM后端：
- NVIDIA的CUTLASS实现：性能最优，适合H100 Tensor Core
- Triton实现：灵活性最好，适合快速开发迭代
- PyTorch实现：兼容性最好

**MoE路由优化**：有一整套Triton优化：
- `moe_scatter`：将token按专家顺序排列
- `moe_weighted_gather`：收集专家输出并按权重加权
- `histogram`：统计专家负载
- `index_compute`：计算scatter索引

这些原本需要多个PyTorch算子完成的操作，现在都被融合为单个Triton kernel。

**Routed Grouped FFN端到端融合**：从token路由到专家计算再到结果收集，整个MoE前向传播被融合为单个函数。这消除了中间结果的内存搬运，大幅提升了MoE计算效率。

**DeepEP融合通信**：在通信层面，DeepEP库提供了融合的Dispatch和Combine操作，将All-to-All通信与kernel执行融合，进一步降低了延迟。

### 细粒度重计算

细粒度重计算也就是梯度检查点。传统方法是对整个层做检查点，粒度太粗。SteptronOss支持submodule级别的细粒度控制，可以单独选择是否重计算attention、feed_forward、attn_norm、ffn_norm。

**SiLU激活融合**：SwiGLU激活函数有两个矩阵投影w1和w2，中间经过SiLU激活。标准实现需要存储SiLU的输出用于反向传播，这个激活值很大。优化方案是只存储w1的输出，反向传播时重新计算SiLU。这节省了约30%的FFN激活内存。

**分布式激活存储**：在TP并行时，激活值被分片存储在不同rank，每rank只保存本地chunk。反向传播时通过AllGather恢复完整激活。

**MoE特殊处理**：Router不能重计算，因为它需要传播梯度来学习路由策略；但Expert计算可以重计算。这个区分很重要。

**逐层配置**：SteptronOss支持逐层配置不同策略。比如早期层使用完整重计算，中期层选择性重计算，晚期层最小重计算。这种灵活性让用户可以根据内存和速度的trade-off精细调优。

示例配置：
```python
def pp_vp_allocation(self, layer_id):
    if layer_id < 10:
        return {"recompute": ["attention", "feed_forward", "attn_norm", "ffn_norm"]}
    elif layer_id < 40:
        return {"recompute": ["attention", "feed_forward"]}
    else:
        return {"recompute": []}
```

---

## 第六部分: RLVR与预训练流程 (约6分钟)

最后介绍SteptronOss在RLVR和预训练方面的设计。

### RLVR训练架构

RLVR，也就是带可验证奖励的强化学习，是Step 3.5 Flash后训练的核心方法。它涉及三个模型：
- **Actor**：策略网络，负责生成回答，可训练
- **Reference**：参考模型，提供KL正则化的基准，权重冻结
- **Critic**：价值网络，估计状态价值，可训练

Critic和Reference都从Actor初始化。

### 训练-推理分离

SteptronOss采用了训练与推理分离的架构。训练使用PyTorch加Megatron框架进行梯度更新，推理则使用vLLM进行高吞吐的token生成。两者通过权重热部署机制同步——训练好的权重通过safetensors格式导出，vLLM在不重启服务的情况下热加载新权重。

### Flow Control异步流控

Flow Control是RLVR训练的核心调度机制，支持三种策略：
- **On-policy**：每次训练前同步最新权重再生成，数据最新但速度较慢
- **One-step-off**：使用当前权重训练，同时用新权重异步生成下一批，是一种折中
- **Fully-async**：持续异步生成，限制最大陈旧步数，吞吐最高但数据可能滞后

### 模型卸载与序列打包

内存方面，PPO训练同时需要Actor、Critic、Reference三个模型，显存压力很大。SteptronOss提供了PackedModel三级卸载：
1. 参数 (params)
2. 梯度缓冲区 (grad_buffer)
3. 优化器状态 (optimizer_state)

这些都可以独立卸载到CPU。在训练流程中精心编排加载和卸载时机，让峰值显存只相当于单个模型训练态。

对于变长序列，框架实现了PackedPPOSamples打包机制。多条轨迹被拼接为一个长序列，通过cu_seqlens记录边界。这样既减少了padding浪费，又满足了TP×CP×2的对齐约束。

### 预训练优化

预训练方面，SteptronOss提供了：
- **DataRecipe**：支持多领域数据混合，可以为不同领域设置不同的采样轮数
- **CompiledDataset**：将原始数据预编译为优化的二进制格式，加速训练启动
- **MixedPackedDataloader**：序列打包，将padding从30-40%降低到5%以下
- **异步检查点**：后台保存不阻塞训练
- **分布式检查点**：支持DP变化时的自动reshape

---

## 总结

Step 3.5 Flash通过以下技术创新实现了高效的大规模语言模型训练：

1. **MoE架构**：196B参数/11B激活，Aux-Loss-Free负载均衡
2. **Hybrid Attention**：S3F1设计平衡长上下文与计算成本
3. **解耦并行**：Attention与MoE使用最适合各自的并行策略
4. **通信优化**：分桶缓冲区、DP/EDP分离、计算通信重叠
5. **Muon优化器**：正交梯度下降+ZeRO-1整合
6. **算子融合**：Triton融合kernel提升MoE计算效率
7. **细粒度重计算**：模块化控制+SiLU融合优化
8. **RLVR支持**：训练推理分离+Flow Control+模型卸载

SteptronOss框架为这些技术创新提供了完整的工程实现，是一个轻量级但功能完备的大规模LLM训练框架。

---

## 附录: 待核实问题

1. **专家数量**：代码配置`moe_num_experts=288`与论文所述64个专家不一致，需核实
2. **通信优化代码**：用户反馈代码中未找到实现，需核实具体位置
3. **Muon优化器**：需确认`steptronoss/optimizer/muon.py`是否存在

---

*文档版本: 2026-03-23*
*基于: Step 3.5 Flash论文 + SteptronOss代码库*
