## StepTronOSS AGENTS

This file stores repo-specific priors for AI coding agents. It contains essential information about project structure, development conventions, and workflows.

## 1. Project Overview

**StepTronOSS** is a lightweight training framework for large-scale language models, focusing on:
- Modular, config-driven experiments with dynamic validation
- Reproducible training workflows (SFT, RLVR, Pretrain)
- Multi-task orchestration with flexible launch tooling
- Extensible data/optimizer/model stacks for rapid research iteration

The framework can run with only PyTorch as a dependency, while also supporting operator-level optimizations (flash-attn, Triton kernels, grouped_gemm).

## 2. Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: uv (modern Python package manager)
- **Build System**: hatchling
- **Deep Learning**: PyTorch 2.9.0, transformers, triton 3.5.0
- **Distributed Training**: Custom parallel state management with TP/PP/DP/CP/EP support
- **Serving**: vLLM for inference deployment
- **Data**: Custom dataloader with packing and compilation support
- **Logging**: loguru, tensorboard, wandb
- **Communication**: Redis for distributed rendezvous

### Key Dependencies
```toml
torch==2.9.0
triton==3.5.0
transformers<5.0
vllm>=0.11
safetensors
redis
megfile
```

## 3. Code Organization

```
steptronoss/                    # Core package
├── core/                       # Parallelism and training infrastructure
│   ├── parallel_state.py       # Global PM (ParallelManager) for TP/PP/DP/CP/EP
│   ├── trainers/               # Trainer implementations
│   ├── pipeline_parallel/      # PP/VPP schedulers
│   ├── tensor_parallel/        # TP utilities
│   └── context_parallel/       # CP implementations
├── model/                      # Model architectures and implementations
│   ├── qwen_dense.py           # Qwen dense model
│   ├── step3p5.py              # Step3.5 model
│   ├── decoder_model.py        # Base decoder
│   ├── common/                 # Shared components (attention, MoE, embedding)
│   ├── ep_dispatcher/          # Expert parallel dispatchers (DeepEP)
│   ├── optimizations/          # Triton-optimized kernels
│   └── utils/                  # Model utilities
├── exp/                        # Experiment configurations
│   ├── base_exp.py             # Core configs (BaseExp, TrainerConfig, etc.)
│   ├── ntp.py                  # Next Token Prediction (PretrainExp)
│   ├── sft.py                  # SFT experiments
│   ├── rl.py                   # RLVR/PPO experiments
│   ├── optimizer.py            # Optimizer configs (Adam, Muon)
│   ├── lr_schedulers.py        # Learning rate schedulers
│   ├── checkpointing.py        # Checkpoint configs
│   ├── inference.py            # Inference configs (VLLMDeployConfig)
│   └── resources.py            # Resource/TaskSpec configs
├── data/                       # Data loading and processing
│   ├── recipe.py               # DataRecipe for compilation
│   ├── dataloader/
│   ├── datasets/
│   └── packing/                # Sequence packing
├── optimizer/                  # Gradient managers and optimizers
│   ├── gradient_manager.py
│   ├── zero1_gradient_manager.py
│   └── muon.py                 # Muon optimizer
├── checkpointing/              # Checkpoint save/load and reshape
│   ├── local_checkpoint.py
│   ├── reshape_ops.py          # Ops for checkpoint reshaping
│   └── hf_checkpoint.py
├── generation/                 # Generation infrastructure
│   ├── async_generation.py
│   └── vllm/                   # vLLM router/controller
├── tokenizer/                  # Tokenizer implementations
├── text_processing/            # Text processing utilities
└── utils/                      # Utilities
    ├── arguments.py            # CLI argument parsing
    ├── comm_utils.py           # Redis rendezvous, queues
    ├── dist_utils.py           # Distributed utilities
    ├── metrics.py              # Metrics system
    ├── optimizable.py          # @optimizable decorator
    ├── logger.py               # Rank-aware logging
    └── memory_tracker.py       # Memory tracking (CMT)

playground/                     # Experiment configurations
├── pretrain/                   # Pretraining experiments
│   └── qwen3/
├── sft/                        # SFT experiments
│   ├── qwen3/
│   └── step3/
├── rlvr/                       # RLVR experiments
└── tools/                      # Data compilation tools

tests/                          # Test suite
├── conftest.py                 # pytest configuration (node2 skip logic)
├── test_*.py                   # Unit/integration tests
└── fixtures/

tools/                          # Runtime tools
├── mp_run.py                   # Multi-process task runner
├── build_scripts.py            # Generate per-replica launch scripts
├── smartrun                    # Smart runner wrapper
└── example_submitter.py        # Template for custom submitters

benchmarks/                     # Performance benchmarks
├── benchmark_*.py              # Component benchmarks
└── run_all.sh

docs/                           # Documentation
├── LAUNCH_EXPERIMENTS.md       # Launch guide
├── SFT_DATA_PREPARATION.md     # Data preparation
├── TRITON_ACCELERATION_WORKFLOW.md  # Triton workflow
└── MODULES.md                  # API documentation
```

## 4. Build and Setup Commands

### Initial Setup
```bash
# Install dependencies and pre-commit hooks
uv sync
apt install -y redis-server
uv run pre-commit install
```

### Optional Optimizations
```bash
# Flash Attention
uv pip install flash-attn --no-build-isolation

# Grouped GEMM (requires CUDA 12.9)
# Ensure CUTLASS headers are linked properly first
CUDA_HOME=/data/cuda/cuda-12.9/cuda \
CUDACXX=/data/cuda/cuda-12.9/cuda/bin/nvcc \
pip install -e third_party/grouped_gemm --no-build-isolation

# DeepEP (if using expert parallelism)
CUDA_HOME=/data/cuda/cuda-12.9/cuda \
CUDACXX=$CUDA_HOME/bin/nvcc \
pip install -e /data/DeepEP --no-build-isolation
```

### Code Quality
```bash
make check          # Run pre-commit hooks (ruff, formatters)
make test           # Run pytest with coverage
make docs           # Build and serve documentation
make docs-test      # Test documentation build
```

### Environment Variables
```bash
# Required for multi-node runs
export STEPTRON_MEET_DIR=/path/to/shared    # Redis rendezvous directory

# Optional
export CANNOT_BE_REDIS_SERVER=1             # Prevent this rank from starting Redis
export MEM_DIAGNOSE=1                       # Enable memory tracking (CMT)
```

## 5. Experiment Development

### Config System (configurize)

All experiments use the `configurize` library for declarative configs:

```python
from configurize import Config, Ref

class MyExpConfig(Config):
    """Config class with type annotations and docstrings."""
    param_a: int                    # Required field
    param_b: float = 1.0           # Default value
    nested_cfg: SubConfig = SubConfig  # Sub-config
    
    # Reference to parent/other config values
    derived_value: int = Ref("..parent_value")
    
    def build(self):
        """Build the runtime object."""
        return MyExp(cfg=self)
    
    def sanity_check(self):
        """Validate config values."""
        super().sanity_check()
        assert self.param_b > 0
```

### Experiment Structure

```python
class MyExp(BaseExp):
    """Experiment docstring describing purpose."""
    # Config declarations (class level)
    trainer_cfg: NTPTrainerConfig = NTPTrainerConfig
    model_cfg: MyModelConfig = MyModelConfig
    data_cfg: MyDataConfig = MyDataConfig
    
    def __init__(self):
        super().__init__()
        # Instance-level config customization
        self.trainer_cfg.global_batch_size = 32
        self.model_cfg.hidden_size = 4096

if __name__ == "__main__":
    Exp().train()
```

### Running Experiments

```bash
# Single-task single-node
uv run torchrun playground/sft/your_exp.py

# Multi-task (e.g., RL with generator + trainer + router)
export STEPTRON_MEET_DIR=/path/to/shared
uv run tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py

# Config inspection and validation
uv run cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py
uv run cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py -k actor_model_cfg

# Override config from CLI
uv run tools/mp_run.py your_exp.py trainer_cfg.lr=1e-4

# Generate launch scripts for multi-node
uv run tools/build_scripts.py your_exp.py /mnt/entrypoints/
```

### Pre-made Experiment Classes

- `PretrainExp` (in `exp/ntp.py`): Next Token Prediction pretraining
- `SFTExp` (in `exp/sft.py`): Supervised Fine-Tuning
- `PPOLikeExp` (in `exp/rl.py`): RLVR with PPO

## 6. Code Style Guidelines

### Python Style
- **Line length**: 120 characters
- **Formatter**: ruff (with black-compatible settings)
- **Import sorting**: isort (profile=black)
- **Type checking**: mypy (configured in pyproject.toml)

### Config Style
- Include triple-quoted docstrings after field definitions
- Use `Ref("..path")` for cross-node linkage (reference exact parameter needed)
- Implement `build()` and `sanity_check()` methods
- Use `writable_property` for computed properties

### Naming Conventions
- Config classes: `*Config` suffix (e.g., `ModelConfig`)
- Experiment classes: `Exp` suffix or descriptive names
- Private/internal: leading underscore

### Documentation
- Docstrings for all public classes and methods
- Type annotations required
- Comments explain "why", not "what"

## 7. Testing Instructions

### Test Markers
```python
@pytest.mark.cpu           # CPU-only tests
@pytest.mark.gpu           # GPU-only tests  
@pytest.mark.node2         # Requires torchrun --nproc-per-node=2
```

### Running Tests
```bash
# All tests
uv run python -m pytest tests/

# Specific markers
uv run python -m pytest tests/ -m cpu
uv run python -m pytest tests/ -m gpu

# Multi-process tests (requires torchrun)
torchrun --nproc-per-node=2 -m pytest -m node2 tests/test_muon_optimizer_node2.py

# With coverage
uv run python -m pytest tests/ --cov --cov-config=pyproject.toml
```

### Test Organization
- `tests/test_*.py`: Unit/integration tests
- `tests/conftest.py`: Shared pytest fixtures and node2 skip logic
- `tests/fixtures/`: Test fixtures and mock data

### GPU Test Notes
- GPU tests should check for GPU availability
- `node2` tests require proper distributed environment
- Tests with `@pytest.mark.node2` should also use `pytest.mark.xdist_group("torchrun")`

## 8. Triton Optimization Workflow

When adding Triton kernels, follow this workflow:

1. **Trace First**: Capture forward/backward traces on real experiments
2. **Choose Boundary**: Single-op replacement → Small fusion → Semantic fused path
3. **Location**: 
   - Semantic API: `steptronoss/model/utils/*`
   - Triton impl: `steptronoss/model/optimizations/<component>/triton.py`
4. **Register**: Use `@optimizable(alternatives={"triton": triton_impl})`
5. **Test**: Add CPU reference and GPU forward/backward tests
6. **Benchmark**: Add benchmark in `benchmarks/benchmark_*.py`
7. **Validate**: Run real experiment (4-5 iterations), inspect traces

See `docs/TRITON_ACCELERATION_WORKFLOW.md` for details.

## 9. Parallelism and Checkpointing

### Fine-grained Selective Checkpointing

The framework supports fine-grained activation recomputation with per-layer, submodule-level toggles:

```python
# In DecoderLLMConfig
recompute: list[str] | bool = []
# Options: "attention", "attn_norm", "feed_forward", "ffn_norm"
# Set True to enable all, False/[] to disable all

# In AttentionConfig
recompute_qknorm_rope: bool  # Recompute QK-Norm and RoPE

# In FeedForwardConfig  
swiglu_recompute_silu_out_proj: bool  # Fuse SiLU with output projection
```

Key implementation files:
- `steptronoss/core/tensor_parallel/random.py`: Core `CheckpointFunction`
- `steptronoss/model/decoder_model.py`: `TransformerBlock` conditional checkpointing
- `steptronoss/model/common/feed_forward.py`: SiLU fusion optimization
- `steptronoss/model/common/moe_block.py`: MoE-aware checkpointing (router excluded)

See `docs/FINE_GRAINED_SELECTIVE_CHECKPOINTING.md` for detailed design.

### Parallel State (PM)

The global `PM` (ParallelManager) manages all parallel groups:

```python
from steptronoss.core.parallel_state import PM

PM.initialize()
PM.set_mesh(parallel_cfg)
# or: with PM.use_mesh(parallel_cfg): ...

# Common helpers
PM.size_of("TP")           # Get TP size
PM.rank_in("DP")           # Get rank in DP group
PM.group_of("PP")          # Get process group
PM.ranks_of("EP")          # Get all ranks in EP group
PM.i_am("PP", 0)           # Check if rank 0 in PP
```

### Parallel Dimensions
- **TP**: Tensor Parallel
- **PP**: Pipeline Parallel
- **DP**: Data Parallel
- **CP**: Context Parallel
- **EP**: Expert Parallel (MoE)
- **ETP**: Expert Tensor Parallel
- **VPP**: Virtual Pipeline Parallel

### Sizing Constraints
- `WORLD_SIZE` must be divisible by attention MP size = `PP * TP * CP`
- `WORLD_SIZE` must be divisible by MoE MP size = `PP * ETP * EP`
- For TP=8 and EP=8 on 8 GPUs, set `expert_tensor_parallel_size=1`

### Checkpoint Reshape

`steptronoss/checkpointing/reshape_ops.py` provides reshape primitives:
- `VocabPad`, `ColumnParallel`/`RowParallel`
- `KeepThisTP`/`KeepThisEP`
- `GQAMergeQKV`, `FFNMergeGateUp`
- `UnbindMoE`, `Rename`, `Inverse`

## 10. Data System

### SFT Data Format

JSON array format (StepChatJsonDataset):
```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "loss_mask": 1}
  ],
  "images": null
}
```

### DataRecipe Workflow

1. Create `CompliableDatasetsConfig` with domains and sources
2. Compile with `playground/tools/compile_recipe.py` for faster loading
3. Use `CompiledDatasetsConfig` in experiment
4. Configure `SFTDataConfig` for training

See `docs/SFT_DATA_PREPARATION_EN.md` for details.

## 11. Common Patterns and Pitfalls

### Config Patterns
```python
# Good: Reference specific parameter
use_qk_norm: bool = Ref("..model_cfg.use_qk_norm")

# Bad: Reference whole config object
model_cfg: ModelConfig = Ref("..model_cfg")

# Good: Writable computed property
@writable_property
def pp_comm_shape(self) -> tuple:
    return (self.seq_length, self.micro_batch_size, self.hidden_size)
```

### Model Development
- Tie embeddings carefully; use `mtp_initialize()` for MTP sync
- Expert params are reduced over EDP, not dense DP
- Check gradient manager path for TP/EP scaling before suspecting extra EP factor

### Debugging
- `debug(56)` in `steptronoss/core/trainers/lm_trainer.py` hangs for rank 56
- TorchDynamo graph breaks: avoid `Tensor.item()` in optimizable helpers
- Use tensor-safe checks: `masked amax + torch._assert`

### Memory Tracking
```python
from steptronoss.utils.memory_tracker import CMT

# Only records when MEM_DIAGNOSE=1
with CMT.record("my_operation"):
    # code
```

## 12. CI/CD

GitHub Actions workflows:
- **quality**: pre-commit hooks (ruff, formatters)
- **tests-and-type-check**: pytest across Python 3.9-3.13, mypy
- **check-docs**: mkdocs build verification

## 13. Useful Commands Reference

```bash
# Config inspection
uv run cfshow <exp.py>                    # Show full config tree
uv run cfshow <exp.py> -k <key>           # Show specific subtree

# Development
uv run mypy <file.py>                     # Type check
uv run pre-commit run -a                  # Run all hooks

# Multi-node script generation
uv run tools/build_scripts.py <exp.py> <out_dir>

# Redis server for distributed
redis-server --port <PORT>
```

## 14. Self-Improvement Loop

Do an improve pass for:
- New environment or task type
- New project area exploration
- High risk / high cost work
- Collaboration / handoff work

Improve pass process:
1. Identify friction
2. Extract reusable priors
3. Write them down in:
   - `AGENTS.md` for repo-wide priors
   - `docs/` for process/runbook details

---

<<<<<<< HEAD
- Core package: `steptronoss/`
  - core, model, data, exp, optimizer, generation, tokenizer, utils, checkpointing
- Experiments: `playground/`
- Tests: `tests/`
- Docs: `docs/`

### Utilities overview

- `steptronoss/utils/arguments.py`: config overrides from CLI
- `steptronoss/utils/comm_utils.py`: Redis rendezvous, queues, `LocalFuture` / `RemoteFuture`
- `steptronoss/utils/dist_utils.py`: broadcast / all-to-all helpers, packing helpers, balancing helpers
- `steptronoss/utils/general.py`: numeric helpers, list split/balance, RNG fork, retry, recursion helpers, git hash
- `steptronoss/utils/logger.py`: rank-aware logging and `StepWriter`
- `steptronoss/utils/metrics.py`: metrics system (`Metric`, `Avg`, `Percentage`, `Histogram`, `Text`, `GradNorm`, `GlobalMetrics`)
- `steptronoss/utils/optimizable.py`: `@optimizable(...)` and `set_optimization(...)`
- `steptronoss/utils/utils.py`: model unwrap, param norms, memory report, layer map, IO helpers, generic load
- `steptronoss/utils/weight_loader.py`: HF safetensors mapping / merge

## 3. Code Style

### Config style

- Config class fields should include a short triple-quoted docstring immediately after the attribute definition.
- Follow the `configurize` pattern:
  - class attrs declare sub-config types
  - instance `__init__` sets concrete values
  - use `Ref("..path")` for cross-node linkage
  - configs expose `build()` / `build_*`, `sanity_check()`, `to_dict()`
- Only `Ref(...)` the exact parameter needed, not whole config objects.

### Experiment style

- SFT experiments under `playground/sft/qwen3/*_sft_step3_data.py` typically follow:
  - `class Exp(BaseExp)`
  - `model_cfg` / `data_cfg` declared as class attrs
  - trainer / checkpoint / model fields adjusted in `__init__`
  - entrypoint is `if __name__ == "__main__": Exp().train()`

## 4. Setup Priors

- After `uv sync`, also install `redis-server`:
  - `apt install -y redis-server`

### DeepEP build

- Set:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda`
  - `CUDACXX=$CUDA_HOME/bin/nvcc`
- Install:
  - `pip install -e /data/DeepEP --no-build-isolation`

### nv-grouped-gemm build

- Do not rely on random prebuilt wheels; ABI mismatch is common.
- Build with CUDA 12.9:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda CUDACXX=/data/cuda/cuda-12.9/cuda/bin/nvcc <python> -m pip install -e <grouped_gemm_source> --no-build-isolation`
- Runtime constraints:
  - `batch_sizes` must be CPU-visible / `torch.int64`
  - inputs must be bf16 for `nv_grouped_gemm`

## 3. Code Style

### Config style

- Config class fields should include a short triple-quoted docstring immediately after the attribute definition.
- Follow the `configurize` pattern:
  - class attrs declare sub-config types
  - instance `__init__` sets concrete values
  - use `Ref("..path")` for cross-node linkage
  - configs expose `build()` / `build_*`, `sanity_check()`, `to_dict()`
- Only `Ref(...)` the exact parameter needed, not whole config objects.

### Experiment style

- SFT experiments under `playground/sft/qwen3/*_sft_step3_data.py` typically follow:
  - `class Exp(BaseExp)`
  - `model_cfg` / `data_cfg` declared as class attrs
  - trainer / checkpoint / model fields adjusted in `__init__`
  - entrypoint is `if __name__ == "__main__": Exp().train()`
- `playground/sft/qwen3/qwen3_sft_base.py` already provides `OneNodeResourceConfig` with `replica=1` and `gpu=8`, so derived SFT experiments default to single-node 8-GPU `torchrun` unless they override `resource_cfg`.
- Under `playground/data/sft`, keep raw source recipes and dataset configs distinct in naming:
  - `*_recipe*.py` for `DataRecipe` / source file lists only
  - `*_data_config*.py` for `CompliableDatasetsConfig`, `CompiledDatasetsConfig`, and `SFTDataConfig`
  - when adding tokenizer variants for the same source recipe, share a common base config and use thin tokenizer-specific subclasses instead of duplicating whole modules
  - for large unified SFT rebuilds, do not mutate source json/jsonl in place; materialize derived json under `/oss/...`, preserve `DataSourceFile.subsample_rate` semantics with sample seed `1234`, and do global shuffle with an external bucketized sort instead of loading everything into memory

## 4. Setup Priors

- After `uv sync`, also install `redis-server`:
  - `apt install -y redis-server`

### DeepEP build

- Set:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda`
  - `CUDACXX=$CUDA_HOME/bin/nvcc`
- Install:
  - `pip install -e /data/DeepEP --no-build-isolation`

### nv-grouped-gemm build

- Do not rely on random prebuilt wheels; ABI mismatch is common.
- Build with CUDA 12.9:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda CUDACXX=/data/cuda/cuda-12.9/cuda/bin/nvcc <python> -m pip install -e <grouped_gemm_source> --no-build-isolation`
- Runtime constraints:
  - `batch_sizes` must be CPU-visible / `torch.int64`
  - inputs must be bf16 for `nv_grouped_gemm`

## 5. Experiments and Configs

### Config/module map

- `steptronoss/exp` provides abstract `*Config` interfaces (`build_*`, `get_trainer_cls`)
- Concrete configs live mainly in `steptronoss/exp/base_exp.py`
- Ready-made experiment families:
  - `PretrainExp` / `NTPTrainerConfig` in `ntp.py`
  - `SFTExp` / `SFTDataConfig` in `sft.py`
  - inference configs in `inference.py`
- Common training configs:
  - `AdamConfig`
  - constant / linear / cosine schedulers
  - checkpoint config (`SaveOptions`, `LoadOptions`, `CheckpointConfig`)

### Experiment workflow

- After creating or editing an experiment:
  - run `cfshow <exp.py>` to inspect the config tree
  - make sure `sanity_check()` passes
  - run `mypy <exp.py>`
- If experiment B is derived from experiment A, use `cfshow` diff to verify changes.

### Pretrain config notes

- Pretrain configs live under `playground/pretrain/`
- `playground/pretrain/step3p5/step3p5_flash.py` is the main recent Qwen3 config reference
- When translating a full `ModelConfig` into `step3p5_flash.py`, update only existing attrs
- Some keys map indirectly:
  - `disable_qk_norm` ↔ `use_qk_norm` (inverted)
  - `use_swiglu_limit` ↔ `swiglu_limit`
- If you change `num_layers`, keep all layer-wise lists in sync:
  - `qk_rope_head_dim`
  - `rope_theta`
  - `use_fused_qknorm_and_rope`
  - `use_swiglu_limit`
  - `use_swiglu_limit_shared`

## 6. Parallelism and Checkpointing

### Parallel state

- Global `PM` in `steptronoss.core.parallel_state` is the `ParallelManager`
- Typical flow:
  - `PM.initialize()`
  - `PM.set_mesh(parallel_cfg)`
  - or `with PM.use_mesh(parallel_cfg): ...`
- Common helpers:
  - `PM.define_parallel(pattern, **sizes)`
  - `PM.size_of("TP")`
  - `PM.rank_in("DP")`
  - `PM.group_of("PP")`
  - `PM.ranks_of("EP")`
- VPP uses:
  - `virtual_pipeline_model_parallel_size`
  - `get_vpp_rank()`
  - `set_vpp_rank()`
- `model_cfg.pipeline_activation_cpu_offload` applies to both `PPScheduler` and `VPPScheduler`; it is implemented with `torch.autograd.graph.save_on_cpu(...)`, currently requires `model_cfg.recompute=True`, and should be treated as graph-preserving activation offload rather than a replacement for `recompute`.

### EP / TP sizing

- `ParallelConfig.sanity_check()` requires:
  - `WORLD_SIZE` divisible by attention MP size = `PP * TP * CP`
  - `WORLD_SIZE` divisible by MoE MP size = `PP * ETP * EP`
- For an 8-GPU run with `TP=8` and `EP=8`, set:
  - `expert_tensor_parallel_size=1`
  - otherwise MoE MP size becomes 64 and the config is invalid
- In mixed dense/MoE topologies, expert params are reduced over `EDP`, not dense `DP`; the current gradient manager compensates with `TP/EP` scaling on expert grad buffers before the `EDP` reduction, so check that path before blaming an apparent extra `EP` factor.

### Checkpoint reshape

- `steptronoss/checkpointing/reshape_ops.py` contains reshape primitives:
  - `VocabPad`
  - `ColumnParallel` / `RowParallel`
  - `KeepThisTP` / `KeepThisEP`
  - `GQAMergeQKV`
  - `FFNMergeGateUp`
  - `UnbindMoE`
  - `Rename`
  - `Inverse`
- Typical usage:
  - build `Script(src=..., op=..., dst=...)`
  - return `OnlineReshaper(scripts)`
- For expert slicing from per-expert keys:
  - use `Inverse(UnbindMoE(...)) + KeepThisEP()` before TP ops

## 7. RLVR Priors

- `TrainableItem` should not pickle / serialize tokenizer instances; drop them in `__getstate__`
- `model_name` often includes `exp_id`; persist a template like `deployed-model-{EXP_ID}` so resume survives `exp_id` changes

## 8. Optimization Guidance

### Muon

- Use `MuonConfig.mark_muon_params(model)` before grouping
- In experiments, prefer overriding `optimizer_cfg` via a `GradientManagerConfig` subclass that sets `optimizer_cfg = MuonConfig`
- Leave distributed optimizer on, but avoid byte-level sharding
- For Muon tests, prefer composing existing reshape ops instead of inventing new ones
- In `playground/sft/step3/*muon*`, `Step3p5MuonConfig.mark_muon_params` inlines the base Muon selection rules but tags trainable params with `ndim >= 2` as Muon candidates (still respecting embedding/name exclusions); Step3.5 Flash `GroupedExperts` merge ops use `UnbindMoE + Inverse(Column/RowParallel + KeepThisTP(group="ETP"))`.

### Triton workflow

- Follow:
  - `docs/TRITON_ACCELERATION_WORKFLOW.md`
  - `docs/TRITON_ACCELERATION_WORKFLOW_ZH.md`
- Rules:
  - optimized implementations belong under `steptronoss/model/optimizations/*`
  - semantic entrypoints stay in `steptronoss/model/utils/*`
  - expose alternatives through `@optimizable(...)`
  - alternatives must be strict drop-in replacements
  - add correctness tests, backward-aware benchmarks, and a short real experiment trace

## 9. Tests and Tooling

### Tooling notes

- `rg` may be unavailable; fall back to `find` / `grep`
- `python` may be missing and `python3` may not include `pytest`; prefer project tooling if available
- `tests/conftest.py` now applies a shared skip to every `@pytest.mark.node2` test unless the run is launched under `torchrun --nproc-per-node=2`; plain `pytest` should skip them instead of hanging in distributed init.

### GPU test notes

- This environment may not have worker / GPU access; avoid running GPU-only tests when the machine does not actually have GPUs
- `@pytest.mark.node2` tests should also use `pytest.mark.xdist_group("torchrun")`
- Test layout:
  - single-node GPU tests: `tests/test_muon_optimizer.py`
  - 2-node GPU tests: `tests/test_muon_optimizer_node2.py`
- `steptronoss/model/ep_dispatcher/deepep_dispatcher.py` must keep `recv_token_probs` differentiable and pass `grad_recv_token_probs` into `buffer.combine(...)`; otherwise router main-loss gradients are cut when `TokenDispatcher="deep_ep"`.

## 10. Debugging Priors

- `steptronoss.utils.memory_tracker.CMT` only records when `MEM_DIAGNOSE=1`
- If training hangs on `Waiting for debugger... ip: ... rank: 56`, check for a stray `debug(56)` in `steptronoss/core/trainers/lm_trainer.py`
- TorchDynamo graph breaks are often triggered by `Tensor.item()` in optimizable helpers; prefer tensor-safe checks like masked `amax` + `torch._assert`
- `steptronoss/model/common/rope.py` should keep RoPE cos/sin caches and cache-generation math in `torch.float32`; module-wide `.to()/cuda()/bfloat16()` may move the cache device, but must not downcast the cache dtype.
=======
*Last updated: 2026-03-13*
>>>>>>> cb4d747 (kimi init)
