# Fine-grained Selective Checkpointing Design

## Overview

StepTronOSS supports fine-grained activation recomputation with per-layer, submodule-level toggles, enabling selective recomputation of only the most memory-intensive components to reduce peak memory with minimal overhead.

## Design Philosophy

The framework provides **modular, configurable checkpointing** at multiple granularity levels:

1. **Submodule-level**: Toggle individual components (attention, FFN, normalization)
2. **Operation-level**: Fine-grained control within operations (SiLU fusion, QK-Norm)
3. **Distributed storage**: TP-aware activation distribution for memory efficiency

## Architecture

### 1. Configuration Hierarchy

```
DecoderLLMConfig.recompute (list[str] | bool)
    ├── "attention"      → Attention module checkpointing
    ├── "attn_norm"      → Pre-attention LayerNorm checkpointing
    ├── "feed_forward"   → FFN/MoE module checkpointing
    └── "ffn_norm"       → Pre-FFN LayerNorm checkpointing

AttentionConfig.recompute_qknorm_rope (bool)
    └── QK-Norm + RoPE recomputation

FeedForwardConfig.swiglu_recompute_silu_out_proj (bool)
    └── SiLU activation fusion recomputation
```

### 2. Core Components

#### 2.1 CheckpointFunction (`steptronoss/core/tensor_parallel/random.py`)

Custom autograd function implementing the checkpointing mechanism:

```python
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, kwargs, *args):
        # Save RNG states for deterministic recomputation
        ctx.rng_states = _get_all_rng_states()
        
        # Execute forward without saving intermediate activations
        with torch.no_grad():
            outputs = run_function(*args, **kwargs)
        
        # Optional: Distribute activations across TP ranks
        if distribute_saved_activations:
            split_tensor_into_1d_equal_chunks(args[0])
        
        # Save only inputs (not intermediate activations)
        ctx.save_for_backward(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Restore distributed activations if needed
        if ctx.distribute_saved_activations:
            gather_split_1d_tensor(inputs[0])
        
        # Restore RNG states for deterministic recomputation
        _set_all_rng_states(*ctx.rng_states)
        
        # Recompute forward pass
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
        
        # Compute gradients
        torch.autograd.backward(outputs, grad_outputs)
        return grads
```

#### 2.2 TransformerBlock Integration (`steptronoss/model/decoder_model.py`)

Conditional checkpointing based on configuration:

```python
class TransformerBlock(nn.Module):
    def forward(self, x, cu_seqlens=None, max_seq_len=None, position_id=None, **kwargs):
        # Pre-attention normalization with optional checkpointing
        if self.training and "attn_norm" in self.recompute:
            attn_in = checkpoint(self.attention_norm, self.distribute_saved_activations, x)
        else:
            attn_in = self.attention_norm(x)
        
        # Attention with optional checkpointing
        if self.training and "attention" in self.recompute:
            h = x + checkpoint(
                self.attention,
                self.distribute_saved_activations,
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )
        else:
            h = x + self.attention(...)
        
        # Pre-FFN normalization with optional checkpointing
        if self.training and "ffn_norm" in self.recompute:
            ffn_in = checkpoint(self.ffn_norm, self.distribute_saved_activations, h)
        else:
            ffn_in = self.ffn_norm(h)
        
        # FFN/MoE with optional checkpointing (handled internally)
        out = h + self.feed_forward(ffn_in, recompute="feed_forward" in self.recompute)
        return out
```

#### 2.3 FeedForward with SiLU Fusion (`steptronoss/model/common/feed_forward.py`)

Advanced optimization: fuse SiLU activation with output projection:

```python
class FeedForward(torch.nn.Module):
    def __init__(self, cfg: FeedForwardConfig, layer_id: int):
        self.fuse_activation_w2 = cfg.swiglu_recompute_silu_out_proj
        
        self.w1 = tensor_parallel.ColumnParallelLinear(...)
        self.w2 = tensor_parallel.RowParallelLinear(
            ...,
            custom_pre_recompute_function=(
                self.activation if self.fuse_activation_w2 else None
            ),
        )

    def forward(self, x, recompute=False, **kwargs):
        if recompute:
            return tensor_parallel.checkpoint(
                self._forward, self.distribute_saved_activations, x
            )
        return self._forward(x)

    def _forward(self, x):
        if self.fuse_activation_w2:
            # Store w1 output (not SiLU output) - saves memory
            x = self.w1(x)[0]
            output = self.w2(x)[0]  # SiLU recomputed in backward via custom_pre_recompute_function
        else:
            x = self.activation(self.w1(x)[0])
            output = self.w2(x)[0]
        return output
```

#### 2.4 MoE Block Handling (`steptronoss/model/common/moe_block.py`)

Special handling for MoE: router is never checkpointed (needs gradients), experts can be:

```python
class MoEBlock(nn.Module):
    def forward(self, x: torch.FloatTensor, recompute: bool = False) -> torch.FloatTensor:
        # Router is NEVER checkpointed - needs to propagate gradients
        logits = self.gate(x)
        token_expert_ids, token_weights, aux_loss = self.forward_router(logits)
        
        # Expert computation CAN be checkpointed
        if PM.size_of("EP") > 1:
            if recompute:
                output = checkpoint(
                    self.forward_experts_ep,
                    self.recompute_dis_activation,
                    x, token_expert_ids, token_weights
                )
            else:
                output = self.forward_experts_ep(x, token_expert_ids, token_weights)
        else:
            if recompute:
                output = checkpoint(
                    self.forward_experts_tp,
                    self.recompute_dis_activation,
                    x, token_expert_ids, token_weights
                )
            else:
                output = self.forward_experts_tp(x, token_expert_ids, token_weights)
        
        return output
```

## Key Optimizations

### 1. Distributed Activation Storage

In Tensor Parallelism, activations are distributed across ranks to save memory:

```python
# Forward: split and save only local chunk
if distribute_saved_activations:
    ctx.input_0_shape = args[0].data.shape
    safely_set_viewless_tensor_data(
        args[0],
        split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
    )

# Backward: gather to restore full activation
if ctx.distribute_saved_activations:
    safely_set_viewless_tensor_data(
        inputs[0],
        gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
    )
```

**Benefit**: Reduces activation memory by `TP_size` times per layer.

### 2. SiLU Activation Fusion

Instead of storing SiLU output (large), store w1 output and recompute SiLU:

```
Standard SwiGLU:
  Forward: x → w1 → split → SiLU(gate) * up → w2 → output
  Store: SiLU(gate) * up (large activation)

Fused SwiGLU:
  Forward: x → w1 → (store this) → [SiLU + mul] → w2 → output
  Store: w1 output (smaller, or recompute from stored input)
  Backward: Recompute SiLU on-the-fly
```

**Benefit**: ~30% memory reduction for FFN activations.

### 3. RNG State Management

Deterministic recomputation requires identical random states:

```python
def _get_all_rng_states():
    """Collect CPU, CUDA, and TP-specific RNG states"""
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        get_cuda_rng_tracker().get_states(),
    )

@contextlib.contextmanager
def _fork_rng():
    """Context manager to restore RNG states after recomputation"""
    current_states = _get_all_rng_states()
    try:
        yield
    finally:
        _set_all_rng_states(*current_states)
```

### 4. Variable Detachment

Create fresh computation graph nodes for recomputation:

```python
def detach_variable(inputs):
    """Detach from original graph but keep gradient tracking"""
    if isinstance(inputs, tuple):
        return tuple(detach_variable(x) for x in inputs)
    return inputs.detach().requires_grad_(inputs.requires_grad)
```

**Purpose**: Prevent gradient flow to original computation graph, avoid "backward twice" errors.

## Configuration Examples

### Example 1: Full Checkpointing (Maximum Memory Saving)

```python
class MyModelConfig(DecoderLLMConfig):
    def __init__(self):
        super().__init__()
        self.recompute = True  # Enable all: ["attention", "attn_norm", "feed_forward", "ffn_norm"]
```

### Example 2: Selective Checkpointing (Balanced)

```python
class MyModelConfig(DecoderLLMConfig):
    def __init__(self):
        super().__init__()
        # Only checkpoint memory-heavy components
        self.recompute = ["attention", "feed_forward"]  # Skip norms (cheap to recompute)
        
        # Attention-specific optimization
        self.attn_cfg.recompute_qknorm_rope = True
        
        # FFN-specific optimization
        self.ffn_cfg.swiglu_recompute_silu_out_proj = True
```

### Example 3: Minimal Checkpointing (Maximum Speed)

```python
class MyModelConfig(DecoderLLMConfig):
    def __init__(self):
        super().__init__()
        self.recompute = []  # Disable all checkpointing
        # Or: self.recompute = False
```

### Example 4: Per-Layer Configuration

```python
class MyModelConfig(DecoderLLMConfig):
    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        """Different recompute strategy per layer"""
        layers = [...]  # layer ids for this pp rank
        allocation = []
        for i, layer_id in enumerate(layers):
            if layer_id < 10:
                # Early layers: full checkpointing
                allocation.append({"recompute": ["attention", "attn_norm", "feed_forward", "ffn_norm"]})
            elif layer_id < 40:
                # Middle layers: selective
                allocation.append({"recompute": ["attention", "feed_forward"]})
            else:
                # Late layers: minimal
                allocation.append({"recompute": []})
        return allocation
```

## Comparison with Industry Standards

| Feature | StepTronOSS | Megatron-LM | PyTorch SAC |
|---------|-------------|-------------|-------------|
| Submodule-level | ✅ String list | ✅ Predefined modes | ❌ |
| SiLU Fusion | ✅ `custom_pre_recompute_function` | ✅ Similar | ❌ |
| QK-Norm/RoPE | ✅ `recompute_qknorm_rope` | ❌ | ❌ |
| MoE-aware | ✅ Router excluded | ⚠️ Partial | ❌ |
| Distributed storage | ✅ `distribute_saved_activations` | ✅ | ❌ |
| Operator-level | ❌ | ❌ | ✅ Policy-based |
| Auto-tuning | ❌ | ⚠️ Manual config | ❌ |

## Best Practices

1. **Start with selective recomputation**: `["attention", "feed_forward"]` is usually optimal
2. **Enable SiLU fusion**: Always set `swiglu_recompute_silu_out_proj=True` for SwiGLU models
3. **Use distributed storage**: Enable `distribute_saved_activations` when TP > 1
4. **Profile memory**: Use `CMT` (Context Memory Tracker) to identify bottlenecks
5. **Layer-wise tuning**: Early/late layers may have different memory characteristics

## Debugging

Enable sanity check mode to verify recomputation correctness:

```bash
export RECOMPUTE_SANITY_CHECK=1
```

This uses `CheckpointFunctionWithSanityCheck` which compares forward outputs with recomputed outputs.

## References

- [Megatron-LM: Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)
- [PyTorch Activation Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Checkmate: Optimal Checkpointing](https://arxiv.org/abs/1910.02653)
