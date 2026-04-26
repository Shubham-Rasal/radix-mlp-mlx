## Update — April 2026: MSMARCO Benchmark Results

Ran the MLX port against 300 real MSMARCO passages (37 queries, 5–10 passages each) using the Qwen3 reranking chat template. Short version: **no measurable speedup** on natural MSMARCO batches.

The reason is structural. The compression formula is:

```
compression = N × (prefix + doc) / (prefix + N × doc)
```

With MSMARCO's natural batching (N ≈ 8, prefix ≈ 80 tokens, doc ≈ 400 tokens), compression ≈ 1.17×. That's too small to beat Python overhead at 1000–2000 total tokens per batch.

The synthetic benchmark (`mlx/benchmark.py`) tells a different story: when the shared prefix dominates (200–384 tokens vs 32-token unique suffix), speedup is 2.7–3.5× on Qwen3-0.6B.

**When RadixMLP pays off:**

| Scenario | Prefix | Doc | N | Compression | Speedup |
|----------|--------|-----|---|-------------|---------|
| MSMARCO reranking (natural) | 80 tok | 400 tok | 8 | 1.2× | ~1.0× |
| Long query reranking | 250 tok | 200 tok | 16 | 2.5× | ~1.5× |
| Synthetic (this port) | 200 tok | 32 tok | 8 | 4.1× | 2.7× |
| Synthetic (this port, large) | 384 tok | 32 tok | 8 | 5.2× | 3.5× |

The optimization is most useful for: long shared system prompts, server-side batching of concurrent requests sharing the same prefix, or RAG scenarios where a long document context is shared across many queries. Short-prefix embedding workloads (standard MSMARCO passage encoding) don't benefit.

---

# Devlog: Porting RadixMLP to MLX

*April 2026 — Shubham Rasal*

---

## What is RadixMLP?

When you run a batch of sequences through a transformer, a lot of them share a common prefix — a system prompt, a document header, whatever. A normal model doesn't care. It processes every token in every sequence, even if those tokens are identical across the whole batch.

RadixMLP exploits a simple observation: **the MLP in a transformer is stateless**. Each token's MLP output depends only on that token's hidden state — not on any neighbors. Attention is different; it needs to see the full sequence. But MLP? You can compute it once for a shared prefix token and reuse it everywhere.

The original repo implements this in Rust with PyTorch bindings and a CUDA kernel. The core idea is a **trie** (prefix tree) over your batch of sequences. Walk every sequence through the trie simultaneously; wherever two sequences share a token at the same depth, they share a trie node. Tokens that map to the same trie node only need to run through MLP once.

This gives you two index arrays:
- `fold_gather` — picks the unique tokens out of your original batch (original → compact)
- `scatter_indices` — maps them back after the MLP (compact → original)

---

## Why MLX?

The original implementation requires CUDA. I wanted to run this on Apple Silicon without any GPU or Rust build step. MLX is Apple's ML framework for M-series chips — it supports Metal, has lazy evaluation, and is reasonably fast for inference.

The goal: pure Python + MLX, no Rust, no CUDA, weights loaded straight from HuggingFace.

---

## The Port

Three files, each doing one thing:

**`compute_fold_and_scatter.py`** — the trie in plain Python. About 30 lines. Takes `input_ids` and cumulative sequence lengths, walks a dict-of-dicts trie, and spits out `fold_gather` and `scatter_indices`. No dependencies.

**`model.py`** — a full Qwen3 transformer in MLX. Attention, MLP, RMSNorm, RoPE, grouped-query attention. The twist is the per-layer logic:

```
compact tokens → norm → Q/K/V → RoPE
              → scatter to original space (mx.take)
              → per-sequence causal attention (one call per sequence)
              → fold back to compact space (mx.take)
              → o_proj → norm → MLP   ← only compact tokens here
```

MLX's `mx.fast.scaled_dot_product_attention` handles GQA natively (no need to tile K/V heads manually), and `mx.take` is the equivalent of PyTorch's `index_select`.

**`benchmark.py`** — loads Qwen3-0.6B weights from HuggingFace, builds a synthetic batch with a shared prefix, and times forward passes with and without the optimization.

---

## Results

Tested on Qwen3-0.6B, batch of 8 sequences, varying prefix length.

```
prefix   compression   baseline   radix     speedup
------   -----------   --------   -----     -------
    32        1.78x     120 ms    81 ms      1.47x
    64        2.40x     165 ms    87 ms      1.89x
   128        3.33x     271 ms   112 ms      2.42x
   200        4.07x     397 ms   148 ms      2.68x
   256        4.50x     491 ms   156 ms      3.15x
   384        5.20x     733 ms   208 ms      3.53x
```

At a 200-token shared prefix (4× compression), we get **2.7× speedup**. At 384 tokens (5.2× compression), **3.5× speedup**.

The speedup is real but doesn't match compression 1-to-1 because attention still runs on all N tokens. Only MLP is deduplicated. As the model gets bigger, MLP takes a larger fraction of total compute (the FFN hidden size grows faster than attention hidden size), so the gains would scale up with model size — consistent with the original paper showing 5× on an 8B model.

---

## Correctness

Output difference vs baseline: max 2e-2, mean 1e-4. This is pure floating-point drift — the weights are bfloat16, and running 28 layers in two slightly different computation orders accumulates rounding differences. The mean error is essentially zero; the max is in the last decimal place of bfloat16 precision.

---

## What didn't work immediately

**Attention masking across sequences.** Initially I ran a single big attention over the whole batch with a block-diagonal mask. This was wrong — each sequence needs its own causal mask and shouldn't attend across boundaries. Fixed by looping over sequences and calling attention individually.

**`mx.take` vs `index_select`.** MLX's equivalent is `mx.take(tensor, indices, axis=0)`. The semantics are identical, just a different name.

**Weight name mismatch.** HuggingFace weights have a `model.` prefix on everything (`model.layers.0.self_attn.q_proj.weight`). My model doesn't. A one-line `removeprefix("model.")` fixed it.

---

## What's interesting about this

The trie is the elegant part. You don't have to do anything clever in the model — just build the right index arrays ahead of time, and `mx.take` does the rest. The model itself stays simple and clean. All the prefix-sharing logic lives in a 30-line pure Python function that runs on CPU in microseconds.

This also means it's completely framework-agnostic. The same `fold_gather` and `scatter_indices` arrays work with PyTorch, MLX, JAX — anything that has an `index_select`-style operation.

---

## Next steps

- Try on Qwen3-1.7B and 4B to see if the speedup scales as expected
- Benchmark on real document embedding workloads (e.g. MSMARCO) instead of synthetic data
- The per-sequence attention loop in Python adds overhead; a batched padded-mask approach might be faster for small prefix lengths




# RadixMLP

RadixMLP enables prefix-based computation sharing for transformer models, eliminating redundant MLP activations when processing batches with shared prefixes. Achieves up to 5× speedup for embedding workloads.

## Key Features

- Prefix deduplication: Automatically identifies and compacts shared subsequences across batched sequences
- Stateless operation: Single forward pass optimization, no cache management required
- Weight compatible: Drop-in replacement for standard transformer models
- compatible with autograd
- Production ready: Integrated into [text-embeddings-inference upstream](https://github.com/huggingface/text-embeddings-inference/pull/761)

## Project Structure

```
radix-mlp/
├── package/           # Rust core library - `cargo add radix_mlp`
├── python_bindings/   # Python interface (PyTorch + NumPy) `pip install radix_mlp`
├── train/             # Training scripts & experimental results for autograd
├── benchmark/         # Performance benchmarks via rest api
├── kernels            # cuda kernels for fast index-select 
└── Readme.md          # This file
```

## Implementations

### Rust Core Library (`package/`)
- High-performance `compute_fold_and_scatter()` algorithm
- Comprehensive test suite and benchmarks
- MIT License, `pip install radix-mlp`

### Python Bindings (`python_bindings/`)
- NumPy interface: `compute_fold_and_scatter()`
- PyTorch interface: `compute_fold_and_scatter_torch()`
- Device support (CPU/GPU) with automatic conversion

### Training & Proofs (`train/`)
- PyTorch: `radix_torch_varlen.py` - Complete Qwen3 implementation
- Rust: `radix_mlp_qwen3_modeling_varlen.rs` - Ground truth implementation
- Mathematical Proofs:
  - `proof_radix_forward_backward.py` - Forward/backward equivalence
  - `proof_radix_identical_inference.py` - Inference equivalence
  - `test_huggingface_comparison.py` - Weight compatibility

## Benchmarks

### Synthetic Benchmarks
- Up to 5× speedup on Qwen3 models (0.6B-8B)
- Performance gains scale with model size and prefix length
- Detailed results in package/benches/

### Real-World Benchmarks (`benchmark/`)
- MSMARCO v1.1 query-passage embedding
- End-to-end TEI integration testing
- 1.4-1.6× latency reduction in production workloads

## Usage

### Python
```python
from radix_mlp import compute_fold_and_scatter_torch
import torch

# Two sequences with shared prefix
input_ids = torch.tensor([1, 2, 3, 1, 2, 4])
position_ids = torch.tensor([0, 1, 2, 0, 1, 2])
cu_seq_lengths = torch.tensor([0, 3, 6])

compact_ids, _, _, _ = compute_fold_and_scatter_torch(
    input_ids, position_ids, cu_seq_lengths
)
print(f"Compression: {len(input_ids)} → {len(compact_ids)} tokens")
```

### Rust
```rust
use radix_mlp::compute_fold_and_scatter;

let input_ids = vec![1, 2, 3, 1, 2, 4];
let position_ids = vec![0, 1, 2, 0, 1, 2];
let cu_seq_lengths = vec![0, 3, 6];

let (compact_ids, _, _, _) = compute_fold_and_scatter(
    &input_ids, &position_ids, &cu_seq_lengths, None
);
```

## Production Integration

### Text-Embeddings-Inference
Upstream PR: [huggingface/text-embeddings-inference#761](https://github.com/huggingface/text-embeddings-inference/pull/761)

- Zero-configuration enablement
- Automatic thresholding
- Production-tested with MSMARCO workloads

## Documentation

- Rust Library: See `package/README.md`
- Python Bindings: See `python_bindings/README.md`  
- Training & Proofs: See `train/README.md`
- Benchmarks: See `benchmark/README.md`

## Verification Status

- Forward pass: Numerically identical to baseline
- Backward pass: Gradients identical to baseline
- Weight compatibility: 100% compatible with transformers
- Production: TEI upstream integration complete

## License

MIT License - Copyright (c) 2025 michaelfeil

## Performance Summary

| Model | Synthetic Speedup | End-to-End Speedup |
|-------|-------------------|-------------------|
| 0.6B  | 2.7×              | 1.44×             |
| 4B    | 4.1×              | 1.56×             |
| 8B    | 5.0×              | 1.59×             |

*Results from paper - see benchmarks for details*
