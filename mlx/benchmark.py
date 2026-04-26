"""
Benchmark: RadixMLP vs Baseline on Qwen3-0.6B in MLX.

Creates a synthetic batch where sequences share a long common prefix,
then times the forward pass with and without RadixMLP prefix-sharing.

Usage:
    python mlx/benchmark.py [--model Qwen/Qwen3-0.6B] [--prefix-len 200]
                            [--suffix-len 32] [--batch-size 8] [--runs 30]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

# local modules (run from repo root or mlx/ dir)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_fold_and_scatter import compute_fold_and_scatter
from model import RadixQwen3, load_weights


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_batch(prefix_len: int, suffix_len: int, batch_size: int, vocab_size: int):
    """
    Build a synthetic batch where every sequence shares the same prefix.

    Returns:
        input_ids       : mx.array [total_tokens]
        position_ids    : mx.array [total_tokens]
        cu_seqlens      : List[int] length batch_size+1
        seq_len         : int  (prefix_len + suffix_len)
    """
    rng = np.random.default_rng(42)
    prefix  = rng.integers(1, vocab_size, size=prefix_len, dtype=np.int32)
    suffixes = rng.integers(1, vocab_size, size=(batch_size, suffix_len), dtype=np.int32)

    tokens_list = []
    pos_list    = []
    seq_len     = prefix_len + suffix_len

    for b in range(batch_size):
        seq = np.concatenate([prefix, suffixes[b]])
        tokens_list.append(seq)
        pos_list.append(np.arange(seq_len, dtype=np.int32))

    input_ids    = mx.array(np.concatenate(tokens_list))
    position_ids = mx.array(np.concatenate(pos_list))
    cu_seqlens   = list(range(0, batch_size * seq_len + 1, seq_len))
    return input_ids, position_ids, cu_seqlens, seq_len


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed_run(fn, warmup: int = 5, runs: int = 30):
    """Run fn() warmup+runs times; return mean and std of the timed runs in ms."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    arr = np.array(times)
    return arr.mean(), arr.std()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prefix-len",  type=int, default=200)
    parser.add_argument("--suffix-len",  type=int, default=32)
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--runs",        type=int, default=30)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Download & load model
    # -----------------------------------------------------------------------
    print(f"Downloading {args.model} …")
    model_dir = Path(snapshot_download(args.model))
    cfg = json.loads((model_dir / "config.json").read_text())

    # Build model
    model_cfg = {
        "vocab_size":             cfg["vocab_size"],
        "hidden_size":            cfg["hidden_size"],
        "intermediate_size":      cfg["intermediate_size"],
        "num_hidden_layers":      cfg["num_hidden_layers"],
        "num_attention_heads":    cfg["num_attention_heads"],
        "num_key_value_heads":    cfg["num_key_value_heads"],
        "rms_norm_eps":           cfg.get("rms_norm_eps", 1e-6),
        "rope_theta":             cfg.get("rope_theta", 10000.0),
        "head_dim":               cfg.get("head_dim",
                                    cfg["hidden_size"] // cfg["num_attention_heads"]),
    }
    print(f"Building model: {cfg['num_hidden_layers']} layers, "
          f"hidden={cfg['hidden_size']}, intermediate={cfg['intermediate_size']}")

    model = RadixQwen3(model_cfg)

    # Load weights from safetensors
    print("Loading weights …")
    hf_weights: dict = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        import mlx.core as mx
        hf_weights.update(mx.load(str(sf)))
    load_weights(model, hf_weights)
    mx.eval(model.parameters())
    print(f"  Loaded {len(hf_weights)} weight tensors.")

    # -----------------------------------------------------------------------
    # 2. Build batch
    # -----------------------------------------------------------------------
    input_ids, position_ids, cu_seqlens, seq_len = make_batch(
        args.prefix_len, args.suffix_len, args.batch_size, cfg["vocab_size"]
    )
    total_tokens   = int(input_ids.shape[0])
    mx.eval(input_ids, position_ids)

    # Compute radix indices (CPU, negligible cost vs forward pass)
    input_ids_np  = np.array(input_ids.tolist(), dtype=np.int32)
    fold_gather_py, scatter_py = compute_fold_and_scatter(input_ids_np, cu_seqlens)
    compact_tokens = len(fold_gather_py)

    fold_gather     = mx.array(fold_gather_py,  dtype=mx.int32)
    scatter_indices = mx.array(scatter_py,      dtype=mx.int32)
    # identity indices for baseline (no sharing)
    identity = mx.arange(total_tokens, dtype=mx.int32)

    compression = total_tokens / compact_tokens

    print(f"\n{'='*55}")
    print(f"Batch config")
    print(f"  Sequences      : {args.batch_size}")
    print(f"  Prefix length  : {args.prefix_len} tokens (shared)")
    print(f"  Suffix length  : {args.suffix_len} tokens (unique)")
    print(f"  Seq length     : {seq_len} tokens")
    print(f"  Original tokens: {total_tokens}")
    print(f"  Compact tokens : {compact_tokens}")
    print(f"  MLP compression: {compression:.2f}x")
    print(f"{'='*55}\n")

    # -----------------------------------------------------------------------
    # 3. Benchmark baseline (skip_radix=True → identity indices)
    # -----------------------------------------------------------------------
    def run_baseline():
        out = model(
            input_ids, position_ids, cu_seqlens,
            fold_gather=identity,
            scatter_indices=identity,
            skip_radix=True,
        )
        mx.eval(out)
        return out

    def run_radix():
        out = model(
            input_ids, position_ids, cu_seqlens,
            fold_gather=fold_gather,
            scatter_indices=scatter_indices,
            skip_radix=False,
        )
        mx.eval(out)
        return out

    # Verify outputs match
    print("Verifying correctness (baseline vs RadixMLP) …")
    out_base  = run_baseline()
    out_radix = run_radix()
    abs_diff = mx.abs(out_base - out_radix)
    max_diff  = float(mx.max(abs_diff).item())
    mean_diff = float(mx.mean(abs_diff).item())
    # bfloat16 has ~0.78% relative precision; over 28 layers 5e-2 is acceptable
    ok = max_diff < 5e-2
    print(f"  Max  |Δ|: {max_diff:.2e}   Mean |Δ|: {mean_diff:.2e}  "
          f"{'✓ numerically OK (bfloat16)' if ok else '✗ UNEXPECTED MISMATCH'}\n")

    # -----------------------------------------------------------------------
    # 4. Timed benchmark
    # -----------------------------------------------------------------------
    print(f"Benchmarking ({args.runs} runs each, {5} warmup) …\n")
    base_mean, base_std  = timed_run(run_baseline, warmup=5, runs=args.runs)
    radix_mean, radix_std = timed_run(run_radix,   warmup=5, runs=args.runs)
    speedup = base_mean / radix_mean

    print(f"{'='*55}")
    print(f"Results")
    print(f"  Baseline  : {base_mean:7.1f} ± {base_std:.1f} ms")
    print(f"  RadixMLP  : {radix_mean:7.1f} ± {radix_std:.1f} ms")
    print(f"  Speedup   : {speedup:.2f}x")
    print(f"{'='*55}")

    # Throughput
    base_tps  = total_tokens / (base_mean  / 1e3)
    radix_tps = total_tokens / (radix_mean / 1e3)
    print(f"\nThroughput (original tokens/sec)")
    print(f"  Baseline  : {base_tps:,.0f} tok/s")
    print(f"  RadixMLP  : {radix_tps:,.0f} tok/s")

    # Sweep over different prefix lengths to show scaling
    print(f"\n{'='*55}")
    print("Prefix-length sweep (suffix=32, batch=8)")
    print(f"  {'prefix':>8}  {'compression':>12}  {'baseline ms':>12}  {'radix ms':>10}  {'speedup':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")

    for pl in [32, 64, 128, 256, 384]:
        ids_s, pos_s, cu_s, sl_s = make_batch(pl, args.suffix_len, args.batch_size, cfg["vocab_size"])
        mx.eval(ids_s, pos_s)
        ids_np = np.array(ids_s.tolist(), dtype=np.int32)
        fg_py, sc_py = compute_fold_and_scatter(ids_np, cu_s)
        n_orig    = int(ids_s.shape[0])
        n_compact = len(fg_py)
        comp_ratio = n_orig / n_compact

        fg_s = mx.array(fg_py, dtype=mx.int32)
        sc_s = mx.array(sc_py, dtype=mx.int32)
        id_s = mx.arange(n_orig, dtype=mx.int32)

        def _base():
            o = model(ids_s, pos_s, cu_s, id_s, id_s, skip_radix=True)
            mx.eval(o)

        def _radix():
            o = model(ids_s, pos_s, cu_s, fg_s, sc_s, skip_radix=False)
            mx.eval(o)

        bm, _ = timed_run(_base,  warmup=3, runs=15)
        rm, _ = timed_run(_radix, warmup=3, runs=15)
        sp = bm / rm
        print(f"  {pl:>8}  {comp_ratio:>12.2f}x  {bm:>11.1f}ms  {rm:>9.1f}ms  {sp:>7.2f}x")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()
