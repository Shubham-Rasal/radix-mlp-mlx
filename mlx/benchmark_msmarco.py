"""
MSMARCO benchmark for RadixMLP vs Baseline on Qwen3-0.6B in MLX.

Loads real MSMARCO passages, applies the Qwen3 reranking chat template,
tokenizes, and times batched forward passes with and without RadixMLP.

The key: for each query, ALL its passages share the same prefix
  [system msg] + [instruction] + [query text]
Only the <Document> part differs — ideal for RadixMLP.

Usage:
    python mlx/benchmark_msmarco.py [--n-passages 300] [--runs 5]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from compute_fold_and_scatter import compute_fold_and_scatter
from model import RadixQwen3, load_weights

# ---------------------------------------------------------------------------
# Qwen3 reranking template (matches existing benchmark/simple_msmarco_embed.py)
# ---------------------------------------------------------------------------
_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_PREFIX_TMPL  = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: "
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_passage(query: str, doc: str) -> str:
    prefix = _PREFIX_TMPL.format(system=_SYSTEM, instruction=_INSTRUCTION, query=query)
    return prefix + doc + _SUFFIX


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_msmarco_batches(n_passages: int):
    """
    Stream MSMARCO validation set, group passages by query.
    Returns list of dicts: {query, passages: [str], formatted: [str]}
    Stops once we have at least n_passages total passages.
    """
    print(f"Streaming MSMARCO validation set (target: {n_passages} passages) …")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)

    batches = []
    total = 0
    for row in ds:
        query = row["query"]
        docs  = row["passages"]["passage_text"]
        formatted = [format_passage(query, d) for d in docs]
        batches.append({"query": query, "passages": docs, "formatted": formatted})
        total += len(docs)
        if total >= n_passages:
            break

    print(f"  Loaded {len(batches)} queries, {total} passages total.")
    return batches


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_batch(texts, tokenizer, max_len=512):
    """
    Tokenize a list of strings without padding.
    Returns (input_ids_flat, position_ids_flat, cu_seqlens, seq_len_list).
    Sequences are truncated to max_len.
    """
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    ids_list = enc["input_ids"]

    cu = [0]
    flat_ids, flat_pos = [], []
    for ids in ids_list:
        flat_ids.extend(ids)
        flat_pos.extend(range(len(ids)))
        cu.append(cu[-1] + len(ids))

    return (
        mx.array(flat_ids,  dtype=mx.int32),
        mx.array(flat_pos,  dtype=mx.int32),
        cu,
        [len(ids) for ids in ids_list],
    )


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def run_forward(model, input_ids, position_ids, cu_seqlens,
                fold_gather, scatter_indices, skip_radix):
    out = model(input_ids, position_ids, cu_seqlens,
                fold_gather, scatter_indices, skip_radix)
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# Per-query benchmark
# ---------------------------------------------------------------------------

def bench_query(model, input_ids, position_ids, cu_seqlens, runs):
    n = int(input_ids.shape[0])
    identity = mx.arange(n, dtype=mx.int32)

    # Radix indices
    ids_np = np.array(input_ids.tolist(), dtype=np.int32)
    fg_py, sc_py = compute_fold_and_scatter(ids_np, cu_seqlens)
    fg = mx.array(fg_py, dtype=mx.int32)
    sc = mx.array(sc_py, dtype=mx.int32)
    compact = len(fg_py)
    compression = n / compact if compact < n else 1.0

    # Warmup
    for _ in range(2):
        run_forward(model, input_ids, position_ids, cu_seqlens, identity, identity, True)
        run_forward(model, input_ids, position_ids, cu_seqlens, fg, sc, compact < n)

    base_times, radix_times = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_forward(model, input_ids, position_ids, cu_seqlens, identity, identity, True)
        base_times.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        run_forward(model, input_ids, position_ids, cu_seqlens, fg, sc, compact < n)
        radix_times.append((time.perf_counter() - t0) * 1e3)

    return {
        "total_tokens":  n,
        "compact_tokens": compact,
        "compression":   compression,
        "base_ms":   np.mean(base_times),
        "radix_ms":  np.mean(radix_times),
        "speedup":   np.mean(base_times) / np.mean(radix_times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n-passages",  type=int, default=300)
    parser.add_argument("--max-len",     type=int, default=512,
                        help="Max tokens per passage (truncation)")
    parser.add_argument("--runs",        type=int, default=5,
                        help="Timed runs per query batch")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    from huggingface_hub import snapshot_download
    print(f"Loading {args.model} …")
    model_dir = Path(snapshot_download(args.model))
    cfg = json.loads((model_dir / "config.json").read_text())

    model_cfg = {
        "vocab_size":          cfg["vocab_size"],
        "hidden_size":         cfg["hidden_size"],
        "intermediate_size":   cfg["intermediate_size"],
        "num_hidden_layers":   cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "rms_norm_eps":        cfg.get("rms_norm_eps", 1e-6),
        "rope_theta":          cfg.get("rope_theta", 10000.0),
        "head_dim":            cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
    }
    model = RadixQwen3(model_cfg)
    hf_weights: dict = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        hf_weights.update(mx.load(str(sf)))
    load_weights(model, hf_weights)
    mx.eval(model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"  Model ready.\n")

    # -----------------------------------------------------------------------
    # Load MSMARCO data
    # -----------------------------------------------------------------------
    batches = load_msmarco_batches(args.n_passages)

    # -----------------------------------------------------------------------
    # Benchmark per query batch
    # -----------------------------------------------------------------------
    print(f"Benchmarking {len(batches)} query batches ({args.runs} runs each) …\n")
    print(f"  {'query':30s}  {'passages':>8}  {'tokens':>7}  {'compress':>9}  "
          f"{'base ms':>9}  {'radix ms':>9}  {'speedup':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

    all_results = []
    total_passages = 0

    for batch in batches:
        texts = batch["formatted"]
        query_short = batch["query"][:28]

        input_ids, position_ids, cu_seqlens, seq_lens = tokenize_batch(
            texts, tokenizer, max_len=args.max_len
        )
        mx.eval(input_ids, position_ids)

        # Skip single-passage queries (no prefix sharing possible)
        if len(texts) < 2:
            continue

        result = bench_query(model, input_ids, position_ids, cu_seqlens, args.runs)
        result["query"]    = batch["query"]
        result["n_passages"] = len(texts)
        all_results.append(result)
        total_passages += len(texts)

        print(f"  {query_short:30s}  {len(texts):>8}  {result['total_tokens']:>7}  "
              f"{result['compression']:>8.2f}x  {result['base_ms']:>8.1f}ms  "
              f"{result['radix_ms']:>8.1f}ms  {result['speedup']:>7.2f}x")

    # -----------------------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------------------
    if not all_results:
        print("No results.")
        return

    mean_compress = np.mean([r["compression"] for r in all_results])
    mean_speedup  = np.mean([r["speedup"]      for r in all_results])
    mean_base     = np.mean([r["base_ms"]      for r in all_results])
    mean_radix    = np.mean([r["radix_ms"]     for r in all_results])

    # Weighted speedup by total tokens (closer to real throughput)
    total_base_ms  = sum(r["base_ms"]  * r["n_passages"] for r in all_results)
    total_radix_ms = sum(r["radix_ms"] * r["n_passages"] for r in all_results)
    weighted_speedup = total_base_ms / total_radix_ms

    print(f"\n{'='*75}")
    print(f"Summary  ({len(all_results)} query batches, {total_passages} passages)")
    print(f"{'='*75}")
    print(f"  Mean compression (MLP tokens saved) : {mean_compress:.2f}x")
    print(f"  Mean latency  — baseline            : {mean_base:.1f} ms/batch")
    print(f"  Mean latency  — RadixMLP            : {mean_radix:.1f} ms/batch")
    print(f"  Mean speedup  (per batch)            : {mean_speedup:.2f}x")
    print(f"  Weighted speedup (by passage count)  : {weighted_speedup:.2f}x")
    print(f"{'='*75}")

    # Show best and worst cases
    best  = max(all_results, key=lambda r: r["speedup"])
    worst = min(all_results, key=lambda r: r["speedup"])
    print(f"\n  Best  speedup: {best['speedup']:.2f}x  — \"{best['query'][:60]}\"")
    print(f"  Worst speedup: {worst['speedup']:.2f}x  — \"{worst['query'][:60]}\"")
    print(f"  (Worst = fewest passages / shortest shared prefix)")

    # -----------------------------------------------------------------------
    # Why so little gain? Explain and show the crossover
    # -----------------------------------------------------------------------
    print(f"""
Why ~1x speedup on MSMARCO?
  MSMARCO has 5-10 passages per query. The shared prefix (system msg + query)
  is ~80 tokens; each document is ~400 tokens. With 8 passages:
    compression = 8*(80+400) / (80 + 8*400) = 3840 / 3280 ≈ 1.17x MLP tokens
  That's too small to overcome overhead at these token counts.

  RadixMLP pays off when: many sequences (≥16) share a LONG prefix (≥128 tok).
  Example: reranking 64 candidates for one query → see benchmark_mlx.py.
""")

    # Crossover table: simulated "what-if N passages per query"
    print("What-if table: fixed prefix=120 tok, doc=380 tok, batch=N")
    print(f"  {'N passages':>10}  {'compress':>10}  {'MLP saving':>12}  {'expected speedup':>18}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*18}")
    prefix_tok = 120
    doc_tok    = 380
    mlp_frac   = 0.55   # rough fraction of compute that is MLP in Qwen3-0.6B
    for n in [2, 5, 8, 16, 32, 64]:
        orig    = n * (prefix_tok + doc_tok)
        compact = prefix_tok + n * doc_tok
        comp    = orig / compact
        # Amdahl's law: only MLP fraction benefits
        speedup = 1 / ((1 - mlp_frac) + mlp_frac / comp)
        print(f"  {n:>10}  {comp:>9.2f}x  {(1-1/comp)*100:>10.1f}%  {speedup:>17.2f}x")


if __name__ == "__main__":
    main()
