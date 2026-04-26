"""
Qwen3 transformer with RadixMLP prefix-sharing, implemented in MLX.

Matches the architecture of the PyTorch reference in train/qwen3_radix_torch_varlen.py
and loads HuggingFace Qwen3 weights directly.

Key shapes (no batch dim — all sequences are flattened):
  original space : [num_original_tokens, ...]
  compact space  : [num_compact_tokens, ...]   (num_compact ≤ num_original)

RadixMLP flow per layer:
  1. LayerNorm + Attention Q/K/V in compact space
  2. scatter → original space  (expand shared-prefix duplicates)
  3. Causal self-attention per sequence
  4. fold → compact space      (collapse duplicates back)
  5. o_proj + LayerNorm + MLP in compact space  ← MLP only runs on M tokens
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """RoPE. x: [T, heads, head_dim], cos/sin: [T, head_dim//2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:, None, :]   # [T, 1, half]
    s = sin[:, None, :]
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# Model sub-modules
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_heads    = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim     = cfg.get("head_dim", cfg["hidden_size"] // self.num_heads)
        self.scale        = self.head_dim ** -0.5
        hidden = cfg["hidden_size"]

        self.q_proj = nn.Linear(hidden, self.num_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden,    bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def __call__(
        self,
        x: mx.array,                   # [compact_T, hidden]
        cos: mx.array,                 # [compact_T, head_dim//2]
        sin: mx.array,
        cu_seqlens: List[int],
        fold_gather:     mx.array,     # [compact_T]
        scatter_indices: mx.array,     # [original_T]
        skip_radix: bool,
    ) -> mx.array:
        T = x.shape[0]

        q = self.q_proj(x).reshape(T, self.num_heads,    self.head_dim)
        k = self.k_proj(x).reshape(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(T, self.num_kv_heads, self.head_dim)

        # Per-head norms (Qwen3 uses qk-norm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE in compact space (positions are already correct)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand to original space so each sequence sees its own KV
        if not skip_radix:
            q = mx.take(q, scatter_indices, axis=0)
            k = mx.take(k, scatter_indices, axis=0)
            v = mx.take(v, scatter_indices, axis=0)

        # Per-sequence causal attention  [B, N_q, T, D]
        outputs: List[mx.array] = []
        for b in range(len(cu_seqlens) - 1):
            s, e = cu_seqlens[b], cu_seqlens[b + 1]
            # [1, heads, seq, dim]
            qb = q[s:e].transpose(1, 0, 2)[None]
            kb = k[s:e].transpose(1, 0, 2)[None]
            vb = v[s:e].transpose(1, 0, 2)[None]
            # GQA handled natively by mlx (no pre-tiling needed)
            out = mx.fast.scaled_dot_product_attention(
                qb, kb, vb, scale=self.scale, mask="causal"
            )  # [1, num_heads, seq, head_dim]
            outputs.append(out[0].transpose(1, 0, 2))   # [seq, heads, dim]

        attn = mx.concatenate(outputs, axis=0)           # [original_T, heads, dim]
        attn = attn.reshape(-1, self.num_heads * self.head_dim)

        # Fold back to compact space before o_proj
        if not skip_radix:
            attn = mx.take(attn, fold_gather, axis=0)

        return self.o_proj(attn)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.self_attn               = Attention(cfg)
        self.mlp                     = MLP(cfg["hidden_size"], cfg["intermediate_size"])
        self.input_layernorm         = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def __call__(self, x, cos, sin, cu_seqlens, fold_gather, scatter_indices, skip_radix):
        r = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, cu_seqlens, fold_gather, scatter_indices, skip_radix)
        x = r + x

        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)    # runs only on compact_T tokens ← the savings
        return r + x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        dim = head_dim // 2                                   # half-dim for RoPE
        inv_freq = 1.0 / (
            base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        self._inv_freq = inv_freq   # [dim//2]

    def __call__(self, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        pos   = position_ids.astype(mx.float32)[:, None]     # [T, 1]
        freqs = pos * self._inv_freq[None, :]                 # [T, dim//2]
        emb   = mx.concatenate([freqs, freqs], axis=-1)       # [T, head_dim//2]
        return mx.cos(emb), mx.sin(emb)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class RadixQwen3(nn.Module):
    """Qwen3 backbone with optional RadixMLP prefix-sharing."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [TransformerLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm   = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.rotary_emb = RotaryEmbedding(
            cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            base=cfg.get("rope_theta", 10000.0),
        )

    def __call__(
        self,
        input_ids:       mx.array,    # [num_original_tokens]
        position_ids:    mx.array,    # [num_original_tokens]
        cu_seqlens:      List[int],
        fold_gather:     mx.array,    # [num_compact_tokens]  compact→original
        scatter_indices: mx.array,    # [num_original_tokens] original→compact
        skip_radix:      bool,
    ) -> mx.array:
        # Work in compact space from the start
        if skip_radix:
            ids_c = input_ids
            pos_c = position_ids
        else:
            ids_c = mx.take(input_ids,    fold_gather)
            pos_c = mx.take(position_ids, fold_gather)

        x        = self.embed_tokens(ids_c)         # [compact_T, hidden]
        cos, sin = self.rotary_emb(pos_c)           # [compact_T, head_dim//2]

        for layer in self.layers:
            x = layer(x, cos, sin, cu_seqlens, fold_gather, scatter_indices, skip_radix)

        x = self.norm(x)

        # Scatter final hidden states back to original token order
        if not skip_radix:
            x = mx.take(x, scatter_indices, axis=0)

        return x   # [num_original_tokens, hidden_size]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_weights(model: RadixQwen3, hf_weights: Dict[str, mx.array]) -> None:
    """
    Map HuggingFace Qwen3 weight names → our model parameter names and load them.

    HF names:   model.embed_tokens.weight
                model.layers.{i}.self_attn.q_proj.weight
                model.layers.{i}.mlp.gate_proj.weight
                model.norm.weight
    Our names:  embed_tokens.weight          (strip "model." prefix)
                layers.{i}.self_attn.q_proj.weight
                layers.{i}.mlp.gate_proj.weight
                norm.weight
    """
    mapped: Dict[str, mx.array] = {}
    for hf_key, val in hf_weights.items():
        key = hf_key.removeprefix("model.")
        mapped[key] = val

    # mlx-lm sometimes stores layers as a dict with string keys; normalise
    params = model.parameters()
    updates = _flatten_and_match(params, mapped)
    model.load_weights(list(updates.items()))


def _flatten_and_match(
    params: Dict, mapped: Dict[str, mx.array], prefix: str = ""
) -> Dict[str, mx.array]:
    """Recursively flatten model parameter tree and match against mapped weights."""
    result: Dict[str, mx.array] = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_and_match(v, mapped, full_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    result.update(_flatten_and_match(item, mapped, f"{full_key}.{i}"))
        elif isinstance(v, mx.array):
            if full_key in mapped:
                result[full_key] = mapped[full_key]
    return result
