"""
Pure-Python trie-based prefix deduplication — drop-in replacement for the
Rust `compute_fold_and_scatter` used by the PyTorch implementation.

Semantics (matching the Rust library):
  fold_gather[compact_idx]   = original_idx   (compact → original)
  scatter_indices[orig_idx]  = compact_idx    (original → compact)

index_select(original_tensor, fold_gather)   → compact tensor
index_select(compact_tensor,  scatter_indices) → original tensor
"""


def compute_fold_and_scatter(input_ids, cu_seq_lengths):
    """
    Args:
        input_ids:       indexable of int, length = total tokens
        cu_seq_lengths:  indexable of int, length = batch_size + 1

    Returns:
        fold_gather     : list[int]  length = num_compact_tokens
        scatter_indices : list[int]  length = num_original_tokens
    """
    num_tokens = len(input_ids)
    scatter_indices = [0] * num_tokens
    fold_gather = []

    # trie node: dict[token_id] -> [compact_idx, children_dict]
    trie: dict = {}

    for b in range(len(cu_seq_lengths) - 1):
        start = int(cu_seq_lengths[b])
        end   = int(cu_seq_lengths[b + 1])
        node  = trie
        for orig_idx in range(start, end):
            tok = int(input_ids[orig_idx])
            if tok not in node:
                compact_idx = len(fold_gather)
                fold_gather.append(orig_idx)
                node[tok] = [compact_idx, {}]
            compact_idx = node[tok][0]
            scatter_indices[orig_idx] = compact_idx
            node = node[tok][1]          # descend into children

    return fold_gather, scatter_indices
