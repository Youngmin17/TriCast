"""AWQ's weight-clipping search (tricast_reimplementation of mit-han-lab/llm-awq d6e797a4, MIT,
``awq/quantize/auto_clip.py``: ``auto_clip_layer`` / ``auto_clip_block`` / ``apply_clip``).

For every output row and group of ``group_size`` input columns, the largest magnitude is shrunk by
``1 - i / n_grid`` for ``i < max_shrink * n_grid``; each candidate clamps the group to ``[-max, max]``,
quantizes it with the layer's own weight quantizer, and the candidate with the smallest mean squared
group output error over ``n_sample_token`` calibration tokens wins (strictly smaller; the earliest on a tie).
The batches of output rows, the token subsampling and the comparisons are the official ones; the
quantizer is the layer's QuantSpec rather than the official ``pseudo_quantize_tensor``, and a calibration
set smaller than ``n_sample_token`` keeps every token (the official stride would be 0). The fidelity test
runs the pinned official function on the same tensors (tests/test_awq_clip.py).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# auto_clip_block skips these Linears: "due to qk bmm, it is hard to clip precisely"
SKIPPED_NAMES = ("q_", "k_", "query", "key", "Wqkv")


def clip_skipped(name: str) -> bool:
    """Whether auto_clip_block leaves the Linear called ``name`` unclipped (its name within the block)."""
    leaf = name.rsplit(".", 1)[-1]
    return any(word in leaf for word in SKIPPED_NAMES)


@torch.no_grad()
def clip_search(
    w: torch.Tensor,
    x: torch.Tensor,
    quantize: Callable[[torch.Tensor], torch.Tensor],
    group_size: int,
    *,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
) -> torch.Tensor:
    """The clip value of every [row, group] of ``w`` ([N, K]) for inputs ``x`` ([tokens, K]); ``quantize``
    maps an [rows, K] weight to its quantize -> dequantize result. Returns [N, K // group_size, 1]."""
    if w.ndim != 2 or x.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("clip_search needs w [N, K] and x [tokens, K]")
    if group_size < 1 or w.shape[1] % group_size:
        raise ValueError(f"K={w.shape[1]} is not a multiple of the group size {group_size}")
    if x.shape[0] < 1:
        raise ValueError("clip_search needs calibration tokens")
    rows = w.shape[0]
    feat = x.reshape(1, x.shape[0], -1, group_size)
    feat = feat[:, 0::max(1, feat.shape[1] // n_sample_token)]
    w = w.reshape(rows, 1, -1, group_size)
    batch = 256 if rows % 256 == 0 else 64
    if rows % batch:
        batch = rows  # the official asserts rows % 64 == 0; one batch keeps every row's result the same
    best_all = []
    for start in range(0, rows, batch):
        part = w[start:start + batch]
        org_max_val = part.abs().amax(dim=-1, keepdim=True)
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        org_out = (feat * part).sum(dim=-1)
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            cur_w = torch.clamp(part, -max_val, max_val)
            q_w = quantize(cur_w.reshape(part.shape[0], -1)).reshape(part.shape).to(part.dtype)
            cur_out = (feat * q_w).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            better = err < min_errs
            min_errs[better] = err[better]
            best_max_val[better] = max_val[better]
        best_all.append(best_max_val)
    return torch.cat(best_all, dim=0).squeeze(1)


def apply_clip(w: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """``w`` ([N, K]) clamped group by group to ``[-max_val, max_val]`` (max_val [N, groups, 1])."""
    max_val = max_val.to(device=w.device, dtype=w.dtype)
    return torch.clamp(w.reshape(*max_val.shape[:2], -1), -max_val, max_val).reshape(w.shape)


__all__ = ["SKIPPED_NAMES", "apply_clip", "clip_search", "clip_skipped"]
