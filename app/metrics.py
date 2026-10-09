"""Comparison metrics for TriCast Studio (app/README.md, "Metrics"): pure functions on runner outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def first_divergence(a_ids: Sequence[int], b_ids: Sequence[int]) -> int | None:
    """First position where two token sequences differ; ``None`` when they are identical.

    When one sequence is a strict prefix of the other they differ at the shorter length."""
    for position, (a, b) in enumerate(zip(a_ids, b_ids, strict=False)):
        if a != b:
            return position
    return None if len(a_ids) == len(b_ids) else min(len(a_ids), len(b_ids))


def teacher_forced(base_logits: torch.Tensor, emu_logits: torch.Tensor) -> dict:
    """Compare next-token distributions predicted over the same prefix at each position.

    Both inputs are ``[positions, vocab]`` logits for the prefixes of the baseline continuation, taken the
    way generation computes them (app/runners/llm.py): the baseline's own greedy-step logits, and the
    emulated model's step logits while its decoding is forced along that continuation. Row t therefore
    depends only on the prompt and the first t continuation tokens. ``kl`` is KL(baseline ‖ emulated) in
    nats per position, computed in float64 from log_softmax; ``top1`` compares argmax indices (lowest index
    on ties).
    """
    if base_logits.ndim != 2 or base_logits.shape != emu_logits.shape or base_logits.shape[0] == 0:
        raise ValueError("teacher_forced needs two non-empty [positions, vocab] tensors of one shape")
    base = torch.log_softmax(base_logits.double(), dim=-1)
    emu = torch.log_softmax(emu_logits.double(), dim=-1)
    kl = (base.exp() * (base - emu)).sum(dim=-1)
    top1 = base_logits.double().argmax(dim=-1) == emu_logits.double().argmax(dim=-1)
    return {"positions": int(kl.numel()), "top1_agreement": top1.double().mean().item(),
            "kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "kl": kl.tolist(), "top1": top1.tolist()}


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x1, y1, x2, y2]`` boxes; 0 when the union is empty."""
    inter = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(base_boxes: Sequence[dict], emu_boxes: Sequence[dict], iou_thr: float = 0.5) -> dict:
    """Pair detections of the same class, greedily.

    Baseline boxes are visited in descending confidence (ties: list order); each takes the still
    unpaired emulated box of its ``cls_id`` with the highest IoU, if that IoU is at least ``iou_thr``
    (ties: lower emulated index). ``pairs`` index into the two input lists.
    """
    free = list(range(len(emu_boxes)))
    pairs = []
    for i in sorted(range(len(base_boxes)), key=lambda index: -base_boxes[index]["conf"]):
        best, best_iou = None, 0.0
        for j in free:
            if emu_boxes[j]["cls_id"] != base_boxes[i]["cls_id"]:
                continue
            overlap = iou(base_boxes[i]["xyxy"], emu_boxes[j]["xyxy"])
            if overlap >= iou_thr and (best is None or overlap > best_iou):
                best, best_iou = j, overlap
        if best is not None:
            free.remove(best)
            pairs.append({"baseline": i, "emulated": best, "iou": best_iou})
    deltas = [abs(base_boxes[p["baseline"]]["conf"] - emu_boxes[p["emulated"]]["conf"]) for p in pairs]
    return {"matched": len(pairs), "baseline_only": len(base_boxes) - len(pairs),
            "emulated_only": len(emu_boxes) - len(pairs),
            "mean_iou": sum(p["iou"] for p in pairs) / len(pairs) if pairs else None,
            "mean_abs_conf_delta": sum(deltas) / len(deltas) if deltas else None, "pairs": pairs}


def classify_metrics(base_probs: torch.Tensor, emu_probs: torch.Tensor) -> dict:
    """Compare two ``[classes]`` probability vectors; KL(baseline ‖ emulated) in nats, float64."""
    base, emu = base_probs.double(), emu_probs.double()
    base_top = torch.argsort(base, descending=True, stable=True)[:5]
    emu_top = torch.argsort(emu, descending=True, stable=True)[:5]
    support = base > 0
    kl = (base[support] * (base[support].log() - emu[support].log())).sum().item()
    return {"top1_same": bool(base_top[0] == emu_top[0]),
            "top5_overlap": len(set(base_top.tolist()) & set(emu_top.tolist())), "kl": kl,
            "baseline_top1_p": base[base_top[0]].item(), "emulated_top1_p": emu[emu_top[0]].item()}


def detection_score(metrics: dict) -> float:
    """How different two detection results are: baseline_only + emulated_only + (1 − mean_iou).

    The IoU term is 1 when nothing matched and 0 when neither side has a box."""
    unmatched = metrics["baseline_only"] + metrics["emulated_only"]
    if metrics["mean_iou"] is None:
        return float(unmatched) + (1.0 if unmatched else 0.0)
    return unmatched + 1.0 - metrics["mean_iou"]


def pick_examples(scores: dict[int, float]) -> list[tuple[int, str]]:
    """The two highest scores (``top1``, ``top2``) and the lower median (``median``) of ``scores``.

    Ties go to the smaller key. The median is the ``(n - 1) // 2``-th item in ascending order and is
    left out when it is already one of the top two."""
    descending = sorted(scores, key=lambda key: (-scores[key], key))
    picks = list(zip(descending[:2], ("top1", "top2"), strict=False))
    if scores:
        median = sorted(scores, key=lambda key: (scores[key], key))[(len(scores) - 1) // 2]
        if median not in descending[:2]:
            picks.append((median, "median"))
    return picks
