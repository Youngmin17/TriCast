"""Short views of a finished run for run lists (demo index.json and ``GET /api/runs``): ``summary``
feeds the design-space charts, ``preview`` the results table (app/README.md, "데모 묶음").

Pure Python on the run's JSON fields, so the demo-only server can import it without torch or tricast.
"""

from __future__ import annotations

PREVIEW_CHARS = 600


def summary(task: str, metrics: dict) -> dict:
    """Chart metrics: llm first_divergence, prefix_match, top1_agreement, kl_mean; detect matched,
    baseline_only, emulated_only, mean_iou; classify top1_same, top5_overlap, kl."""
    if task == "llm.generate":
        forced = metrics["teacher_forced"]
        return {"first_divergence": metrics["first_divergence"], "prefix_match": metrics["prefix_match"],
                "top1_agreement": forced["top1_agreement"], "kl_mean": forced["kl_mean"]}
    keys = (("matched", "baseline_only", "emulated_only", "mean_iou") if task == "vision.detect"
            else ("top1_same", "top5_overlap", "kl"))
    return {key: metrics[key] for key in keys}


def preview(task: str, baseline: dict, emulated: dict) -> dict:
    """Outputs at a glance: llm texts (at most 600 characters each), detect box counts, classify top-1."""
    if task == "llm.generate":
        return {"baseline": baseline["text"][:PREVIEW_CHARS], "emulated": emulated["text"][:PREVIEW_CHARS]}
    if task == "vision.detect":
        return {"baseline_boxes": len(baseline["boxes"]), "emulated_boxes": len(emulated["boxes"])}
    return {"baseline_top1": baseline["top"][0]["label"], "emulated_top1": emulated["top"][0]["label"],
            "emulated_top1_p": emulated["top"][0]["p"]}
