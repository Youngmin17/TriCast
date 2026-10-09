"""Hand-computed cases for app/metrics.py, and the run-list views of app/listing.py."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from app import listing
from app.metrics import (
    classify_metrics,
    detection_score,
    first_divergence,
    iou,
    match_boxes,
    pick_examples,
    teacher_forced,
)


def box(cls_id: int, conf: float, xyxy: tuple[float, float, float, float]) -> dict:
    return {"cls": f"class{cls_id}", "cls_id": cls_id, "conf": conf, "xyxy": list(xyxy)}


@pytest.mark.parametrize(("a", "b", "expected"), [
    ([1, 2, 3], [1, 2, 3], None),
    ([1, 2, 3], [1, 5, 3], 1),
    ([7], [8], 0),
    ([1, 2], [1, 2, 3], 2),
    ([], [], None),
])
def test_first_divergence(a: list[int], b: list[int], expected: int | None) -> None:
    assert first_divergence(a, b) == expected


def test_teacher_forced_identical_distributions_have_zero_kl() -> None:
    logits = torch.tensor([[0.0, 1.0, -2.0], [3.0, 3.0, 0.5]])
    result = teacher_forced(logits, logits.clone())
    assert result["kl"] == [0.0, 0.0] and result["kl_mean"] == 0.0 and result["kl_max"] == 0.0
    assert result["top1"] == [True, True] and result["top1_agreement"] == 1.0 and result["positions"] == 2


def test_teacher_forced_known_kl_and_top1() -> None:
    base = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float64))
    emu = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64))
    result = teacher_forced(base, emu)
    expected = [0.5 * math.log(25 / 9), 0.9 * math.log(0.9 / 0.2) + 0.1 * math.log(0.1 / 0.8)]
    assert result["kl"] == pytest.approx(expected, rel=1e-12)
    assert result["kl_mean"] == pytest.approx(sum(expected) / 2, rel=1e-12)
    assert result["kl_max"] == pytest.approx(max(expected), rel=1e-12)
    # Row 0 ties in the baseline: argmax takes the lowest index, as the emulated argmax does.
    assert result["top1"] == [True, False] and result["top1_agreement"] == 0.5


def test_teacher_forced_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        teacher_forced(torch.zeros(2, 3), torch.zeros(3, 3))
    with pytest.raises(ValueError):
        teacher_forced(torch.zeros(0, 3), torch.zeros(0, 3))


@pytest.mark.parametrize(("a", "b", "expected"), [
    ((0, 0, 2, 2), (0, 0, 2, 2), 1.0),
    ((0, 0, 2, 2), (3, 3, 4, 4), 0.0),
    ((0, 0, 2, 2), (1, 0, 3, 2), 1 / 3),
    ((1, 1, 1, 1), (1, 1, 1, 1), 0.0),
])
def test_iou(a: tuple, b: tuple, expected: float) -> None:
    assert iou(a, b) == pytest.approx(expected, abs=1e-15)


def test_match_requires_same_class() -> None:
    result = match_boxes([box(0, 0.9, (0, 0, 10, 10))], [box(1, 0.9, (0, 0, 10, 10))])
    assert result == {"matched": 0, "baseline_only": 1, "emulated_only": 1, "mean_iou": None,
                      "mean_abs_conf_delta": None, "pairs": []}


@pytest.mark.parametrize(("emulated", "matched"), [
    ((5, 0, 15, 10), 0),   # IoU 1/3
    ((0, 0, 10, 5), 1),    # IoU exactly 0.5 matches (>=)
    ((2, 0, 12, 10), 1),   # IoU 2/3
])
def test_match_iou_threshold(emulated: tuple, matched: int) -> None:
    result = match_boxes([box(3, 0.5, (0, 0, 10, 10))], [box(3, 0.5, emulated)])
    assert result["matched"] == matched


def test_match_is_greedy_by_baseline_confidence() -> None:
    # The more confident baseline box claims the only emulated box first, even though the other
    # baseline box overlaps it better.
    base = [box(0, 0.6, (0, 0, 10, 10)), box(0, 0.9, (1, 0, 11, 10))]
    result = match_boxes(base, [box(0, 0.8, (0, 0, 10, 10))])
    assert result["pairs"] == [{"baseline": 1, "emulated": 0, "iou": pytest.approx(90 / 110)}]
    assert (result["matched"], result["baseline_only"], result["emulated_only"]) == (1, 1, 0)
    assert result["mean_abs_conf_delta"] == pytest.approx(0.1)


def test_match_takes_highest_iou_candidate_and_averages_pairs() -> None:
    base = [box(0, 0.9, (0, 0, 10, 10)), box(2, 0.4, (20, 20, 30, 30))]
    emu = [box(0, 0.7, (2, 0, 12, 10)), box(0, 0.8, (1, 0, 11, 10)), box(2, 0.5, (20, 20, 30, 30))]
    result = match_boxes(base, emu)
    assert [(pair["baseline"], pair["emulated"]) for pair in result["pairs"]] == [(0, 1), (1, 2)]
    assert result["mean_iou"] == pytest.approx((90 / 110 + 1.0) / 2)
    assert result["mean_abs_conf_delta"] == pytest.approx((0.1 + 0.1) / 2)
    assert (result["matched"], result["baseline_only"], result["emulated_only"]) == (2, 0, 1)


def test_classify_metrics_known_values() -> None:
    result = classify_metrics(torch.tensor([0.6, 0.3, 0.1]), torch.tensor([0.2, 0.5, 0.3]))
    expected = 0.6 * math.log(3) + 0.3 * math.log(0.6) + 0.1 * math.log(1 / 3)
    assert result["kl"] == pytest.approx(expected, rel=1e-6)
    assert result["top1_same"] is False and result["top5_overlap"] == 3
    assert result["baseline_top1_p"] == pytest.approx(0.6) and result["emulated_top1_p"] == pytest.approx(0.5)


def test_classify_top5_overlap_with_stable_ties() -> None:
    base = torch.tensor([0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05], dtype=torch.float64)
    emu = torch.tensor([0.05, 0.1, 0.1, 0.1, 0.15, 0.2, 0.3], dtype=torch.float64)
    # top-5: baseline {0, 1, 2, 3, 4}; emulated {6, 5, 4, 1, 2} (ties keep the lower index)
    assert classify_metrics(base, emu)["top5_overlap"] == 3
    assert classify_metrics(base, base)["kl"] == 0.0


@pytest.mark.parametrize(("metrics", "expected"), [
    ({"baseline_only": 1, "emulated_only": 2, "mean_iou": 0.75}, 3.25),
    ({"baseline_only": 0, "emulated_only": 0, "mean_iou": None}, 0.0),
    ({"baseline_only": 2, "emulated_only": 0, "mean_iou": None}, 3.0),
    ({"baseline_only": 0, "emulated_only": 0, "mean_iou": 1.0}, 0.0),
])
def test_detection_score(metrics: dict, expected: float) -> None:
    assert detection_score(metrics) == pytest.approx(expected)


@pytest.mark.parametrize(("scores", "expected"), [
    ({10: 1.0, 11: 5.0, 12: 3.0, 13: 0.5, 14: 2.0}, [(11, "top1"), (12, "top2"), (14, "median")]),
    ({1: 2.0, 2: 2.0, 3: 0.0}, [(1, "top1"), (2, "top2")]),
    ({5: 1.0}, [(5, "top1")]),
    ({}, []),
])
def test_pick_examples(scores: dict, expected: list) -> None:
    assert pick_examples(scores) == expected


def test_listing_summary_keeps_the_chart_metrics() -> None:
    forced = {"top1_agreement": 0.75, "kl_mean": 0.1, "kl": [0.1], "top1": [True]}
    llm = {"first_divergence": 3, "prefix_match": 3, "teacher_forced": forced}
    assert listing.summary("llm.generate", llm) == {"first_divergence": 3, "prefix_match": 3,
                                                    "top1_agreement": 0.75, "kl_mean": 0.1}
    detect = match_boxes([box(0, 0.9, (0, 0, 10, 10))], [])
    assert listing.summary("vision.detect", detect) == {"matched": 0, "baseline_only": 1, "emulated_only": 0,
                                                        "mean_iou": None}
    classify = classify_metrics(torch.tensor([0.6, 0.4]), torch.tensor([0.5, 0.5]))
    assert set(listing.summary("vision.classify", classify)) == {"top1_same", "top5_overlap", "kl"}


def test_listing_preview_caps_texts_and_counts_outputs() -> None:
    text = "가" * 700
    preview = listing.preview("llm.generate", {"text": text}, {"text": "short"})
    assert preview == {"baseline": "가" * 600, "emulated": "short"}
    boxes = [box(0, 0.9, (0, 0, 1, 1))] * 3
    assert listing.preview("vision.detect", {"boxes": boxes}, {"boxes": boxes[:1]}) == {
        "baseline_boxes": 3, "emulated_boxes": 1}
    top = [{"label": "tabby", "class_id": 281, "p": 0.7}]
    other = [{"label": "tiger cat", "class_id": 282, "p": 0.4}]
    assert listing.preview("vision.classify", {"top": top}, {"top": other}) == {
        "baseline_top1": "tabby", "emulated_top1": "tiger cat", "emulated_top1_p": 0.4}


def test_listing_imports_without_torch_or_tricast() -> None:
    probe = "import sys, app.listing; print(sorted({'torch', 'tricast'} & set(sys.modules)))"
    output = subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=True).stdout
    assert output.strip() == "[]"
