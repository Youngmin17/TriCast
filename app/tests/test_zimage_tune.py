"""CPU checks of scripts/quantizer/zimage_tune.py's pure parts and of the exported Z-Image-Turbo dataset.

The pipeline itself (refs / table / evaluate) runs on a cluster GPU; here: bits and bytes accounting, the
common rule's layers, the knapsack (integer program) against brute force on tiny synthetic tables, the image
metrics on hand-made images, each method's exact FP32 dequantization, the paired verdict (sign-flip, Holm),
the validator on a synthetic dataset and on the shipped one.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "zimage_tune", ROOT / "scripts" / "quantizer" / "zimage_tune.py"
)
ZT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ZT  # its dataclass looks the module up
SPEC.loader.exec_module(ZT)
EXPORT = ROOT / "app" / "web" / "demo" / "autotune" / "zimage.json"


def test_stored_bits_and_bytes() -> None:
    assert ZT.stored_bits("bf16", 3840) == 16  # the weight as it is (the checkpoint's values are BF16)
    # INT8: one fp32 scale per output row of K inputs
    assert ZT.stored_bits("int8", 3840) == 8 + Fraction(32, 3840)
    for method, bits in (("int4", 4), ("int3", 3), ("int2", 2)):
        # b-bit codes + an fp16 scale and a b-bit zero point per 128 inputs
        assert ZT.stored_bits(method, 3840) == bits + Fraction(16 + bits, 128)
    layers = [ZT.Layer("a", 3840 * 256, 256, 3840), ZT.Layer("b", 3840 * 3840, 3840, 3840)]
    assignment = {"a": "int8", "b": "int4"}
    total = 3840 * 256 * (8 + Fraction(32, 256)) + 3840 * 3840 * (4 + Fraction(20, 128))
    assert ZT.total_bits(layers, assignment) == total
    assert ZT.model_bytes(layers, assignment, 1000) == math.ceil((total + 16 * 1000) / 8)
    assert ZT.original_bytes(layers, 1000) == 2 * (3840 * 256 + 3840 * 3840 + 1000)


def test_the_common_rule_keeps_embedders_adaln_and_the_final_layer_at_int8() -> None:
    names = [
        "t_embedder.mlp.0",
        "cap_embedder.1",
        "layers.0.adaLN_modulation.0",
        "layers.0.attention.to_q",
        "layers.0.feed_forward.w2",
        "all_final_layer.2-1.linear",
        "all_final_layer.2-1.adaLN_modulation.1",
    ]
    layers = [ZT.Layer(name, 128 * 128, 128, 128) for name in names]
    rule = ZT.baseline_assignment(ZT.HEURISTIC, layers)
    assert [name for name, method in rule.items() if method == "int8"] == [
        "t_embedder.mlp.0",
        "cap_embedder.1",
        "layers.0.adaLN_modulation.0",
        "all_final_layer.2-1.linear",
        "all_final_layer.2-1.adaLN_modulation.1",
    ]
    assert {rule[n] for n in ("layers.0.attention.to_q", "layers.0.feed_forward.w2")} == {"int4"}
    assert set(ZT.baseline_assignment("u_int3", layers).values()) == {"int3"}
    assert set(ZT.baseline_assignment("native", layers).values()) == {"bf16"}


def test_the_knapsack_finds_the_brute_force_minimum() -> None:
    pytest.importorskip("scipy")
    rng = random.Random(42)
    for _ in range(20):
        outs = [rng.choice((1, 2, 3)) for _ in range(4)]
        layers = [ZT.Layer(f"l{i}", 128 * out, 128, out) for i, out in enumerate(outs)]
        table = {}
        for layer in layers:
            d = sorted(rng.random() for _ in ZT.QUANT_METHODS)  # fewer bits, larger D
            table[layer.name] = dict(zip(ZT.QUANT_METHODS, d, strict=True))
        params = sum(layer.params for layer in layers)
        budget = ZT.avg_bits(layers, {layer.name: "int4" for layer in layers}) * params
        best = min(
            (
                ZT.predicted_d(dict(zip([x.name for x in layers], choice, strict=True)), table)
                for choice in itertools.product(ZT.METHODS, repeat=len(layers))
                if ZT.total_bits(layers, dict(zip([x.name for x in layers], choice, strict=True))) <= budget
            ),
        )
        assignment, solution = ZT.solve(layers, table, budget)
        assert ZT.total_bits(layers, assignment) <= budget
        assert ZT.predicted_d(assignment, table) == pytest.approx(best, abs=1e-12)
        assert solution.value == pytest.approx(best, abs=1e-12)
        assert solution.cost == ZT.total_bits(layers, assignment) and solution.gap <= 1e-9


def test_the_knapsack_takes_a_real_sized_budget() -> None:
    """275 layers of the transformer's shapes: the exact DP of app.autotune would need ~5e10 cells here."""
    pytest.importorskip("scipy")
    rng = random.Random(42)
    shapes = [(3840, 3840)] * 120 + [(10240, 3840)] * 60 + [(3840, 10240)] * 30 + [(15360, 256)] * 30
    shapes += [(3840, 3840)] * 30 + [(64, 3840), (1024, 256), (3840, 1024), (3840, 2560), (3840, 256)]
    layers = [ZT.Layer(f"l{i}", out * k, k, out) for i, (out, k) in enumerate(shapes)]
    table = {
        x.name: dict(zip(ZT.QUANT_METHODS, sorted(rng.random() for _ in range(4)), strict=True))
        for x in layers
    }
    params = sum(x.params for x in layers)
    budget = ZT.avg_bits(layers, {x.name: "int4" for x in layers}) * params
    assignment, solution = ZT.solve(layers, table, budget)
    assert ZT.total_bits(layers, assignment) <= budget and solution.gap <= 1e-9 * max(1.0, solution.value)
    # no worse than the uniform assignment it is compared with
    assert solution.value <= ZT.predicted_d({x.name: "int4" for x in layers}, table) + 1e-12


def test_image_metrics_on_hand_made_images() -> None:
    a = np.full((32, 32, 3), 100, dtype=np.uint8)
    assert ZT.psnr(a, a) is None  # equal images: no finite PSNR (JSON has no infinity)
    assert ZT.ssim(a, a) == pytest.approx(1.0)
    b = a + 1
    assert ZT.psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 1.0))
    noise = np.random.default_rng(42).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    assert ZT.ssim(a, noise) < 0.1
    assert ZT.ssim(noise, noise) == pytest.approx(1.0)


def test_the_sign_flip_p_is_the_share_of_sign_patterns_at_least_as_extreme() -> None:
    rng = random.Random(42)
    for n in (1, 3, 6, 9):
        diffs = [rng.uniform(-1, 1) + 0.3 for _ in range(n)]
        observed = abs(sum(diffs))
        brute = (
            sum(
                abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12
                for signs in itertools.product((1, -1), repeat=n)
            )
            / 2**n
        )
        assert ZT.sign_flip_p(diffs) == pytest.approx(brute)
    assert ZT.sign_flip_p([1.0] * 10) == pytest.approx(2 / 2**10)  # only all-plus and all-minus reach it
    assert ZT.sign_flip_p([0.0] * 4) == 1.0


def test_holm_adjusts_in_the_input_order() -> None:
    assert ZT.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ZT.holm([0.5]) == [0.5]
    assert ZT.verdict(0.01, 0.002) == "ours" and ZT.verdict(0.01, -0.002) == "theirs"
    assert ZT.verdict(0.2, 0.5) == "same"


def test_d_is_summed_in_float64_and_must_be_finite() -> None:
    big = [torch.tensor([2e20], dtype=torch.float32)]
    assert ZT.relative_mse(big, [torch.tensor([1e20], dtype=torch.float32)]) == pytest.approx(1.0)
    with pytest.raises(ZT.TuneError):
        ZT.relative_mse([torch.ones(3)], [torch.zeros(3)])  # no reference energy


@pytest.mark.parametrize("method", ["int8", "int4", "int3", "int2"])
def test_each_method_dequantizes_to_fp32_exactly(method: str) -> None:
    """The fake-quantized weight is (q - z) * s computed in FP32 — not rounded onto BF16 on the way."""
    pytest.importorskip("tricast")
    from tricast.quant.api import quantize
    from tricast.reference.quantize import expand_scale

    weight = torch.randn(4, 256, generator=torch.Generator().manual_seed(42))
    spec = ZT.weight_spec(method)
    qt = quantize(weight, spec, backend="reference")
    scale = qt.scale_per_element().double()
    zero = 0.0 if qt.zero_point is None else expand_scale(qt.zero_point, spec, qt.rows, qt.K).double()
    expected = ((qt.values.double() - zero) * scale).float()
    assert torch.equal(ZT.fake_quant_weight(weight, method), expected)
    if method != "int8":  # the codes x fp16 scales are not BF16 values: a BF16 detour would change them
        assert not torch.equal(expected, expected.bfloat16().float())


@pytest.mark.parametrize(("method", "levels"), [("int8", 255), ("int4", 16), ("int3", 8), ("int2", 4)])
def test_each_method_puts_a_group_on_its_integer_grid(method: str, levels: int) -> None:
    pytest.importorskip("tricast")
    weight = torch.randn(4, 256, generator=torch.Generator().manual_seed(42))
    quantized = ZT.fake_quant_weight(weight, method)
    assert quantized.shape == weight.shape and quantized.dtype == weight.dtype
    span = 256 if method == "int8" else 128  # a row's scale, or one per 128 inputs
    for row in quantized:
        for group in row.split(span):
            assert len(torch.unique(group)) <= levels
    assert not torch.equal(quantized, weight)


def _dataset(prompts: int = 6) -> dict:
    """A small dataset with the export's structure: the common rule, uniform and Auto-tune setups."""
    rng = random.Random(42)
    names = [
        "t_embedder.mlp.0",
        "layers.0.adaLN_modulation.0",
        "layers.0.attention.to_q",
        "layers.0.feed_forward.w1",
    ]
    layers = [ZT.Layer(name, 128 * 256, 128, 256) for name in names]
    table = {
        x.name: dict(zip(ZT.QUANT_METHODS, sorted(rng.random() for _ in range(4)), strict=True))
        for x in layers
    }
    params = sum(x.params for x in layers)
    setups = []
    for setup in ZT.SETUPS[1:]:
        if setup in ZT.PAIR:
            budget = ZT.auto_budget(setup, layers) * params
            methods, _ = ZT.solve(layers, table, budget)
        else:
            methods = ZT.baseline_assignment(setup, layers)
        bits = ZT.avg_bits(layers, methods)
        ssim = [rng.uniform(0.6, 0.99) for _ in range(prompts)]
        psnr = [rng.uniform(15, 35) for _ in range(prompts)]
        entry = {
            "id": setup,
            "kind": "auto" if setup in ZT.PAIR else "uniform" if setup in ZT.UNIFORM else "heuristic",
            "bits_fraction": ZT.fraction_text(bits),
            "bytes": ZT.model_bytes(layers, methods, 100),
            "methods": methods,
            "predicted_d": ZT.predicted_d(methods, table),
            "measured_d": 0.01,
            "metrics": {"ssim": sum(ssim) / prompts, "psnr": sum(psnr) / prompts, "identical": 0},
            "per_prompt": {"ssim": ssim, "psnr": psnr},
        }
        if setup in ZT.PAIR:
            entry["budget_bits"] = ZT.fraction_text(budget / params)
        setups.append(entry)
    ZT.compare(setups)
    return {
        "schema": ZT.SCHEMA,
        "model": {
            "linear_layers": len(layers),
            "other_params": 100,
            "original_bytes": ZT.original_bytes(layers, 100),
        },
        "evaluation": {"prompts": prompts},
        "layers": [
            {"name": x.name, "params": x.params, "k": x.k, "out": x.out, "d": table[x.name]} for x in layers
        ],
        "setups": setups,
        "native": {"metrics": {"ssim": 1.0, "psnr": None}},
        "images": [{"files": {s: f"{s}.jpg" for s in ZT.SETUPS}}],
    }


def test_the_validator_accepts_a_consistent_dataset_and_catches_each_break() -> None:
    pytest.importorskip("scipy")
    data = _dataset()
    assert ZT.validate(data) == []
    paired = {s["id"]: s["paired"] for s in data["setups"] if s["kind"] == "auto"}
    assert {i: p["vs"] for i, p in paired.items()} == ZT.PAIR  # fixed before the setups were measured
    for break_it, message in (
        (lambda d: d["setups"][0].__setitem__("measured_d", float("nan")), "measured_d"),
        (lambda d: d["native"].pop("metrics"), "native metrics"),
        (lambda d: d["setups"][1]["metrics"].__setitem__("ssim", 0.5), "mean SSIM"),
        (
            lambda d: d["setups"][-1]["paired"].__setitem__(
                "verdict", "ours" if d["setups"][-1]["paired"]["verdict"] != "ours" else "same"
            ),
            "verdict",
        ),
        (lambda d: d["setups"][2]["per_prompt"]["ssim"].pop(), "per-prompt"),
        (lambda d: d["setups"][1]["metrics"].__setitem__("ssim", float("nan")), "mean SSIM"),
        (lambda d: d["setups"][-1]["paired"].__setitem__("mean_diff", 0.5), "do not follow"),
    ):
        broken = json.loads(json.dumps(data))
        break_it(broken)
        assert any(message in problem for problem in ZT.validate(broken)), message


def test_holm_runs_within_each_family_of_sizes() -> None:
    pytest.importorskip("scipy")
    paired = {s["id"]: s["paired"] for s in _dataset()["setups"] if s["kind"] == "auto"}
    for family in ZT.FAMILIES:
        raw = [paired[i]["p_sign_flip"] for i in family]
        assert [paired[i]["p_holm"] for i in family] == pytest.approx(ZT.holm(raw))
        assert all(paired[i]["family"] == list(family) for i in family)
    # the between-step sizes were added after the first results, and say so
    assert all(paired[i]["fixed"].startswith("before") for i in ZT.AUTO)
    assert not any(paired[i]["fixed"].startswith("before") for i in ZT.BETWEEN)


def test_holm_corrects_within_a_family_only() -> None:
    # every Auto-tune setup ahead of its pair on all 6 prompts: the same exact p, 2 / 2^6, for all five
    n = 6
    setups = [
        {"id": s, "kind": "uniform" if s in ZT.UNIFORM else "heuristic", "per_prompt": {"ssim": [0.5] * n}}
        for s in (*ZT.UNIFORM, ZT.HEURISTIC)
    ]
    setups += [{"id": a, "kind": "auto", "per_prompt": {"ssim": [0.6] * n}} for a in ZT.PAIR]
    ZT.compare(setups)
    paired = {s["id"]: s["paired"] for s in setups if s["kind"] == "auto"}
    one = 2 / 2**n
    assert all(paired[a]["p_sign_flip"] == pytest.approx(one) for a in ZT.PAIR)
    # Holm over 3 and over 2 — not over all 5 (that would be 5 * one)
    assert [paired[a]["p_holm"] for a in ZT.AUTO] == pytest.approx([3 * one] * 3)
    assert [paired[a]["p_holm"] for a in ZT.BETWEEN] == pytest.approx([2 * one] * 2)


def test_duplicate_setups_are_refused() -> None:
    pytest.importorskip("scipy")
    data = _dataset()
    data["setups"].append(json.loads(json.dumps(data["setups"][-1])))
    assert "duplicate setup ids" in ZT.validate(data)


def test_a_between_step_budget_leaves_its_pair_the_only_existing_way(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("scipy")
    assert ZT.validate(_dataset()) == []
    # a budget above INT3 uniform (3 + 19/128 bits) lets INT3 uniform fit too, so INT2 is not the only one
    monkeypatch.setitem(ZT.BETWEEN, "auto_b300", (Fraction(13, 4), "u_int2"))
    assert any("only existing way" in problem for problem in ZT.validate(_dataset()))


@pytest.mark.skipif(not EXPORT.exists(), reason="the Z-Image-Turbo dataset is not exported yet")
def test_the_exported_dataset_is_consistent_and_its_images_exist() -> None:
    data = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert ZT.validate(data) == []
    for image in data["images"]:
        for path in image["files"].values():
            assert (ROOT / "app" / "web" / "demo" / path).is_file(), path
    autos = [s for s in data["setups"] if s["kind"] == "auto"]
    assert autos and all(s["bits"] <= float(Fraction(s["budget_bits"])) for s in autos)


def test_the_fingerprint_changes_with_everything_a_table_row_depends_on() -> None:
    pytest.importorskip("tricast")
    info = {"replay_sha256": "a" * 64, "embeds_sha256": "b" * 64, "dtype": "fp32", "size": 512}
    base = ZT.fingerprint(info, 8)
    assert base == ZT.fingerprint(dict(info), 8)
    others = [
        ZT.fingerprint(info, 4),
        ZT.fingerprint({**info, "size": 1024}, 8),
        ZT.fingerprint({**info, "dtype": "fp16"}, 8),
    ]
    others.append(ZT.fingerprint({**info, "replay_sha256": "c" * 64}, 8))
    assert len({base["sha256"], *(x["sha256"] for x in others)}) == 5
    assert (
        "dequant_format=FloatFormat(name='fp32'" in base["methods"]["int4"]
        or "fp32" in base["methods"]["int4"]
    )
