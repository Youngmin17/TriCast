"""Every Quantizer option family on a tiny random Qwen3 (CPU, the check_qd.py config): the recipe comes from
app.quantizer_recipe.build with the option set, tricast patches (and calibrates when the recipe needs it, with
the recipe's own calibration settings on in-memory token ids), the forward pass is finite, and the option's
effect is observed against the same recipe without it: quantized weights, logits, observer / transform state
or the calibration report. "same" in the effect column is reported, not hidden.

usage: cd <repo> && PYTHONPATH=src:. python scripts/quantizer/qz_options_tiny.py   (torch + transformers, CPU;
ends with QZ_OPTIONS_TINY_OK)
"""

from __future__ import annotations

import torch
import transformers
from app import quantizer_recipe as qr

from tricast.calibration import calibrate
from tricast.nn import iter_emulinear
from tricast.nn.patch import patch_model
from tricast.recipe import load_recipe

CALIB = {"samples": 2, "seqlen": 16}  # in-memory windows from IDS (the tiny model has no dataset)
torch.manual_seed(42)
CFG = transformers.Qwen3Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
)
IDS = torch.randint(0, 128, (1, 48))


def run(
    method: str,
    options: dict | None = None,
    *,
    grid=None,
    kv: str = "none",
    kv_options=None,
    calibration=None,
) -> dict:
    data = qr.build(
        "tiny",
        method,
        grid or qr.uniform(method),
        kv,
        options={method: options or {}} if options else None,
        calibration=CALIB if calibration is None else calibration,
        kv_options=kv_options,
    )
    recipe = load_recipe(data)
    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(CFG).eval()
    patch_model(model, recipe)
    report = calibrate(model, recipe, input_ids=IDS) if recipe.needs_calibration else None
    torch.manual_seed(0)
    with torch.no_grad():
        logits = model(IDS).logits
    layers = dict(iter_emulinear(model))
    assert torch.isfinite(logits).all(), (method, options)
    return {
        "recipe": recipe,
        "data": data,
        "model": model,
        "logits": logits,
        "layers": layers,
        "report": report,
    }


def weights(result: dict, name: str = "model.layers.0.self_attn.q_proj") -> torch.Tensor:
    return result["layers"][name]._weight_operand.values.float()


def scale_calls(action) -> int:
    """How many candidate scales tricast's reference quantizer evaluates during ``action`` (MSE: one per
    ratio)."""
    import tricast.reference.quantize as ref

    calls = 0
    original = ref._absmax_scale

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    ref._absmax_scale = counting
    try:
        action()
    finally:
        ref._absmax_scale = original
    return calls


def probe(spec, other) -> str:
    """The two operand specs on a heavy-tailed tensor (where clipping can win), for options a tiny random
    weight does not move."""
    from tricast.quant.api import quantize

    x = torch.randn(8, 64, generator=torch.Generator().manual_seed(3))
    x[:, ::16] *= 30.0
    return differs(quantize(x, spec).dequantize(), quantize(x, other).dequantize())


def differs(a: torch.Tensor, b: torch.Tensor) -> str:
    return "same" if torch.equal(a, b) else f"differs (max |Δ| {float((a - b).abs().max()):.3g})"


ROWS: list[tuple[str, str, str]] = []


def row(family: str, setting: str, effect: str) -> None:
    ROWS.append((family, setting, effect))


def main() -> int:
    base = {m: run(m) for m in ("nvfp4", "int8", "gptq_int4", "smoothquant_nvfp4", "awq_nvfp4", "mxfp4")}
    q0 = "model.layers.0.self_attn.q_proj"

    # element rounding
    for method, option, value in (
        ("nvfp4", "weight.rounding", "rtz"),
        ("nvfp4", "weight.rounding", "sr"),
        ("mxfp4", "weight.rounding", "rtz"),
        ("gptq_int4", "weight.rounding", "rtz"),
    ):
        r = run(method, {option: value})
        assert r["layers"][q0].spec.weight.rounding.value == value
        effect = differs(weights(r), weights(base[method]))
        assert effect != "same", (method, option, value)
        row("원소 반올림", f"{method} {option}={value}", f"가중치 {effect}")
    r = run("nvfp4", {"activation.rounding": "rtz"})
    assert torch.equal(weights(r), weights(base["nvfp4"])) and not torch.equal(
        r["logits"], base["nvfp4"]["logits"]
    )
    row(
        "원소 반올림",
        "nvfp4 activation.rounding=rtz",
        "가중치 same · logits " + differs(r["logits"], base["nvfp4"]["logits"]),
    )

    # scale method
    for method, options in (
        ("nvfp4", {"weight.scale.method": "mse"}),
        ("nvfp4", {"weight.scale.method": "percentile", "weight.scale.percentile": 90.0}),
        ("int8", {"weight.scale.method": "percentile", "weight.scale.percentile": 99.0}),
        ("int8", {"weight.scale.method": "mse"}),
    ):
        r = run(method, options)
        assert r["layers"][q0].spec.weight.scale.method == options["weight.scale.method"]
        effect = f"가중치 {differs(weights(r), weights(base[method]))}"
        if effect.endswith("same"):  # MSE may keep ratio 1.0 (INT8's fine grid rarely gains from clipping)
            effect += " · 이상치 텐서 probe " + probe(
                r["layers"][q0].spec.weight, base[method]["layers"][q0].spec.weight
            )
        if effect.endswith("same"):  # then show that the search ran: candidate scales per requantize
            searched = scale_calls(r["layers"][q0].requantize)
            plain = scale_calls(base[method]["layers"][q0].requantize)
            assert searched == 20 * plain, (searched, plain)
            effect += f" · 후보 스케일 {searched}개 평가 (absmax {plain}개) → 비율 1.0 이 최적"
        row("스케일 고르기", f"{method} {options}", effect)
    for method in ("nvfp4", "int8"):
        r = run(method, {"activation.scale.method": "mse"})
        assert r["layers"][q0].spec.activation.scale.method == "mse"
        assert torch.equal(weights(r), weights(base[method]))
        effect = "logits " + differs(r["logits"], base[method]["logits"])
        if effect.endswith("same"):
            with torch.no_grad():
                searched = scale_calls(lambda model=r["model"]: model(IDS))
                plain = scale_calls(lambda model=base[method]["model"]: model(IDS))
            assert searched == 20 * plain, (searched, plain)
            effect += f" · 순전파 한 번에 후보 스케일 {searched}개 (absmax {plain}개) → 비율 1.0 이 최적"
        row("스케일 고르기", f"{method} activation.scale.method=mse", f"가중치 same · {effect}")

    # group size and GPTQ solver
    r = run("gptq_int4", {"weight.group_size": 32})
    assert r["layers"][q0].spec.weight.group_size == 32 and r["report"] is not None
    effect = differs(weights(r), weights(base["gptq_int4"]))
    assert effect != "same"
    row("그룹 크기", "gptq_int4 weight.group_size=32 (K=64: 2 그룹 vs 1)", f"가중치 {effect}")
    for options in (
        {"weight_algo.damp": 0.5},
        {"weight_algo.act_order": True},
        {"weight_algo.block_size": 32},
    ):
        r = run("gptq_int4", options)
        ((option, value),) = options.items()
        assert getattr(r["layers"][q0].spec.weight_algo, option.split(".")[1]) == value
        row(
            "GPTQ", f"gptq_int4 {option}={value}", f"가중치 {differs(weights(r), weights(base['gptq_int4']))}"
        )

    # activation granularity and observers (int8)
    r = run("int8", {"activation.granularity": "tensor"})
    assert r["layers"][q0].spec.activation.granularity == "tensor" and r["report"] is None
    row(
        "활성 스케일 단위",
        "int8 activation.granularity=tensor",
        "logits " + differs(r["logits"], base["int8"]["logits"]),
    )
    amax = {}
    for kind, extra in (
        ("minmax", {}),
        ("ema", {"activation.observer.decay": 0.5}),
        ("percentile", {"activation.observer.percentile": 99.0}),
    ):
        r = run("int8", {"activation.granularity": "tensor", "activation.observer.kind": kind, **extra})
        layer = r["layers"][q0]
        assert r["recipe"].needs_calibration and r["report"] is not None and layer.observer is not None
        assert (
            layer.observer.spec.kind == kind
            and layer.observer.count > 0
            and layer.observer.static_amax is not None
        )
        if kind == "ema":
            assert layer.observer.spec.decay == 0.5
        amax[kind] = float(layer.observer.static_amax)
        row(
            "observer",
            f"int8 tensor + {kind} {extra or ''}".strip(),
            f"보정 {r['report']['samples']}창 · observer count {layer.observer.count} · "
            f"static amax {amax[kind]:.4f}",
        )
    assert amax["ema"] != amax["minmax"] and amax["percentile"] < amax["minmax"], amax
    r = run(
        "int8",
        {
            "activation.granularity": "tensor",
            "activation.observer.kind": "history",
            "activation.observer.history_len": 2,
        },
        calibration={},
    )
    layer = r["layers"][q0]
    assert not r["recipe"].needs_calibration and r["report"] is None and "calibration" not in r["data"]
    with torch.no_grad():
        for _ in range(3):
            r["model"](IDS)
    assert (
        layer.observer.spec.history_len == 2
        and len(layer.observer.history) == 2
        and layer.observer.count == 4
    )
    row(
        "observer",
        "int8 tensor + history history_len=2 (보정 없음)",
        f"순전파 {layer.observer.count}회 후 history {len(layer.observer.history)}개 (상한 2)",
    )

    # transforms
    r = run("smoothquant_nvfp4", {"transform.alpha": 0.8})
    assert r["layers"][q0].spec.transform.alpha == 0.8
    effect = differs(r["layers"][q0].transform.diag, base["smoothquant_nvfp4"]["layers"][q0].transform.diag)
    assert effect != "same"
    row("SmoothQuant", "alpha=0.8 (기본 0.5)", f"스케일 벡터 {effect}")
    r = run("awq_nvfp4", {"transform.grid": 2})
    assert r["layers"][q0].spec.transform.grid == 2
    row(
        "AWQ",
        "grid=2 (기본 20)",
        "스케일 벡터 "
        + differs(r["layers"][q0].transform.diag, base["awq_nvfp4"]["layers"][q0].transform.diag),
    )
    r = run("awq_nvfp4", {"transform.share_inputs": False})
    shared, alone = (
        [g for g in result["report"]["groups"] if len(g) > 1] for result in (base["awq_nvfp4"], r)
    )
    diag = {k: r["layers"][f"model.layers.0.self_attn.{k}"].transform.diag for k in ("q_proj", "k_proj")}
    base_diag = {
        k: base["awq_nvfp4"]["layers"][f"model.layers.0.self_attn.{k}"].transform.diag
        for k in ("q_proj", "k_proj")
    }
    assert shared and not alone and torch.equal(base_diag["q_proj"], base_diag["k_proj"])
    assert not torch.equal(diag["q_proj"], diag["k_proj"])
    row(
        "AWQ", "share_inputs=false", f"공유 그룹 {len(shared)}개 → {len(alone)}개 · q/k 스케일 벡터 서로 다름"
    )

    # a recipe mixing variants: plain layers must not inherit the default's transform options
    grid = qr.uniform("awq_nvfp4")
    for b in range(qr.BLOCKS):
        grid[b][qr.KINDS.index("o_proj")] = "nvfp4"
        grid[b][qr.KINDS.index("down_proj")] = "smoothquant_nvfp4"
    options = {"awq_nvfp4": {"transform.grid": 2}, "smoothquant_nvfp4": {"transform.alpha": 0.8}}
    data = qr.build("tiny", "awq_nvfp4", grid, options=options, calibration=CALIB)
    recipe = load_recipe(data)
    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(CFG).eval()
    patch_model(model, recipe)
    calibrate(model, recipe, input_ids=IDS)
    for name, layer in iter_emulinear(model):
        method = (
            next(m for m, k in (("nvfp4", "o_proj"), ("smoothquant_nvfp4", "down_proj")) if name.endswith(k))
            if name.endswith(("o_proj", "down_proj"))
            else "awq_nvfp4"
        )
        assert (
            layer.spec
            == load_recipe({"name": "a", "defaults": qr.variant(method, options.get(method))}).defaults
        )
    row(
        "섞인 레시피",
        "기본 AWQ grid=2 · o_proj NVFP4 · down_proj SQ α=0.8",
        f"패치된 {len(list(iter_emulinear(model)))}개 Linear 의 spec = 각 방법 단독 spec "
        "(transform 값 새지 않음)",
    )

    # KV residual
    k_base = run("nvfp4", kv="kivi2")
    r = run("nvfp4", kv="kivi2", kv_options={"residual": 32})
    assert r["recipe"].kv.residual == 32 and k_base["recipe"].kv.residual == 128
    effect = differs(r["logits"], k_base["logits"])
    assert effect != "same"
    row("KV residual", "kivi2 residual=32 (기본 128, 48 토큰)", f"logits {effect}")

    # calibration settings
    settings = {"samples": 3, "seqlen": 8, "seed": 7, "sequential": True}
    r = run("gptq_int4", calibration=settings)
    report = r["report"]
    assert r["data"]["calibration"] == {"samples": 3, "seqlen": 8, "seed": 7, "sequential": True}
    assert (report["samples"], report["n_tokens"], report["seed"], report["sequential"]) == (3, 24, 7, True)
    base_report = base["gptq_int4"]["report"]
    row(
        "보정 설정",
        str(settings),
        f"보정 {report['samples']}창 · {report['n_tokens']} 토큰 · mode {report['mode']} "
        f"(기본 2창 · {base_report['n_tokens']} 토큰 · mode {base_report['mode']}) · 가중치 "
        + differs(weights(r), weights(base["gptq_int4"])),
    )

    width = max(len(f) for f, _, _ in ROWS)
    for family, setting, effect in ROWS:
        print(f"{family:<{width}}  {setting}  ->  {effect}")
    print(f"QZ_OPTIONS_TINY_OK ({len(ROWS)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
