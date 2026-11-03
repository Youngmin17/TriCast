"""Low-bit per-layer methods (track M: 2-4 bit weight-only; track C: W4A4 with a rotation) vs official code.

GPTQ 3 / 2-bit = weight_algo gptq on the gptq_int4 weight spec with its element format replaced (uint3 /
uint2, group 128, FP16 scale, integer zero point), against the pinned official IST-DASLab/gptq 2d65066
(``GPTQ`` + ``quant.Quantizer``, asymmetric as llama.py's default) on one random Linear on CPU, the official
``add_batch`` Hessian given to both. TriCast is not bit-identical to the unmodified official code; the
causes, read from the two sources:

1. scale storage: ScaleSpec(FP16) rounds each group scale to fp16 and takes the zero point from the rounded
   scale (reference/quantize.py ``_zero_point_scale``); ``Quantizer.find_params`` keeps
   ``(xmax - xmin) / maxq`` in fp32.
2. arithmetic: TriCast factorizes H and propagates the errors in fp64 (weight_quant/gptq.py); ``fasterquant``
   runs in fp32 (``W = W.float()``, the fp32 ``add_batch`` Hessian).
3. TriCast reads the compensated fp64 weights as fp32 for the group parameters and the element rounding
   (``compute_scale`` / ``quantize_elements`` call ``.float()``).
4. groups that start inside a lazy block (group_size < block_size, no act_order): ``fasterquant`` refreshes
   them from ``W[:, i1 + i:...]``, which gets the block's updates only after the block
   (``W[:, i2:] -= ...``); TriCast refreshes from the compensated block.
5. act_order: TriCast fixes the groups from the original weights before permuting, which is the official
   ``actorder=True, static_groups=True``.

The relation tested exactly: official ``fasterquant`` with 1-3 made the same (a ``Quantizer`` subclass rounds
the scale to fp16 where ``find_params`` sets it and recomputes the zero with the official formula;
``Tensor.float`` keeps fp64 tensors fp64 during the call and H is passed in fp64; ``find_params`` /
``quantize`` get fp32-rounded inputs) equals TriCast bit for bit in codes, scales, zero points and dequantized
weights, with official ``static_groups`` = TriCast ``act_order`` (5) and, for 4, the official block size = the
group size. The unmodified official code differs (scale mismatches > 0: 1 is real).

HQQ 3 / 2-bit (weight_algo hqq3 / hqq2, official_adapter): the official HQQLinear call bit for bit, the solver
ran. AWQ INT4 (weight-only) and Hadamard + NVFP4 / MXFP4 (W4A4): TriCast compositions without official-code
parity (tricast_native): what each must do is tested on single EmuLinears (no calibration.py driver).
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch

from tricast.mma.api import as_operand, gemm
from tricast.nn.linear import EmuLinear
from tricast.quant import api as quant_api
from tricast.quant.api import fake_quant, quantize
from tricast.quant.spec import WeightAlgoSpec, get_scheme
from tricast.recipe import WEIGHT_METHODS, WeightMethodSpec, load_recipe
from tricast.transforms import LinearTransform, StatsCollector, fit_transform
from tricast.weight_quant import quantize_weight, replace_weight

ROWS, COLS, TOKENS = 64, 256, 1024  # K = 256: two groups of 128; Accelerate's LAPACK is not run-to-run
#                                     deterministic at K = 512 on CPU (seen while writing this test)


def gptq_low(bits: int, **weight: object) -> dict:
    """The recipe fields of the low-bit GPTQ method (gptq_int4's spec with ``format: uint<bits>``)."""
    return {"weight": {"scheme": "int4_g128_zp", "format": f"uint{bits}", **weight}, "activation": None,
            "transform": "none", "weight_algo": "gptq"}


def _q(scheme: str) -> dict:
    return {"scheme": scheme, "mma_input": "dequant"}


HAD = {f: {"weight": _q(f), "activation": _q(f), "transform": {"kind": "hadamard"}, "weight_algo": "rtn"}
       for f in ("nvfp4", "mxfp4")}
PLAIN = {f: {**fields, "transform": "none"} for f, fields in HAD.items()}
AWQ_INT4 = {"weight": "int4_g128_zp", "activation": None, "transform": {"kind": "awq", "share_inputs": True},
            "weight_algo": "rtn"}


def spec_of(fields: dict):
    return load_recipe({"name": "t", "defaults": fields}).defaults


# ---------- GPTQ 3 / 2-bit vs the official code ----------
@pytest.fixture
def official(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")  # official gptq.py imports it
    from tricast.official import repos

    if not torch.cuda.is_available():  # official fasterquant calls torch.cuda.synchronize(): no-op on CPU
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    with contextlib.redirect_stdout(io.StringIO()):  # quant.py prints when its CUDA extension is absent
        return repos.load_module("gptq", "gptq"), repos.load_module("gptq", "quant"), monkeypatch


def _layer(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(ROWS, COLS, generator=g) * 0.02, torch.randn(TOKENS, COLS, generator=g)


def _official_hessian(module, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    layer = torch.nn.Linear(COLS, ROWS, bias=False)
    layer.weight.data = W.clone()
    probe = module.GPTQ(layer)
    probe.add_batch(X, None)
    H = probe.H
    # mirrored to exact symmetry, which TriCast's gptq requires (an sgemm need not return a symmetric X^T X)
    return torch.triu(H) + torch.triu(H, 1).T


def run_official(official, W: torch.Tensor, H: torch.Tensor, bits: int, *, group: int, block: int,
                 act_order: bool, damp: float, instrumented: bool) -> dict:
    """Official ``fasterquant`` on ``W`` with Hessian ``H``: the dequantized weight and every group's (scale,
    zero) as ``find_params`` set them (instrumented: causes 1-3 of the module docstring made TriCast's)."""
    gptq_module, quant, monkeypatch = official
    found, seen = [], []

    class Recording(quant.Quantizer):
        def find_params(self, x, weight=False):
            seen.append(x.dtype)
            if instrumented:
                x = x.to(torch.float32)  # cause 3 (not .float(): lifted below)
            super().find_params(x, weight)
            if instrumented:  # cause 1: the fp16 scale, the zero point from it by the official formula
                xmin = torch.minimum(x.min(1)[0], torch.zeros(x.shape[0], dtype=x.dtype))
                self.scale = self.scale.to(torch.float16).to(torch.float32)
                self.zero = torch.round(-xmin / self.scale.flatten()).reshape(self.zero.shape)
            found.append((self.scale.flatten().clone(), self.zero.flatten().clone()))

    layer = torch.nn.Linear(COLS, ROWS, bias=False)
    layer.weight.data = W.double() if instrumented else W.clone()
    solver = gptq_module.GPTQ(layer)
    solver.H = H.double() if instrumented else H.clone()  # cause 2: H in fp64
    solver.quantizer = Recording()
    solver.quantizer.configure(bits, perchannel=True, sym=False, mse=False)
    with monkeypatch.context() as patch:
        if instrumented:
            original_float, original_quantize = torch.Tensor.float, gptq_module.quantize
            patch.setattr(torch.Tensor, "float", lambda self, *args, **kwargs: self
                          if self.dtype == torch.float64 else original_float(self, *args, **kwargs))
            patch.setattr(gptq_module, "quantize", lambda x, scale, zero, maxq: original_quantize(
                x.to(torch.float32), scale, zero, maxq).to(x.dtype))
        solver.fasterquant(blocksize=block, percdamp=damp, groupsize=group, actorder=act_order,
                           static_groups=act_order)
    groups = -(-COLS // group)
    # find_params runs once on the whole weight first (quantizer not ready); static groups follow it, dynamic
    # groups are the last ones
    found = found[1:1 + groups] if act_order else found[-groups:]
    scale = torch.stack([s for s, _ in found], 1).float()
    zero = torch.stack([z for _, z in found], 1).float()
    weight = layer.weight.data.float()
    codes = torch.round(weight / scale.repeat_interleave(group, 1)) + zero.repeat_interleave(group, 1)
    return {"codes": codes, "scale": scale, "zero": zero, "weight": weight, "dtypes": set(seen)}


def run_tricast(W: torch.Tensor, H: torch.Tensor, bits: int, *, group: int, block: int, act_order: bool,
                damp: float) -> dict:
    spec = spec_of(gptq_low(bits, group_size=group))
    algo = WeightAlgoSpec("gptq", block_size=block, damp=damp, act_order=act_order)
    q = quantize_weight(W, spec.weight, algo, hessian=H)
    return {"codes": q.values.float(), "scale": q.scale.float(), "zero": q.zero_point.float(),
            "weight": q.dequantize()}


def mismatches(ours: dict, theirs: dict) -> dict[str, int]:
    return {key: int((ours[key] != theirs[key]).sum()) for key in ("codes", "scale", "zero", "weight")}


NONE = {"codes": 0, "scale": 0, "zero": 0, "weight": 0}


@pytest.mark.parametrize("bits", [3, 2])
@pytest.mark.parametrize(("group", "block", "official_block", "act_order", "damp", "seed"), [
    (128, 128, 128, False, 0.01, 0), (128, 128, 128, False, 0.01, 1), (128, 128, 128, True, 0.01, 0),
    (128, 64, 64, False, 0.01, 0), (128, 32, 32, False, 0.01, 0), (256, 128, 128, False, 0.01, 0),
    (128, 128, 128, False, 0.1, 0), (64, 128, 128, True, 0.01, 0),
    (64, 128, 64, False, 0.01, 0), (32, 128, 32, False, 0.01, 0),  # cause 4: official block = group
])
def test_low_bit_gptq_is_the_official_code_with_causes_1_to_3_made_the_same(
        official, bits, group, block, official_block, act_order, damp, seed) -> None:
    W, X = _layer(seed)
    H = _official_hessian(official[0], W, X)
    ours = run_tricast(W, H, bits, group=group, block=block, act_order=act_order, damp=damp)
    theirs = run_official(official, W, H, bits, group=group, block=official_block, act_order=act_order,
                          damp=damp, instrumented=True)
    assert theirs["dtypes"] == {torch.float64}  # the official loop ran on fp64 weights (cause 2 fired)
    assert mismatches(ours, theirs) == NONE
    assert set(ours["codes"].unique().tolist()) <= set(range(2 ** bits))
    assert len(ours["codes"].unique()) == 2 ** bits  # every level of the n-bit grid is used
    plain = run_official(official, W, H, bits, group=group, block=official_block, act_order=act_order,
                         damp=damp, instrumented=False)
    assert plain["dtypes"] == {torch.float32}
    assert mismatches(ours, plain)["scale"] > 0  # cause 1 is real: fp16 vs fp32 group scales


@pytest.mark.parametrize("bits", [3, 2])
@pytest.mark.parametrize("group", [32, 64])
def test_groups_inside_a_lazy_block_are_refreshed_from_compensated_weights(official, bits, group) -> None:
    """Cause 4: with the official block size 128 the official code refreshes these groups from weights that
    lack the block's earlier updates, so the codes differ; with block = group (every group starts a block)
    they are bit-identical (test above)."""
    W, X = _layer(0)
    H = _official_hessian(official[0], W, X)
    ours = run_tricast(W, H, bits, group=group, block=128, act_order=False, damp=0.01)
    theirs = run_official(official, W, H, bits, group=group, block=128, act_order=False, damp=0.01,
                          instrumented=True)
    assert mismatches(ours, theirs)["codes"] > 0


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_a_low_bit_gptq_layer_applies_like_gptq_int4(bits: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe path: an EmuLinear of the method calibrates, calls weight_algo gptq with the method's
    spec and the fp64 Hessian 2 X^T X / n, and uses that QTensor (dequantized onto BF16) as its weight."""
    import tricast.nn.linear as linear_module

    calls = []

    def recording(W, spec, algo, *, hessian=None, backend="reference"):
        calls.append((spec, algo, hessian))
        return quantize_weight(W, spec, algo, hessian=hessian, backend=backend)

    monkeypatch.setattr(linear_module, "quantize_weight", recording)
    fields = {"weight": "int4_g128_zp", "activation": None, "transform": "none", "weight_algo": "gptq"} \
        if bits == 4 else gptq_low(bits)
    recipe = load_recipe({"name": "t", "defaults": fields})
    assert recipe.needs_calibration
    spec = recipe.defaults
    assert (spec.weight.format.name, spec.weight.group_size, spec.weight.scale.format.name,
            spec.weight.zero_point) == (f"uint{bits}", 128, "fp16", "int")
    W, X = _layer(3)
    linear = torch.nn.Linear(COLS, ROWS, bias=False)
    linear.weight.data = W.clone()
    layer = EmuLinear(linear, spec, "layer")
    with pytest.raises(RuntimeError, match="calibration is required"):
        layer(X[:2])
    layer.begin_calibration()
    with torch.no_grad():
        layer(X)
    layer.finish_calibration()
    ((used_spec, algo, hessian),) = calls
    assert used_spec == spec.weight and algo == WeightAlgoSpec("gptq")
    gram = X.double().T @ X.double()
    assert hessian.dtype == torch.float64 and torch.equal(hessian, gram * (2.0 / TOKENS))
    expected = quantize_weight(W, spec.weight, algo, hessian=hessian).dequantize().bfloat16()
    assert torch.equal(layer._weight_operand.values.view(torch.int16), expected.view(torch.int16))


# ---------- HQQ 3 / 2-bit ----------
@pytest.mark.parametrize(("kind", "nbits"), [("hqq3", 3), ("hqq2", 2)])
def test_hqq_low_bit_is_the_official_call(kind: str, nbits: int) -> None:
    quantize_module = pytest.importorskip("hqq.core.quantize")
    from tricast.weight_quant.hqq import hqq

    torch.manual_seed(0)
    cases = {"normal": torch.randn(256, 512) * 0.02,
             "student_t2": torch.distributions.StudentT(2.0).sample((256, 512)) * 0.02,
             "bf16": (torch.randn(128, 256) * 0.05).bfloat16()}
    algo = load_recipe({"name": kind, "defaults": {"weight": None, "weight_algo": kind}}).defaults.weight_algo
    assert type(algo) is WeightMethodSpec and algo.kind == kind
    for name, x in cases.items():
        config = quantize_module.BaseQuantizeConfig(nbits=nbits, group_size=64)
        assert config["weight_quant_params"] == {"nbits": nbits, "channel_wise": True, "group_size": 64,
                                                 "optimize": True, "round_zero": False, "axis": 1,
                                                 "view_as_float": False}
        layer = quantize_module.HQQLinear.from_weights(x.clone(), None, config, compute_dtype=torch.bfloat16,
                                                       device="cpu")
        ours = replace_weight(x, algo)
        assert ours.dtype == torch.bfloat16 and torch.equal(ours.view(torch.int16),
                                                            layer.dequantize().view(torch.int16)), name
        assert torch.equal(ours.view(torch.int16), hqq(x, nbits).view(torch.int16))
        groups = x.numel() // 64
        assert {key: (layer.meta[key].dtype, tuple(layer.meta[key].shape)) for key in ("scale", "zero")} == {
            "scale": (torch.bfloat16, (groups, 1)), "zero": (torch.bfloat16, (groups, 1))}
        assert max(len(group.unique()) for group in ours.reshape(-1, 64)) <= 2 ** nbits
        plain = dict(config, weight_quant_params={**config["weight_quant_params"], "optimize": False})
        unoptimized = quantize_module.HQQLinear.from_weights(x.clone(), None, plain,
                                                             compute_dtype=torch.bfloat16,
                                                             device="cpu").dequantize()
        assert not torch.equal(ours, unoptimized), name  # the proximal solver ran


def test_hqq_kinds_and_the_4_bit_recipe_hash() -> None:
    pytest.importorskip("hqq")
    from tricast.weight_quant.hqq import KIND_NBITS, hqq, hqq_int4

    assert KIND_NBITS == {"hqq": 4, "hqq3": 3, "hqq2": 2} and set(KIND_NBITS) <= set(WEIGHT_METHODS)
    x = torch.randn(64, 128, generator=torch.Generator().manual_seed(1)) * 0.02
    assert torch.equal(replace_weight(x, WeightMethodSpec("hqq")).view(torch.int16),
                       hqq_int4(x).view(torch.int16))
    outs = {nbits: hqq(x, nbits) for nbits in (4, 3, 2)}
    assert len({out.view(torch.int16).numpy().tobytes() for out in outs.values()}) == 3
    with pytest.raises(ValueError, match="nbits"):
        hqq(x, 8)
    # the 4-bit method's recipe as recorded at 7120c41 (before hqq3 / hqq2 existed)
    alone = {"name": "hqq_int4", "defaults": {"weight": None, "activation": None, "transform": "none",
                                               "weight_algo": "hqq"}}
    assert load_recipe(alone).sha256 == "f9fc0b27c173b15f3305dc0cafc195f8ed48b7789601cacbf83f390f5dd95406"


# ---------- Hadamard + NVFP4 / MXFP4 (W4A4) ----------
def _hadamard_layer(fmt: str, fields: dict, seed: int = 0) -> tuple[EmuLinear, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(COLS, ROWS, bias=False)
    linear.weight.data = torch.randn(ROWS, COLS, generator=g) * 0.02
    x = torch.randn(5, COLS, generator=g) * torch.linspace(0.1, 4.0, COLS)
    x[:, 7] *= 30  # an outlier channel, which the rotation spreads
    return EmuLinear(linear, spec_of(fields), fmt), x


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_hadamard_block_is_128_on_qwen3_and_function_preserving(fmt: str) -> None:
    spec = spec_of(HAD[fmt])
    # block 0 = the largest power of two <= 128 that divides K
    assert spec.transform.kind == "hadamard" and spec.transform.block == 0
    assert not load_recipe({"name": "t", "defaults": HAD[fmt]}).needs_calibration
    for K in (1024, 2048, 3072, COLS):  # Qwen3-0.6B in_features, and this test's K
        assert fit_transform(spec.transform, torch.ones(2, K), None).block == 128
    g = torch.Generator().manual_seed(0)
    x = torch.randn(9, COLS, dtype=torch.float64, generator=g)
    w = torch.randn(7, COLS, dtype=torch.float64, generator=g)
    transform = fit_transform(spec.transform, w, None)
    actual = transform.apply_activation(x) @ transform.apply_weight(w).T
    # the criterion of tests/test_transforms.py::test_transform_preserves_linear_function
    torch.testing.assert_close(actual, x @ w.T, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_hadamard_methods_quantize_both_rotated_operands(fmt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import tricast.nn.linear as linear_module

    calls = []

    def recording(x, spec, **kwargs):
        calls.append((spec, x.detach().clone()))
        return quantize(x, spec, **kwargs)

    monkeypatch.setattr(linear_module, "quantize", recording)
    layer, x = _hadamard_layer(fmt, HAD[fmt])
    spec = layer.spec
    assert spec.weight == spec.activation == get_scheme(fmt).with_(mma_input="dequant")
    assert layer.transform.kind == "hadamard" and layer.transform.block == 128
    out = layer(x)
    rotated_w = layer.transform.apply_weight(layer.weight.detach())
    rotated_x = layer.transform.apply_activation(x)
    (wspec, wseen), (aspec, aseen) = calls  # requantize (weight), then forward (activation)
    assert wspec == spec.weight and torch.equal(wseen, rotated_w)
    assert aspec == spec.activation and torch.equal(aseen, rotated_x)
    q_w = as_operand(quantize(rotated_w, spec.weight), compact=True)
    q_x = as_operand(quantize(rotated_x, spec.activation))
    assert torch.equal(out, gemm(q_x, q_w, spec.mma))
    assert not torch.equal(out, gemm(as_operand(rotated_x), q_w, spec.mma))  # activations quantized
    assert not torch.equal(out, gemm(q_x, as_operand(rotated_w), spec.mma))  # weights quantized
    plain, _ = _hadamard_layer(fmt, PLAIN[fmt])
    assert plain.transform.kind == "none" and not torch.equal(out, plain(x))  # the transform fired


# ---------- AWQ INT4 (weight-only) ----------
def test_awq_int4_is_calibrated_and_searches_with_the_int4_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The AWQ grid search quantizes every candidate weight with the method's own int4_g128_zp spec (the
    weight the layer then uses), never an activation (weight-only); the chosen scale is the grid minimum of
    that error, and the forward GEMM takes unquantized (scaled) activations."""
    import tricast.nn.linear as linear_module

    recipe = load_recipe({"name": "t", "defaults": AWQ_INT4})
    spec = recipe.defaults
    assert recipe.needs_calibration and spec.activation is None and spec.weight_algo.kind == "rtn"
    assert (spec.weight.format.name, spec.weight.group_size, spec.weight.zero_point) == ("uint4", 128, "int")
    g = torch.Generator().manual_seed(0)
    linear = torch.nn.Linear(COLS, ROWS, bias=False)
    linear.weight.data = torch.randn(ROWS, COLS, generator=g) * 0.02
    x = torch.randn(96, COLS, generator=g) * torch.linspace(0.05, 6.0, COLS)
    layer = EmuLinear(linear, spec, "layer")
    assert layer.needs_calibration
    with pytest.raises(RuntimeError, match="calibration is required"):
        layer(x[:2])
    searched = []

    def recording(value, qspec, **kwargs):
        searched.append(qspec)
        return fake_quant(value, qspec, **kwargs)

    monkeypatch.setattr(quant_api, "fake_quant", recording)
    layer.begin_calibration()
    with torch.no_grad():
        layer(x)
    layer.finish_calibration()
    assert searched == [spec.weight] * spec.transform.grid  # one int4 weight per candidate, no activation

    collector = StatsCollector(COLS, want_xtx=False, want_samples=True)
    collector.update(x)
    stats = collector.result()
    target = linear.weight.detach().double() @ stats.samples.double().T
    candidates, errors = [], []
    for alpha in torch.linspace(0, 1, spec.transform.grid, dtype=torch.float64):
        s = stats.absmean.pow(float(alpha))
        s = torch.where(s == 0, 1e-5, s)
        s /= (s.amax() * s.amin()).sqrt()
        candidate = LinearTransform("awq", diag=s)
        weight = fake_quant(candidate.apply_weight(linear.weight.detach()), spec.weight, backend="reference")
        activation = candidate.apply_activation(stats.samples)
        candidates.append(candidate)
        errors.append(float((weight.double() @ activation.double().T - target).square().sum()))
    best = min(range(len(errors)), key=errors.__getitem__)
    # the int4 error ranks the candidates, and the minimum is a real scale (alpha 0 = all ones)
    assert len(set(errors)) == len(errors) and best != 0
    assert torch.equal(layer.transform.diag, candidates[best].diag)

    calls = []
    monkeypatch.setattr(linear_module, "quantize", lambda *args, **kwargs: calls.append(args) or
                        quantize(*args, **kwargs))
    out = layer(x[:4])
    assert calls == []  # the forward quantizes no activation
    weight = as_operand(quantize(layer.transform.apply_weight(linear.weight.detach()), spec.weight),
                        compact=True)
    assert torch.equal(out, gemm(as_operand(layer.transform.apply_activation(x[:4])), weight, spec.mma))
