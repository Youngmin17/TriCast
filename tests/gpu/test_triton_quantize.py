"""L1 bit-exact checks; run on CUDA with Triton 3.4 or later."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from types import ModuleType

import pytest
import torch

from tests.conftest import bit_equal, wide_fp32
from tricast.formats import REGISTRY, FloatFormat, IntFormat, Pow2Format, container_dtype, get_format
from tricast.quant.spec import SCHEMES, QuantSpec, ScaleSpec
from tricast.reference.cast import round_to_format
from tricast.rounding import Rounding

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("tricast.quant.qtensor")
    return pytest.importorskip("tricast.kernels.quantize")


@pytest.fixture(scope="module")
def reference():
    return pytest.importorskip("tricast.reference.quantize")


@pytest.fixture(scope="module")
def wide():
    return wide_fp32(100_000, torch.Generator().manual_seed(42), lo_exp=0, hi_exp=255)


def _boundaries(fmt):
    points = [0.0, -0.0, 2.0**-149, 2.0**-126, 1.0, 1.5, 2.0, fmt.min_normal, fmt.max_normal]
    quantum = getattr(fmt, "min_subnormal", fmt.min_normal)
    points += [quantum, quantum / 2, quantum * 1.5, quantum * 2.5]
    if isinstance(fmt, FloatFormat):
        points += [fmt.max_normal + 2.0**(fmt.emax - fmt.mbits - 1)]
    elif isinstance(fmt, IntFormat):
        points += [abs(fmt.qmin) * 2.0**-fmt.frac_bits]
    finite = torch.tensor(points, dtype=torch.float64).float()
    finite = torch.cat((finite, torch.nextafter(finite, torch.full_like(finite, float("inf"))),
                        torch.nextafter(finite, torch.full_like(finite, -float("inf")))))
    specials = [float("nan"), -float("inf"), float("inf")]
    return torch.cat((finite, -finite, torch.tensor(specials)))


@pytest.mark.parametrize("fmt", REGISTRY.values(), ids=REGISTRY.keys())
@pytest.mark.parametrize("rounding", list(Rounding))
@pytest.mark.parametrize("saturate", [False, True])
def test_cast_registry(backend, wide, fmt, rounding, saturate):
    x = torch.cat((wide, _boundaries(fmt)))
    if isinstance(fmt, Pow2Format) and rounding is Rounding.SR:
        with pytest.raises(ValueError, match="stochastic rounding"):
            backend.round_to_format_triton(x.cuda(), fmt, rounding, saturate=saturate)
        with pytest.raises(ValueError, match="stochastic rounding"):
            round_to_format(x, fmt, rounding, saturate=saturate)
        return
    noise = torch.randint(0, 2**32, x.shape, generator=torch.Generator().manual_seed(42), dtype=torch.int64)
    expected = round_to_format(x, fmt, rounding, saturate=saturate, noise=noise)
    actual = backend.round_to_format_triton(x.cuda(), fmt, rounding, saturate=saturate, noise=noise.cuda())
    assert bit_equal(actual.cpu(), expected)


@pytest.mark.parametrize("fmt", ["e3m4:nosub", "e8m0:ieee", "int8:full", "int24", "uint24",
                                  "int9:frac=130", "e8m23:ieee:bias=127"])
@pytest.mark.parametrize("rounding", list(Rounding))
def test_custom_formats(backend, fmt, rounding):
    fmt = get_format(fmt)
    x = torch.cat((_boundaries(fmt), wide_fp32(1024, torch.Generator().manual_seed(42), 0, 255)))
    noise = torch.arange(x.numel(), dtype=torch.int64) * 104729
    expected = round_to_format(x, fmt, rounding, saturate=False, noise=noise)
    actual = backend.round_to_format_triton(x.cuda(), fmt, rounding, saturate=False, noise=noise.cuda())
    assert bit_equal(actual.cpu(), expected)


@pytest.mark.parametrize("sr_bits", [1, 8, 16, 32])
@pytest.mark.parametrize("fmt", ["fp32", "bf16", "fp16", "fp4", "int8"])
def test_sr_far_underflow(backend, sr_bits, fmt):
    x = torch.tensor([2.0**-149, 2.0**-126, 2.0**-40, 2.0**-25, 0.125, 1.0625])
    x = torch.cat((x, -x))[:, None]
    noise = torch.tensor([0, 1, 255, 65535, 2**23 - 1, 2**31 - 1, 2**31, 2**32 - 1])[None, :]
    x = x.expand(-1, noise.numel()).contiguous()
    actual = backend.round_to_format_triton(x.cuda(), fmt, "sr", noise=noise.cuda(), sr_bits=sr_bits)
    assert bit_equal(actual.cpu(), round_to_format(x, fmt, "sr", noise=noise, sr_bits=sr_bits))


@pytest.mark.parametrize("rounding", list(Rounding)[:-1])
@pytest.mark.parametrize("saturate", [False, True])
def test_pow2_infinity_contract(backend, rounding, saturate):
    x = torch.tensor([float("inf")])
    fmt = REGISTRY["e8m0"]
    actual = backend.round_to_format_triton(x.cuda(), fmt, rounding, saturate=saturate).cpu()
    contract = torch.tensor([fmt.max_normal if saturate else float("nan")])
    assert bit_equal(actual, contract)
    expected = round_to_format(x, fmt, rounding, saturate=saturate)
    assert bit_equal(actual, expected), "reference Pow2 +Inf must implement ENGINE §2 overflow"


@pytest.mark.parametrize("fmt", ["fp32", "bf16", "fp16", "fp8_e4m3fnuz", "ue4m3"])
def test_zero_sign_and_subnormal_identity(backend, fmt):
    x = torch.tensor([0, -2147483648, 1, -2147483647, 0x007FFFFF], dtype=torch.int32).view(torch.float32)
    actual = backend.round_to_format_triton(x.cuda(), fmt).cpu()
    expected = round_to_format(x, fmt)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("fmt", ["uint8", "uint4", "uint24"])
def test_unsigned_integer_zero_has_no_sign(backend, fmt):
    x = torch.tensor([-1.0, -0.0, -0.25, -3e38, -float("inf"), 0.0])
    actual = backend.round_to_format_triton(x.cuda(), fmt).cpu()
    expected = round_to_format(x, fmt)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert not (actual.view(torch.int32) < 0).any()


@pytest.mark.parametrize("method", ["pow2_floor", "pow2_ceil"])
@pytest.mark.parametrize("scale_format", ["ue4m3", "e8m0"])
def test_two_level_pow2_methods(backend, reference, method, scale_format):
    scale = ScaleSpec(scale_format, method, two_level=True)
    spec = QuantSpec("fp4_e2m1", "group", group_size=16, scale=scale)
    x = torch.cat((wide_fp32(4096, torch.Generator().manual_seed(42), 110, 140), torch.tensor([6.0] * 32)))
    _assert_qtensor(backend.quantize_triton(x.reshape(-1, 32).cuda(), spec),
                    reference.quantize_reference(x.reshape(-1, 32), spec))


def _assert_qtensor(actual, expected):
    assert actual.shape == expected.shape
    assert actual.spec == expected.spec
    for name in ("values", "scale", "zero_point", "global_scale"):
        left, right = getattr(actual, name), getattr(expected, name)
        if right is None:
            assert left is None, name
        else:
            assert left.shape == right.shape, name
            assert left.dtype == right.dtype, name
            assert bit_equal(left.cpu(), right.cpu()), name


@pytest.mark.parametrize("name", SCHEMES)
@pytest.mark.parametrize("shape", [(37,), (3, 35), (2, 3, 129), (129, 7)])
def test_schemes(backend, reference, name, shape):
    spec = SCHEMES[name]
    x = torch.randn(shape, generator=torch.Generator().manual_seed(42)) * 3.25
    x.reshape(-1)[0] = 0
    x = x.cuda()
    expected = reference.quantize_reference(x, spec)
    actual = backend.quantize_triton(x, spec, seed=42)
    _assert_qtensor(actual, expected)
    assert actual.values.dtype == container_dtype(spec.format)


@pytest.mark.parametrize("method", ["absmax", "pow2_floor", "pow2_ceil", "mse", "percentile"])
@pytest.mark.parametrize("granularity", ["tensor", "row", "group", "block"])
@pytest.mark.parametrize("rounding", list(Rounding))
def test_scale_methods_and_rounding(backend, reference, method, granularity, rounding):
    spec = QuantSpec("fp4", granularity, group_size=7, block=(2, 5),
                     scale=ScaleSpec("e8m0" if method.startswith("pow2") else "fp16", method,
                                     mse_grid=3), rounding=rounding)
    x = torch.randn((3, 17), generator=torch.Generator().manual_seed(42)).cuda()
    noise = torch.randint(0, 2**32, x.shape, generator=torch.Generator().manual_seed(42),
                          dtype=torch.int64).cuda()
    # Reference MSE candidates draw CPU SR noise separately from element noise.
    state = torch.Generator().manual_seed(42).get_state()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(state)
        expected = reference.quantize_reference(x, spec, noise=noise)
        torch.set_rng_state(state)
        actual = backend.quantize_triton(x, spec, noise=noise)
    _assert_qtensor(actual, expected)


@pytest.mark.parametrize("name", ["mxfp4", "nvfp4", "fp8_tensor", "int4_g128"])
@pytest.mark.parametrize("value", [0.0, 2.0**-149, 2.0**-126, float("inf"), float("nan")])
def test_scale_specials(backend, reference, name, value):
    x = torch.full((2, 35), value, device="cuda")
    spec = SCHEMES[name]
    _assert_qtensor(backend.quantize_triton(x, spec), reference.quantize_reference(x, spec))


@pytest.mark.parametrize("zero_point", ["int", "float"])
@pytest.mark.parametrize("rounding", list(Rounding))
def test_zero_points(backend, reference, zero_point, rounding):
    spec = QuantSpec("uint4", "group", group_size=5, zero_point=zero_point, mma_input="dequant",
                     scale=ScaleSpec("fp16"), rounding=rounding)
    x = torch.tensor([[-3.1, -1.25, -0.1, 0.0, 7.3, 0.125, 2.75]], device="cuda")
    noise = torch.tensor([0, 1, 2**31, 2**32 - 1, 17, 31, 255], device="cuda")
    _assert_qtensor(backend.quantize_triton(x, spec, noise=noise),
                    reference.quantize_reference(x, spec, noise=noise))


@pytest.mark.parametrize("name", ["kivi2", "kivi4"])
@pytest.mark.parametrize("rounding", list(Rounding))
def test_float_zero_point_domains_and_no_element_fallback(
    backend: ModuleType, reference: ModuleType, monkeypatch: pytest.MonkeyPatch,
    name: str, rounding: Rounding,
) -> None:
    spec = SCHEMES[name].with_(group_size=4, rounding=rounding)
    x = torch.tensor([[-1., .1, .5, 2., 2., 2., 2.],
                      [2., 2.25, 3.5, 5., -2., -2., -2.],
                      [-5., -4., -3., -2., 0., 0., 0.]])
    noise = torch.tensor([0, 2**30 - 1, 2**30, 2**32 - 1, 0, 1, 2**31])
    expected = reference.quantize_reference(x, spec, noise=noise)

    def forbidden(*args, **kwargs):
        raise AssertionError("float zero-point Triton path called reference element rounding")

    monkeypatch.setattr(reference, "quantize_elements", forbidden)
    actual = backend.quantize_triton(x.cuda(), spec, noise=noise.cuda(), seed=42)
    _assert_qtensor(actual, expected)
    assert bit_equal(actual.dequantize().cpu(), expected.dequantize())
    assert bit_equal(actual.mma_operand().values.cpu(), expected.mma_operand().values)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_float_zero_point_subnormal_arithmetic(
    backend: ModuleType, reference: ModuleType, sign: float,
) -> None:
    spec = SCHEMES["kivi2"].with_(scale=ScaleSpec("fp32"), dequant_format="fp32")
    x = torch.tensor([[sign * k * 2.**-149 for k in (1, 2, 3, 4)]])
    expected = reference.quantize_reference(x, spec)
    actual = backend.quantize_triton(x.cuda(), spec)
    _assert_qtensor(actual, expected)
    assert bit_equal(actual.dequantize().cpu(), x)
    assert bit_equal(actual.mma_operand().values.cpu(), x)


def test_float_zero_point_subtract_rounds_before_divide(backend: ModuleType, reference: ModuleType) -> None:
    spec = SCHEMES["kivi2"]
    x = torch.tensor([[-2.**-24, 1., 6.]])
    expected = reference.quantize_reference(x, spec)
    actual = backend.quantize_triton(x.cuda(), spec)
    _assert_qtensor(actual, expected)
    # s=2 and RN(1+2^-24)=1; the middle code is the even result of the tie at 1/2.
    assert bit_equal(actual.values.cpu(), torch.tensor([[0., 0., 3.]]))


def test_observer_amax_and_noncontiguous_input(backend, reference):
    spec = SCHEMES["fp8_tensor_ema"]
    x = torch.randn((7, 3), generator=torch.Generator().manual_seed(42)).cuda().T
    amax = torch.tensor(8.0, device="cuda")
    _assert_qtensor(backend.quantize_triton(x, spec, amax=amax),
                    reference.quantize_reference(x, spec, amax=amax))


@pytest.mark.parametrize("method", ["absmax", "pow2_floor", "pow2_ceil"])
def test_negative_zero_amax(backend, reference, method):
    spec = QuantSpec("fp4", scale=ScaleSpec("e8m0" if method.startswith("pow2") else "fp32", method))
    x = torch.zeros((2, 7), device="cuda")
    amax = torch.tensor(-0.0, device="cuda")
    _assert_qtensor(backend.quantize_triton(x, spec, amax=amax),
                    reference.quantize_reference(x, spec, amax=amax))


@pytest.mark.parametrize("name", ["bf16", "mxfp4", "nvfp4", "fp8_block128"])
def test_no_reference_element_fallback(backend, reference, monkeypatch, name):
    def forbidden(*args, **kwargs):
        raise AssertionError("normal Triton path called reference arithmetic")

    monkeypatch.setattr(reference, "compute_scale", forbidden)
    monkeypatch.setattr(reference, "quantize_elements", forbidden)
    monkeypatch.setattr(reference, "round_to_format", forbidden)
    x = torch.tensor([[0.1, 0.3, 0.7, 1.1, 1.7]], device="cuda")
    actual = backend.quantize_triton(x, SCHEMES[name])
    assert torch.isfinite(actual.values).all()
    assert not torch.equal(actual.values.float(), x)


def test_seeded_sr_is_repeatable(backend):
    spec = SCHEMES["mxfp4"].with_(rounding=Rounding.SR)
    x = torch.linspace(-1.5, 1.5, 1025, device="cuda")
    first = backend.quantize_triton(x, spec, seed=42)
    _assert_qtensor(first, backend.quantize_triton(x, spec, seed=42))
    assert not bit_equal(first.values, backend.quantize_triton(x, spec, seed=43).values)


def test_throughput_smoke(backend, tmp_path):
    torch.manual_seed(42)
    x = torch.randn((256, 1024), device="cuda")
    spec = SCHEMES["mxfp4"]
    for _ in range(3):
        backend.quantize_triton(x, spec, seed=42)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = backend.quantize_triton(x, spec, seed=42)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    median = statistics.median(samples)
    p99 = torch.quantile(torch.tensor(samples, dtype=torch.float64), 0.99).item()
    nbytes = x.numel() * x.element_size() + output.values.numel() * output.values.element_size()
    gbps = nbytes / median / 1e6
    env = {"torch": torch.__version__, "triton": pytest.importorskip("triton").__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
           "device": torch.cuda.current_device(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
           "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "seed": 42, "warmup": 3, "repeat": 5, "shape": list(x.shape)}
    (tmp_path / "env.json").write_text(json.dumps(env, indent=2))
    (tmp_path / "result.json").write_text(json.dumps({"milliseconds": samples, "gbps": gbps}))
    print(f"TriCast smoke: {gbps:.3f} GB/s median={median:.3f} mean={statistics.mean(samples):.3f} "
          f"p99={p99:.3f} ms env={tmp_path / 'env.json'}")


@pytest.mark.parametrize("granularity", ["tensor", "group"])
def test_amax_flat_grid_beyond_cuda_y_limit(backend: ModuleType, granularity: str) -> None:
    # Tensor needs 65,536 partials; group needs 65,536 domain blocks. Neither
    # dimension can be moved to grid.y, whose CUDA limit is 65,535.
    size = 65535 * (1024 if granularity == "tensor" else 4) + 1
    spec = QuantSpec("fp8_e4m3", granularity, group_size=1, scale=ScaleSpec("fp32"))
    x = torch.zeros((1, size), device="cuda")
    x[0, -1] = spec.format.max_normal
    actual = backend.quantize_triton(x, spec)
    # Every domain is zero or has maximum equal to max_normal: scale is exactly
    # one and each input is already on the representable grid.
    assert torch.equal(actual.scale, torch.ones_like(actual.scale))
    assert torch.equal(actual.values.float().view(torch.int32), x.view(torch.int32))


@pytest.mark.parametrize("operation", ["cast", "quantize"])
def test_quantize_rejects_int32_index_overflow(backend: ModuleType, operation: str) -> None:
    # A broadcast view exercises the pre-launch bound without a multi-GB allocation.
    x = torch.zeros((), device="cuda").expand(2**31)
    with pytest.raises(ValueError, match=r"fewer than 2\^31 input elements"):
        if operation == "cast":
            backend.round_to_format_triton(x, "fp32")
        else:
            backend.quantize_triton(x, SCHEMES["fp8_tensor"])


def test_quantize_rejects_int32_domain_overflow(backend: ModuleType) -> None:
    x = torch.zeros((1, 1), device="cuda")
    spec = QuantSpec("fp8_e4m3", "block", block=(2**16, 2**15), scale=ScaleSpec("fp32"))
    with pytest.raises(ValueError, match=r"domains smaller than 2\^31"):
        backend.quantize_triton(x, spec)


def test_quantize_rejects_int32_partial_overflow(backend: ModuleType) -> None:
    x = torch.zeros((2**12, 1), device="cuda")
    spec = QuantSpec("fp8_e4m3", "group", group_size=2**29, scale=ScaleSpec("fp32"))
    with pytest.raises(ValueError, match=r"fewer than 2\^31 partials"):
        backend.quantize_triton(x, spec)


@pytest.mark.parametrize("granularity", ["tensor", "row", "group", "block"])
@pytest.mark.parametrize("two_level", [False, True])
def test_scale_sr_explicit_streams_without_reference_fallback(
    backend: ModuleType, reference: ModuleType, monkeypatch: pytest.MonkeyPatch,
    granularity: str, two_level: bool,
) -> None:
    from tricast.quant.api import quantize

    generator = torch.Generator().manual_seed(42)
    x = torch.randn((5, 19), generator=generator) * 7
    spec = QuantSpec("fp4", granularity, group_size=7, block=(2, 5), rounding="sr",
                     scale=ScaleSpec("fp16", rounding="sr", two_level=two_level))
    element_noise = torch.randint(0, 2**32, (1, 19), generator=generator, dtype=torch.int64)
    scale_noise = torch.randint(0, 2**32, reference._layout(spec, *x.shape),
                                generator=torch.Generator().manual_seed(43), dtype=torch.int64)
    state = torch.random.get_rng_state()
    expected = quantize(x, spec, backend="reference", noise=element_noise, scale_noise=scale_noise)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("absmax scale SR must execute the Triton scale and element kernels")

    monkeypatch.setattr(reference, "compute_scale", forbidden)
    monkeypatch.setattr(reference, "quantize_elements", forbidden)
    actual = quantize(x.cuda(), spec, backend="triton", noise=element_noise.cuda(),
                      scale_noise=scale_noise.cuda())
    repeated = backend.quantize_triton(x.cuda(), spec, noise=element_noise.cuda(),
                                       scale_noise=scale_noise.cuda())
    assert torch.equal(torch.random.get_rng_state(), state)
    _assert_qtensor(actual, expected)
    _assert_qtensor(repeated, expected)
    for name in ("values", "scale", "global_scale"):
        value = getattr(expected, name)
        if value is not None:
            assert torch.equal(getattr(actual, name).cpu().float().view(torch.int32),
                               value.float().view(torch.int32))


@pytest.mark.parametrize("granularity", ["group", "block"])
@pytest.mark.parametrize("method,zero_point", [
    ("mse", "none"), ("percentile", "none"), ("absmax", "int"), ("absmax", "float"),
])
def test_scale_sr_search_and_zero_point_explicit_streams(
    reference: ModuleType, granularity: str, method: str, zero_point: str,
) -> None:
    from tricast.quant.api import fake_quant, quantize

    generator = torch.Generator().manual_seed(42)
    x = torch.randn((5, 19), generator=generator) * 7
    spec = QuantSpec("fp4" if zero_point == "none" else "uint4", granularity,
                     group_size=7, block=(2, 5), rounding="sr", zero_point=zero_point,
                     mma_input="dequant", dequant_format="fp32",
                     scale=ScaleSpec("fp16", method, rounding="sr", mse_grid=3, percentile=73.25))
    noise = torch.randint(0, 2**32, x.shape, generator=generator, dtype=torch.int64)
    scale_noise = torch.randint(0, 2**32, reference._layout(spec, *x.shape),
                                generator=torch.Generator().manual_seed(43), dtype=torch.int64)
    state = torch.random.get_rng_state()
    expected = quantize(x, spec, backend="reference", noise=noise, scale_noise=scale_noise)
    expected_fake = fake_quant(x, spec, backend="reference", noise=noise, scale_noise=scale_noise)
    actual = quantize(x.cuda(), spec, backend="triton", noise=noise.cuda(), scale_noise=scale_noise.cuda())
    repeated = quantize(x.cuda(), spec, backend="triton", noise=noise.cuda(), scale_noise=scale_noise.cuda())
    actual_fake = fake_quant(x.cuda(), spec, backend="triton", noise=noise.cuda(),
                             scale_noise=scale_noise.cuda())
    assert torch.equal(torch.random.get_rng_state(), state)
    _assert_qtensor(actual, expected)
    _assert_qtensor(repeated, expected)
    for name in ("values", "scale", "zero_point"):
        value = getattr(expected, name)
        if value is not None:
            assert torch.equal(getattr(actual, name).cpu().float().view(torch.int32),
                               value.float().view(torch.int32))
    assert torch.equal(actual_fake.cpu().view(torch.int32), expected_fake.view(torch.int32))


def test_scale_sr_noise_is_separate_from_element_noise(backend: ModuleType, reference: ModuleType) -> None:
    x = torch.tensor([[.15, 1.05, 6.003], [.15, 1.05, 6.003]])
    spec = QuantSpec("fp4", "row", rounding="sr", scale=ScaleSpec("fp16", rounding="sr"))
    scale_noise = torch.tensor([[0], [2**32 - 1]])
    noise = torch.zeros_like(x, dtype=torch.int64)
    expected = reference.quantize_reference(x, spec, noise=noise, scale_noise=scale_noise)
    actual = backend.quantize_triton(x.cuda(), spec, noise=noise.cuda(), scale_noise=scale_noise.cuda())
    changed = backend.quantize_triton(x.cuda(), spec, noise=(noise + 2**32 - 1).cuda(),
                                      scale_noise=scale_noise.cuda())
    _assert_qtensor(actual, expected)
    assert torch.equal(actual.scale.cpu(), torch.tensor([[1. + 2.**-10], [1.]]))
    assert torch.equal(actual.scale, changed.scale)
    assert not torch.equal(actual.values, changed.values)


@pytest.mark.parametrize("method,zero_point", [
    ("absmax", "none"), ("mse", "none"), ("percentile", "none"),
    ("absmax", "int"), ("absmax", "float"),
])
def test_cuda_scale_selection_has_constant_domain_dispatch(
    reference: ModuleType, monkeypatch: pytest.MonkeyPatch, method: str, zero_point: str,
) -> None:
    from torch.profiler import ProfilerActivity, profile

    spec = QuantSpec("fp4" if zero_point == "none" else "uint4", "group", group_size=16,
                     zero_point=zero_point, mma_input="dequant", rounding="sr",
                     scale=ScaleSpec("fp16", method, rounding="sr", search=(1., .75, .5)))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA scale selection must not iterate scale domains in Python")

    monkeypatch.setattr(reference, "_domains", forbidden)
    operation_counts = []
    for domains in (16, 256):
        generator = torch.Generator().manual_seed(42)
        x = torch.randn((domains, 16), generator=generator).cuda()
        noise = torch.randint(0, 2**32, x.shape, generator=generator, dtype=torch.int64).cuda()
        scale_noise = torch.randint(0, 2**32, (domains, 1), generator=generator,
                                    dtype=torch.int64).cuda()
        reference.compute_scale(x, spec, noise=noise, scale_noise=scale_noise)
        previous = torch.cuda.get_sync_debug_mode()
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            try:
                torch.cuda.set_sync_debug_mode("error")
                maximum = reference.compute_amax(x, spec)
                scale, _, _ = reference.compute_scale(x, spec, noise=noise, scale_noise=scale_noise)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        assert maximum.shape == scale.shape == (domains, 1)
        counts = {event.key: event.count for event in profiler.key_averages()
                  if event.key.startswith("aten::")}
        assert "aten::_local_scalar_dense" not in counts
        operation_counts.append(counts)
    # This is a dispatch/host-sync regression, not a latency claim: a 16-fold
    # domain increase must not increase the number of dispatched ATen operators.
    assert operation_counts[0] == operation_counts[1]


def test_percentile_domain_beyond_quantile_size_limit(reference: ModuleType) -> None:
    # More than 2**24 elements used to fail inside torch.quantile, before rounding.
    x = torch.ones((1, 2**24 + 1), device="cuda")
    x[0, -1] = 3.
    spec = QuantSpec("int2", scale=ScaleSpec(method="percentile", percentile=99.9))
    scale, zero, global_scale = reference.compute_scale(x, spec)
    assert torch.equal(scale.cpu().view(torch.int32), torch.tensor(1.).view(torch.int32))
    assert zero is None and global_scale is None


@pytest.mark.parametrize("seed", [2, 3, 4, 7])
def test_float_zero_point_signed_zero_ragged_domains(
    backend: ModuleType, reference: ModuleType, seed: int,
) -> None:
    bits = torch.randint(0, 2, (17, 33), generator=torch.Generator().manual_seed(seed)) * -(2**31)
    x = bits.int().view(torch.float32)
    spec = QuantSpec("uint4", "block", block=(16, 16), zero_point="float", mma_input="dequant")
    expected = reference.quantize_reference(x, spec)
    actual = backend.quantize_triton(x.cuda(), spec)
    _assert_qtensor(actual, expected)
    assert torch.equal(actual.zero_point.cpu().view(torch.int32), expected.zero_point.view(torch.int32))
