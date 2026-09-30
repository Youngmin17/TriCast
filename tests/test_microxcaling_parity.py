"""Exact parity with the independent microxcaling PyTorch implementation."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import bit_equal
from tricast.formats import E8M0, FloatFormat, get_format
from tricast.quant.spec import QuantSpec, ScaleSpec
from tricast.reference.cast import round_to_format
from tricast.rounding import from_microxcaling

mx = pytest.importorskip("mx")
pytestmark = pytest.mark.oracle

ELEMENTS = [
    ("fp8_e4m3", "fp8_e4m3"),
    ("fp8_e5m2", "fp8_e5m2"),
    ("fp6_e3m2", "fp6_e3m2"),
    ("fp6_e2m3", "fp6_e2m3"),
    ("fp4_e2m1", "fp4_e2m1"),
    ("int8", "mxint8"),
    ("int4", "mxint4"),
]
ROUNDS = ("even", "nearest", "floor")


def _random_values(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(shape, generator=gen) * 10 ** (6 * torch.rand(shape, generator=gen) - 3)
    x.reshape(-1)[::97] = 0
    return x


def _log2_error(x: torch.Tensor) -> torch.Tensor:
    """Identify only fp32 log2 exponent errors, without looking at outputs."""
    ax = x.abs()
    positive = torch.where(ax == 0, torch.ones_like(ax), ax)
    exact = torch.frexp(positive)[1] - 1
    return torch.floor(torch.log2(positive)) != exact


def _group_exclusions(x: torch.Tensor, elem_name: str) -> torch.Tensor:
    fmt = get_format(elem_name)
    x2d = x.reshape(-1, x.shape[-1])
    excluded = torch.zeros_like(x2d, dtype=torch.bool)
    emax = QuantSpec(fmt).emax_elem
    for start in range(0, x2d.shape[-1], 32):
        block = x2d[:, start:start + 32]
        amax = block.abs().amax(dim=-1, keepdim=True)
        shared_bad = _log2_error(amax)
        safe_amax = torch.where(amax == 0, torch.full_like(amax, 2.0**-126), amax)
        exponent = (torch.frexp(safe_amax)[1] - 1 - emax).clamp(min=-127, max=127)
        scale = torch.ldexp(torch.ones_like(amax), exponent)
        private_bad = _log2_error(block / scale) if isinstance(fmt, FloatFormat) else False
        excluded[:, start:start + 32] = shared_bad | private_bad
    return excluded.reshape(x.shape)


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, excluded: torch.Tensor) -> None:
    count = int(excluded.sum())
    print(f"microxcaling log2 exclusions: {count}/{excluded.numel()}")
    assert count / excluded.numel() < 1e-3
    assert actual.shape == expected.shape
    assert bit_equal(actual[~excluded], expected[~excluded])


@pytest.mark.parametrize("oracle_elem,elem_name", ELEMENTS)
@pytest.mark.parametrize("round_name", ROUNDS)
@pytest.mark.parametrize("shape", [(3, 17), (3, 32), (3, 67), (2, 3, 35)])
def test_mx_group_parity(oracle_elem, elem_name, round_name, shape, gen):
    quantize_reference = pytest.importorskip("tricast.reference.quantize").quantize_reference
    x = _random_values(shape, gen)
    x.reshape(-1, shape[-1])[0] = 0
    spec = QuantSpec(
        elem_name, "group", group_size=32, scale=ScaleSpec(E8M0, "pow2_floor"),
        rounding=from_microxcaling(round_name),
    )
    expected = mx.mx_ops._quantize_mx(
        x, 8, oracle_elem, shared_exp_method="max", axes=[-1], block_size=32, round=round_name,
    )
    actual = quantize_reference(x, spec).dequantize()
    _assert_parity(actual, expected, _group_exclusions(x, elem_name))


@pytest.mark.parametrize("oracle_elem,elem_name", ELEMENTS)
@pytest.mark.parametrize("round_name", ROUNDS)
def test_elemwise_parity(oracle_elem, elem_name, round_name, gen):
    fmt = get_format(elem_name)
    boundaries = torch.tensor([
        0.0, -0.0, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 6.5, -6.5,
        fmt.min_normal, -fmt.min_normal, fmt.max_normal, -fmt.max_normal,
        2 * fmt.max_normal, -2 * fmt.max_normal,
    ])
    below_power = torch.nextafter(torch.tensor([16.0, -16.0]), torch.zeros(2))
    x = torch.cat((_random_values((8192,), gen), boundaries, below_power))
    ebits, mbits, _, max_norm, _ = mx.formats._get_format_params(oracle_elem)
    expected = mx.elemwise_ops._quantize_elemwise_core(
        x, mbits, ebits, max_norm, round=round_name, saturate_normals=True,
    )
    actual = round_to_format(x, fmt, from_microxcaling(round_name), saturate=True)
    excluded = _log2_error(x) if ebits else torch.zeros_like(x, dtype=torch.bool)
    _assert_parity(actual, expected, excluded)


def test_log2_exclusion_detects_only_inaccurate_exponents():
    powers = torch.tensor([16.0, 32.0, 64.0])
    below = torch.nextafter(powers, torch.zeros_like(powers))
    assert bool(_log2_error(below).any())
    assert not bool(_log2_error(torch.cat((powers, -powers, torch.zeros(2)))).any())
    assert torch.equal(_log2_error(below), _log2_error(-below))


def test_group_exclusion_covers_shared_domain_only():
    x = torch.ones(1, 64)
    x[0, 0] = torch.nextafter(torch.tensor(16.0), torch.tensor(0.0))
    excluded = _group_exclusions(x, "mxint8")
    assert bool(excluded[0, :32].all())
    assert not bool(excluded[0, 32:].any())


def test_group_exclusion_covers_private_exponent_only():
    x = torch.ones(1, 32)
    x[0, 0] = 512.0
    x[0, 1] = torch.nextafter(torch.tensor(16.0), torch.tensor(0.0))
    excluded = _group_exclusions(x, "fp8_e4m3")
    assert bool(excluded[0, 1])
    assert int(excluded.sum()) == 1
