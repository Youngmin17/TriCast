"""Hand-derived FDA, GDFS and epilogue cases from ENGINE.md §4."""

from __future__ import annotations

import pytest
import torch

from tricast.formats import BF16, E8M0, FP4_E2M1, FP8_E4M3, FP16, FP32, UE4M3, Format, IntFormat
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec

mma = pytest.importorskip("tricast.reference.mma")
gemm_reference = mma.gemm_reference
resolve_scale_apply = mma.resolve_scale_apply


def operand(values: list[float], fmt: Format = FP8_E4M3, **kwargs) -> Operand:
    return Operand(torch.tensor([values], dtype=torch.float32), fmt, **kwargs)


def from_bits(bits: int) -> float:
    return torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32).item()


def assert_bits(out: torch.Tensor, expected: int) -> None:
    assert out.dtype == torch.float32
    assert out.shape == (1, 1)
    assert out.view(torch.int32).item() & 0xFFFFFFFF == expected


def spec(**kwargs) -> MMASpec:
    return MMASpec(out_format=FP32, **kwargs)


@pytest.mark.parametrize("f_bits, expected", [(3, 0x40100000), (13, 0x40110000), (25, 0x40110100)])
def test_fp8_products_and_running_accumulator(f_bits: int, expected: int) -> None:
    # (9/8)^2 truncates to 10/8 at F=3. Then add 1 and 2^-14;
    # F=13 drops the last term, while F=25 keeps all 145/64 + 2^-14.
    a = operand([1.125, 1.0, 2.0**-7])
    assert_bits(gemm_reference(a, a, spec(f_bits=f_bits, chunk_size=1)), expected)


def test_decoupled_preserves_small_chunk_sum() -> None:
    # Fused: each 1 aligns as 8 >> 4 = 0 against c=16. Decoupled: P=2, then 16+2 at F2=23.
    a, b = operand([16, 0, 1, 1]), operand([1, 1, 1, 1])
    base = spec(f_bits=3, chunk_size=2)
    assert_bits(gemm_reference(a, b, base), 0x41800000)
    assert_bits(gemm_reference(a, b, base.with_(c_mode="decoupled", f2_bits=23)), 0x41900000)


def test_truncated_product_is_not_marked_zero() -> None:
    # 2^-149 * 2^127 has m_F=0 but exponent=1, so it still sets Emax;
    # 1.125 at exponent=0 aligns as 9 >> 1 = 4, yielding exactly 1 instead of 1.125.
    a = operand([2.0**-149, 1.125], FP32)
    b = operand([2.0**127, 1.0], FP32)
    assert_bits(gemm_reference(a, b, spec(f_bits=3, chunk_size=2)), 0x3F800000)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([2.0**-9], [2.0**-9], 0x36800000),
        ([2.0**-9, -(2.0**-9)], [2.0**-9, 2.0**-9], 0x00000000),
    ],
)
def test_fp8_subnormal_products(a: list[float], b: list[float], expected: int) -> None:
    # E4M3 subnormals use exponent=-6, m=1; their product is 1 * 2^(-12-6).
    assert_bits(gemm_reference(operand(a), operand(b), spec(f_bits=25)), expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([2.0**-74], [2.0**-75], 0x00000001),
        ([3 * 2.0**-75], [2.0**-75], 0x00000001),
        ([-(2.0**-75)], [2.0**-75], 0x80000000),
        ([3 * 2.0**-149, 0, 0], [1, 1, 1], 0x00000003),
    ],
)
def test_fp32_subnormal_result_and_accumulator(a: list[float], b: list[float], expected: int) -> None:
    # Subnormal normalization truncates at 2^-149, not F fraction bits;
    # c=3*2^-149 must keep its normalized 1.5 significand through zero chunks.
    assert_bits(gemm_reference(operand(a, FP32), operand(b, FP32), spec(f_bits=25, chunk_size=1)), expected)


@pytest.mark.parametrize("gap", [64, 120, 254])
def test_alignment_shift_at_least_64_is_zero(gap: int) -> None:
    # Equal large products cancel; the middle product is shifted by >=64, not shift modulo 64.
    a = operand([2.0 ** (gap // 2), 1, -(2.0 ** (gap // 2))], FP32)
    b = operand([2.0 ** (gap - gap // 2), 1, 2.0 ** (gap - gap // 2)], FP32)
    assert_bits(gemm_reference(a, b, spec(f_bits=25, chunk_size=3)), 0x00000000)


@pytest.mark.parametrize("sign, expected", [(1, 0x7F800000), (-1, 0xFF800000)])
def test_fixed_point_overflow_is_infinity(sign: int, expected: int) -> None:
    # Product exponent=200 lies above the fp32 maximum; no finite saturation is allowed.
    a, b = operand([sign * 2.0**100], FP32), operand([2.0**100], FP32)
    assert_bits(gemm_reference(a, b, spec(f_bits=25)), expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([float("nan"), 1], [0, 1], None),
        ([float("inf")], [0], None),
        ([float("inf"), -float("inf")], [1, 1], None),
        ([float("inf"), float("nan")], [1, 1], None),
        ([float("inf"), 1], [1, 1], 0x7F800000),
        ([-float("inf"), 1], [1, 1], 0xFF800000),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_special_values(a: list[float], b: list[float], expected: int | None, chunk_size: int) -> None:
    out = gemm_reference(operand(a, FP32), operand(b, FP32), spec(chunk_size=chunk_size))
    if expected is None:
        assert torch.isnan(out).all()
    else:
        assert_bits(out, expected)


@pytest.mark.parametrize("value, expected", [(0.0, 0x00000000), (1.25, 0x3FA00000), (-1.25, 0xBFA00000)])
def test_zero_chunks_preserve_accumulator(value: float, expected: int) -> None:
    a, b = operand([value, 0, 0]), operand([1, 1, 1])
    assert_bits(gemm_reference(a, b, spec(f_bits=3, chunk_size=1)), expected)


@pytest.mark.parametrize(
    "a, b, rtz, rne",
    [
        ([1.625, 1.5], [1, 1], 0x40400000, 0x40400000),
        ([1.875, 1.5], [1, 1], 0x40500000, 0x40600000),
        ([1.875, 1.875], [1, 1.125], 0x40700000, 0x40800000),
    ],
)
def test_normalization_rne_ties_and_exponent_carry(
    a: list[float], b: list[float], rtz: int, rne: int,
) -> None:
    # F=3 gives a 1/4 quantum near 3: ties 3.125->3, 3.375->3.5, 3.875->4.
    base = spec(f_bits=3, chunk_size=2)
    assert_bits(gemm_reference(operand(a), operand(b), base), rtz)
    assert_bits(gemm_reference(operand(a), operand(b), base.with_(norm_rounding="rne")), rne)


def scaled(values: list[float], scales: list[float], fmt: Format, domain: int = 2, **kwargs) -> Operand:
    return operand(
        values, FP4_E2M1, scale=torch.tensor([scales], dtype=torch.float32),
        scale_fmt=fmt, scale_kind="k", k_domain=domain, **kwargs,
    )


def gdfs(**kwargs) -> MMASpec:
    return spec(algorithm="gdfs", f_bits=35, g_bits=6, group_size=2, k_tile=4, **kwargs)


def test_nvfp4_group_significands_and_two_level_alpha() -> None:
    # Raw groups are 4 and -1; UE4M3 scales make 4*1.5*2 + (-1)*0.5*4 = 10.
    # The two alpha factors then multiply to 1/8, yielding 1.25.
    a = scaled([1, 2, 3, 4], [1.5, 0.5], UE4M3, alpha=torch.tensor(0.25))
    b = scaled([2, 1, -1, 0.5], [2, 4], UE4M3, alpha=torch.tensor(0.5))
    assert_bits(gemm_reference(a, b, gdfs()), 0x3FA00000)


def test_gdfs_general_float_scales() -> None:
    # Signed BF16 scales generalize UE4M3: 4*1.5*2 + (-1)*(-0.5)*4 = 14.
    a = scaled([1, 2, 3, 4], [1.5, -0.5], BF16)
    b = scaled([2, 1, -1, 0.5], [2, 4], BF16)
    assert_bits(gemm_reference(a, b, gdfs()), 0x41600000)


def test_gdfs_sums_groups_before_final_alignment() -> None:
    # The second group forms 2 before alignment against 16, so its contribution survives at F=3.
    a, b = operand([16, 0, 1, 1]), operand([1, 1, 1, 1])
    config = spec(algorithm="gdfs", f_bits=3, g_bits=6, group_size=2, k_tile=4)
    assert_bits(gemm_reference(a, b, config), 0x41900000)


def test_gdfs_tile_is_one_fda_not_sequential_group_fdas() -> None:
    # One tile aligns the two unit groups individually against exponent=4; both disappear.
    a, b = operand([1, 1, 16, 0]), operand([1, 1, 1, 1])
    config = spec(algorithm="gdfs", f_bits=3, g_bits=6, group_size=1, k_tile=4)
    assert_bits(gemm_reference(a, b, config), 0x41800000)


def test_mxfp4_scales_are_exponent_additions() -> None:
    # Raw groups 4 and -1, scales (2, 1/2) and (4, 2), give 4*8 - 1 = 31.
    a = scaled([1, 2, 3, 4], [2, 0.5], E8M0)
    b = scaled([2, 1, -1, 0.5], [4, 2], E8M0)
    assert_bits(gemm_reference(a, b, gdfs()), 0x41F80000)


def test_e8m0_field_zero_suppresses_nonzero_group() -> None:
    # Numerically 2^-127 * 2^127 = 1; field-0 semantics nevertheless make the group zero.
    a = scaled([1, 1], [2.0**-127], E8M0)
    b = scaled([1, 1], [2.0**127], E8M0)
    assert_bits(gemm_reference(a, b, gdfs()), 0x00000000)


@pytest.mark.parametrize("values, sa, sb, expected", [([1, 1], 0, 2, 0), ([1, -1], 2, 2, 0),
                                                     ([1, -1], float("nan"), 0, None)])
def test_gdfs_zero_and_nan_scale_precedence(values, sa, sb, expected) -> None:
    a = scaled(values, [sa], UE4M3)
    b = scaled([1, 1], [sb], UE4M3)
    out = gemm_reference(a, b, gdfs())
    if expected is None:
        assert torch.isnan(out).all()
    else:
        assert_bits(out, expected)


def test_promote_interval_uses_fp32_accumulator() -> None:
    # Independent partials 16 and 2 merge in fp32, avoiding F=3's loss of unit products against c=16.
    a, b = operand([16, 0, 1, 1]), operand([1, 1, 1, 1])
    config = spec(f_bits=3, chunk_size=2, promote_interval=2)
    assert_bits(gemm_reference(a, b, config), 0x41900000)


def test_promote_interval_applies_its_own_scale() -> None:
    # First partial 4 uses 1/2, second partial 2 uses 2: 4/2 + 2*2 = 6.
    a = scaled([4, 0, 1, 1], [0.5, 2], UE4M3)
    b = scaled([1, 1, 1, 1], [1, 1], UE4M3)
    config = spec(f_bits=3, chunk_size=2, promote_interval=2)
    assert_bits(gemm_reference(a, b, config), 0x40C00000)


def test_cutlass_epilogue_order_is_a_times_b_times_acc() -> None:
    # These three fp32 operands give 0x403b4828 only with sa*(sb*acc);
    # both (sa*sb)*acc and sb*(sa*acc) give the adjacent 0x403b4829.
    a = operand([from_bits(0x3FBDB857)], FP32, scale=torch.tensor(from_bits(0x3FDE914A)),
                scale_fmt=FP32, scale_kind="tensor")
    b = operand([1], FP32, scale=torch.tensor(from_bits(0x3F9155CA)),
                scale_fmt=FP32, scale_kind="tensor")
    assert_bits(gemm_reference(a, b, spec(algorithm="fp32_fma")), 0x403B4828)


def test_alpha_precedes_bias() -> None:
    # acc=2, alpha=1/4, bias=1 => 1.5, not (2+1)/4.
    a, b = operand([2], alpha=torch.tensor(0.25)), operand([1])
    assert_bits(gemm_reference(a, b, spec(), torch.tensor([1.0])), 0x3FC00000)


@pytest.mark.parametrize(
    "fmt, dtype, value, bias, expected",
    [
        (BF16, torch.bfloat16, 1.0, 2.0**-8, 1.0),
        (BF16, torch.bfloat16, 1 + 2.0**-7, 2.0**-8, 1 + 2.0**-6),
        (FP16, torch.float16, 1.0, 2.0**-11, 1.0),
        (FP16, torch.float16, 1 + 2.0**-10, 2.0**-11, 1 + 2.0**-9),
        (FP32, torch.float32, 1.0, 2.0**-24, 1.0),
    ],
)
def test_output_rounding_is_rne(fmt, dtype, value, bias, expected) -> None:
    config = MMASpec(algorithm="fp32_fma", out_format=fmt)
    out = gemm_reference(operand([value], FP32), operand([1], FP32), config, torch.tensor([bias]))
    assert out.dtype == dtype
    assert out.item() == expected


@pytest.mark.parametrize(
    "algorithm, pi, expected",
    [("cofda", 0, "product"), ("cofda", 2, "promote"), ("gdfs", 0, "group"),
     ("fp32_fma", 0, "operand"), ("fp64", 0, "operand")],
)
def test_auto_scale_dispatch(algorithm: str, pi: int, expected: str) -> None:
    config = spec(algorithm=algorithm, chunk_size=2, group_size=2, k_tile=4, promote_interval=pi)
    assert resolve_scale_apply(config, scaled([1, 1], [2], UE4M3), operand([1, 1])) == expected


@pytest.mark.parametrize("kind", ["tensor", "row"])
def test_auto_constant_scales_go_to_epilogue(kind: str) -> None:
    scale = torch.tensor(2.0) if kind == "tensor" else torch.tensor([[2.0]])
    a = operand([1, 1], scale=scale, scale_fmt=UE4M3, scale_kind=kind)
    assert resolve_scale_apply(spec(), a, operand([1, 1])) == "epilogue"


@pytest.mark.parametrize(
    "config",
    [spec(algorithm="int_exact"), spec(algorithm="gdfs", group_size=3, k_tile=6),
     spec(chunk_size=1, promote_interval=3), spec(scale_apply="epilogue"),
     spec(scale_apply="group"), spec(scale_apply="promote"), spec(algorithm="fp64", scale_apply="product")],
)
def test_incompatible_scale_paths_are_rejected(config: MMASpec) -> None:
    a = scaled([1, 1, 1, 1], [2, 4], UE4M3)
    with pytest.raises(ValueError, match=".+"):
        resolve_scale_apply(config, a, operand([1, 1, 1, 1]))


def test_int_exact_rejects_noninteger_operands() -> None:
    with pytest.raises(ValueError, match="[Ii]nt|integer"):
        gemm_reference(operand([1]), operand([1]), spec(algorithm="int_exact"))


def test_int64_headroom_is_validated_before_computation() -> None:
    # F=48 plus two 23-bit integer significands already exceeds 62 bits before summation.
    fmt = IntFormat("int24", 24)
    a = operand([1, 1], fmt)
    with pytest.raises(ValueError, match="headroom|int64|62|overflow"):
        gemm_reference(a, a, spec(f_bits=48, chunk_size=2))


def test_product_scales_precede_radix_truncation() -> None:
    # Combined radix=12: 9*9*10*14 >> 9 = 22, giving 2.75;
    # truncating the unscaled product first would instead give 2.5.
    a = operand([1.125], scale=torch.tensor([[1.25]]), scale_fmt=UE4M3, scale_kind="k", k_domain=1)
    b = operand([1.125], scale=torch.tensor([[1.75]]), scale_fmt=UE4M3, scale_kind="k", k_domain=1)
    assert_bits(gemm_reference(a, b, spec(f_bits=3, chunk_size=1)), 0x40300000)


def test_mixed_k_and_row_scales_keep_row_factor_in_epilogue() -> None:
    # K-scale alone gives (9*9*10 >> 6)/8 = 1.5; the row-scale then yields 1.5*1.75 = 2.625.
    a = operand([1.125], scale=torch.tensor([[1.25]]), scale_fmt=UE4M3, scale_kind="k", k_domain=1)
    b = operand([1.125], scale=torch.tensor([[1.75]]), scale_fmt=UE4M3, scale_kind="row")
    assert_bits(gemm_reference(a, b, spec(f_bits=3, chunk_size=1)), 0x40280000)


@pytest.mark.parametrize("algorithm", ["fp32_fma", "fp64"])
def test_operand_scales_round_to_fp32_before_fma_chain(algorithm: str) -> None:
    # (1+2^-23)^2 rounds to 1+2^-22 before accumulation, then cancels exactly.
    # Carrying the unrounded scaled operand into the chain would leave 2^-46.
    a = operand([1 + 2.0**-23, -(1 + 2.0**-22)], FP32,
                scale=torch.tensor([[1 + 2.0**-23, 1]]), scale_fmt=FP32, scale_kind="k", k_domain=1)
    b = operand([1, 1], FP32)
    assert_bits(gemm_reference(a, b, spec(algorithm=algorithm)), 0x00000000)


def test_gdfs_short_final_group_is_zero_padded() -> None:
    # Groups [1,1] and [1,pad] have sums 2 and 1; scales 1 and 2 give 4 in total.
    a = scaled([1, 1, 1], [1, 2], UE4M3)
    b = operand([1, 1, 1], FP4_E2M1)
    assert_bits(gemm_reference(a, b, gdfs()), 0x40800000)


def test_product_scaling_handles_significands_wider_than_int64() -> None:
    # Four fp32 factors have a 96-bit raw product. F=40 drops terms of order 2^-46,
    # leaving (1+2^-23)^4 -> 1+4*2^-23 exactly after RTZ normalization.
    x = 1 + 2.0**-23
    a = operand([x], FP32, scale=torch.tensor([[x]]), scale_fmt=FP32, scale_kind="k", k_domain=1)
    assert_bits(gemm_reference(a, a, spec(f_bits=40)), 0x3F800004)


def test_normalization_rne_carry_overflows_to_infinity() -> None:
    # max_fp32 + 2^103 is the overflow midpoint: RTZ keeps max, RNE carries into exponent 255.
    a = operand([from_bits(0x7F7FFFFF), 2.0**103], FP32)
    b = operand([1, 1], FP32)
    config = spec(f_bits=25, chunk_size=2)
    assert_bits(gemm_reference(a, b, config), 0x7F7FFFFF)
    assert_bits(gemm_reference(a, b, config.with_(norm_rounding="rne")), 0x7F800000)


@pytest.mark.parametrize(
    "values, sa, sb, expected",
    [
        ([1, -1], float("inf"), 1, 0),
        ([float("inf"), 0], 0, 1, 0),
        ([1, 1], 0, float("inf"), 0),
        ([1, -1], float("nan"), 1, None),
        ([float("nan"), 0], 0, 1, None),
    ],
)
def test_gdfs_group_zero_precedes_scale_infinity_but_not_nan(values, sa, sb, expected) -> None:
    # §4.5 defines zero from the group or either scale; actual input NaNs still take precedence.
    a = operand(values, FP32, scale=torch.tensor([[sa]]), scale_fmt=FP32, scale_kind="k", k_domain=2)
    b = operand([1, 1], FP32, scale=torch.tensor([[sb]]), scale_fmt=FP32, scale_kind="k", k_domain=2)
    out = gemm_reference(a, b, gdfs())
    if expected is None:
        assert torch.isnan(out).all()
    else:
        assert_bits(out, expected)


@pytest.mark.parametrize("group_size, k_tile, g_bits", [(4096, 4096, 48), (128, 1024, 6)])
def test_gdfs_validates_group_and_tile_headroom(group_size: int, k_tile: int, g_bits: int) -> None:
    # The first configuration exceeds group headroom; the second exceeds scaled tile headroom.
    a = scaled([1], [1], FP32, domain=1)
    config = spec(algorithm="gdfs", f_bits=48, g_bits=g_bits, group_size=group_size, k_tile=k_tile)
    with pytest.raises(ValueError, match="headroom|int64|62|overflow"):
        gemm_reference(a, a, config)


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("format_name", ["e8m0", "e8m0:bias=128", "e7m0:bias=127"])
@pytest.mark.parametrize("exponent", [-127, -126])
def test_pow2_field_zero_is_specific_to_standard_e8m0(
    algorithm: str, side: str, format_name: str, exponent: int,
) -> None:
    from tricast.formats import get_format

    # Eight exact unit products give 2^(exponent+3); only standard E8M0's
    # field zero suppresses them. With bias 128, 2^-127 is ordinary field one.
    fmt = get_format(format_name)
    values = [1.0] * 8
    scaled_op = scaled(values, [2.0**exponent], fmt, domain=8)
    unscaled_op = operand(values, FP4_E2M1)
    a, b = (scaled_op, unscaled_op) if side == "a" else (unscaled_op, scaled_op)
    config = spec(algorithm=algorithm, f_bits=25, chunk_size=8,
                  g_bits=6, group_size=8, k_tile=8)
    expected = 0 if format_name == "e8m0" and exponent == -127 else (exponent + 130) << 23
    assert_bits(gemm_reference(a, b, config), expected)
