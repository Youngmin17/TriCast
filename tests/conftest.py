"""Shared pytest setup.

Markers (see pyproject.toml): ``gpu`` tests are skipped without CUDA + triton,
``oracle`` tests skip themselves via ``pytest.importorskip`` when the oracle package
is missing (microxcaling: put a clone on PYTHONPATH, it imports as ``mx``).
"""

from __future__ import annotations

import pytest
import torch


def _triton_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


GPU_OK = _triton_gpu_available()


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU with triton")
    for item in items:
        if "gpu" in item.keywords and not GPU_OK:
            item.add_marker(skip_gpu)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator().manual_seed(42)


def wide_fp32(n: int, gen: torch.Generator, lo_exp: int = 100, hi_exp: int = 150) -> torch.Tensor:
    """Random finite fp32 values with exponents spread over ``[lo_exp, hi_exp)`` (biased),
    both signs, every fraction pattern — a wide net for rounding bugs."""
    sign = torch.randint(0, 2, (n,), generator=gen) << 31
    exp = torch.randint(lo_exp, hi_exp, (n,), generator=gen) << 23
    frac = torch.randint(0, 2**23, (n,), generator=gen)
    return (sign | exp | frac).to(torch.int32).view(torch.float32)


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal values with NaN == NaN (sign of zero ignored)."""
    a, b = a.float(), b.float()
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
