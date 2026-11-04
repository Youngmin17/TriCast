"""The per-layer profile (app/layer_profile.py) on a tiny torch model: the pruning masks keep what they say,
the streaming input moments equal the direct ones, outlier channels and pre-sparse weights show up, and the
weight histogram accounts for every value. CPU only, no transformers."""

from __future__ import annotations

import torch
from torch import nn

from app import layer_profile as lp


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(50, 16)
        self.up = nn.Linear(16, 32, bias=False)
        self.down = nn.Linear(32, 16, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(self.embed(ids))))


def test_masks_keep_half_of_every_row_and_two_of_every_four() -> None:
    torch.manual_seed(1)
    w = torch.randn(8, 16)
    kept = lp.masks(w, torch.ones(16))
    for name in ("magnitude_50", "wanda_50"):
        assert (kept[name].sum(1) == 8).all()
    for name in ("magnitude_2of4", "wanda_2of4"):
        assert (kept[name].reshape(8, -1, 4).sum(2) == 2).all()
    assert torch.equal(kept["magnitude_50"], kept["wanda_50"])  # equal channel norms: the same scores
    norms = torch.ones(16)
    norms[0] = 1e6  # a dominant input channel survives every Wanda mask
    assert lp.masks(w, norms)["wanda_2of4"][:, 0].all() and lp.masks(w, norms)["wanda_50"][:, 0].all()


def test_profile_reports_what_the_layers_hold() -> None:
    model = Tiny()
    with torch.no_grad():
        model.up.weight[:, 3] *= 30  # an outlier input column of up
        sparse = torch.randn(16, 32)
        sparse[:, ::2] = 0  # half of every row of down is already zero
        model.down.weight.copy_(sparse)
    windows = [torch.randint(0, 50, (1, 24), generator=torch.Generator().manual_seed(s)) for s in (2, 3)]
    seen = []
    handle = model.down.register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].reshape(-1, 32).double()))
    with torch.no_grad():
        for ids in windows:
            model(ids)
    handle.remove()
    result = lp.profile(model, ["up", "down"], windows)
    down = result["down"]
    assert down["sparsity"]["magnitude_50"] == 0.0 and down["weight"]["zero_fraction"] == 0.5
    assert all(0 <= value <= 1.5 for value in result["up"]["sparsity"].values())
    assert result["up"]["weight"]["col_absmax_ratio"] > 6
    x = torch.cat(seen)
    centred = x - x.mean()
    direct = (centred.pow(4).mean() / centred.pow(2).mean().pow(2) - 3).item()
    assert abs(down["activation"]["kurtosis"] - direct) < 1e-9
    assert abs(down["activation"]["rms"] - x.pow(2).mean().sqrt().item()) < 1e-12
    assert down["activation"]["absmax"] == x.abs().max().item()
    histogram = result["up"]["weight"]["histogram"]
    total = sum(histogram["fraction"]) + histogram["below"] + histogram["above"]
    assert abs(total - 1) < 1e-12 and len(histogram["fraction"]) == lp.BINS


def test_a_gaussian_weight_has_near_zero_kurtosis_and_the_gaussian_tail() -> None:
    torch.manual_seed(4)
    stats = lp._weight_stats(torch.randn(512, 512))
    assert abs(stats["kurtosis"]) < 0.05 and abs(stats["outlier_3sigma"] - 0.0027) < 0.0006
    assert stats["outlier_6sigma"] == 0.0
