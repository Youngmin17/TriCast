"""Per-layer profile of a model's decoder Linears: what a layer looks like before any method touches it,
so the tune loop's heuristics and its agent read the same facts (the ledger turns every number into a record
with its environment and the layer documents ``tricast.rag.search_evidence`` retrieves).

    python -m app.layer_profile --model Qwen/Qwen3-0.6B --revision <sha> --device cuda --out profile.json
        [--split train --windows 8 --seqlen 512 --seed 42]

For each Linear, on the unquantized model and WikiText-2 windows drawn as tricast's calibration draws them:
- ``weight``: mean, std, absmax, excess kurtosis, the fraction beyond 3 and 6 standard deviations, the
  largest row and column absmax over their medians (output- and input-channel spread), the fraction of exact
  zeros, and a histogram of (w - mean) / std over [-6, 6] in 24 bins plus the two tails;
- ``activation`` (the Linear's input over every window position): absmax, rms, excess kurtosis, the largest
  input-channel absmax over the median, the fraction of channels above 6 x the median, and channel-absmax
  quantiles (p50, p90, p99, max) over the median;
- ``sparsity``: the output relative error ||X (W - W_pruned)^T||_F / ||X W^T||_F on the same windows for
  four unstructured-50% / 2:4 masks: magnitude (|W|) and Wanda (|W| x the input channel's L2 norm), each
  keeping the largest scores per output row (2:4: the 2 largest of every 4 consecutive inputs). Two passes:
  channel norms first, then the errors with the masks fixed.
Everything is accumulated in float64; the model runs in its own dtype (bfloat16 on CUDA, float32 on CPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "tricast.layer_profile/1"
BINS = 24
SPAN = 6.0
STAT_NOTES = {
    "kurtosis": "excess kurtosis E[(x - mean)^4] / std^4 - 3 (0 for a Gaussian)",
    "outlier_3sigma": "fraction of values with |x - mean| > 3 std (Gaussian: 0.0027)",
    "row_absmax_ratio": "max over output rows of the row absmax / their median",
    "col_absmax_ratio": "max over input columns of the column absmax / their median",
    "channel_absmax_ratio": "max over input channels of the activation absmax / their median",
    "outlier_channels": "fraction of input channels whose activation absmax exceeds 6 x the median",
    "sparsity": "output relative error ||X (W - W_pruned)^T||_F / ||X W^T||_F on the profile windows",
}


def _weight_stats(w: Any) -> dict[str, Any]:
    import torch

    x = w.double()
    mean, std = x.mean(), x.std(unbiased=False)
    z = (x - mean) / std
    rows, cols = x.abs().amax(1), x.abs().amax(0)
    edges = torch.linspace(-SPAN, SPAN, BINS + 1, dtype=torch.float64, device=x.device)
    inside = z[(z >= -SPAN) & (z <= SPAN)]
    counts = torch.histc(inside, bins=BINS, min=-SPAN, max=SPAN)
    n = x.numel()
    return {"mean": mean.item(), "std": std.item(), "absmax": x.abs().max().item(),
            "kurtosis": (z.pow(4).mean() - 3).item(), "outlier_3sigma": (z.abs() > 3).double().mean().item(),
            "outlier_6sigma": (z.abs() > 6).double().mean().item(),
            "row_absmax_ratio": (rows.max() / rows.median()).item(),
            "col_absmax_ratio": (cols.max() / cols.median()).item(),
            "zero_fraction": (x == 0).double().mean().item(),
            "histogram": {"edges": [round(edge, 6) for edge in edges.tolist()],
                          "fraction": (counts / n).tolist(), "below": (z < -SPAN).double().mean().item(),
                          "above": (z > SPAN).double().mean().item()}}


class _Inputs:
    """Streaming float64 moments and per-channel absmax / sum of squares of one Linear's input."""

    def __init__(self, features: int, device: Any) -> None:
        import torch

        self.absmax = torch.zeros(features, dtype=torch.float64, device=device)
        self.square = torch.zeros(features, dtype=torch.float64, device=device)
        self.moments = torch.zeros(4, dtype=torch.float64, device=device)
        self.count = 0

    def add(self, x: Any) -> None:
        import torch

        flat = x.reshape(-1, x.shape[-1]).double()
        self.absmax = torch.maximum(self.absmax, flat.abs().amax(0))
        self.square += flat.pow(2).sum(0)
        self.moments += torch.stack([flat.sum(), flat.pow(2).sum(), flat.pow(3).sum(), flat.pow(4).sum()])
        self.count += flat.numel()

    def result(self) -> dict[str, Any]:
        s1, s2, s3, s4 = (value / self.count for value in self.moments.tolist())
        var = s2 - s1 * s1
        fourth = s4 - 4 * s1 * s3 + 6 * s1 * s1 * s2 - 3 * s1 ** 4
        median = self.absmax.median()
        quantiles = self.absmax.quantile(self.absmax.new_tensor([0.5, 0.9, 0.99])) / median
        ratio = (self.absmax.max() / median).item()
        return {"absmax": self.absmax.max().item(), "rms": s2 ** 0.5, "kurtosis": fourth / var ** 2 - 3,
                "channel_absmax_ratio": ratio,
                "outlier_channels": (self.absmax > 6 * median).double().mean().item(),
                "channel_absmax_quantiles": {"p50": quantiles[0].item(), "p90": quantiles[1].item(),
                                             "p99": quantiles[2].item(), "max": ratio}}


def masks(w: Any, channel_norm: Any) -> dict[str, Any]:
    """Keep-masks of the four pruning rules (True = kept); scores are tied by the lower input index."""
    import torch

    magnitude = w.abs().double()
    wanda = magnitude * channel_norm.double()[None, :]
    out = {}
    for name, score in (("magnitude", magnitude), ("wanda", wanda)):
        keep = torch.zeros_like(score, dtype=torch.bool)
        top = score.topk(score.shape[1] // 2, dim=1, sorted=False).indices
        keep.scatter_(1, top, True)
        out[f"{name}_50"] = keep
        groups = score.reshape(score.shape[0], -1, 4)
        top4 = groups.topk(2, dim=2, sorted=False).indices
        keep24 = torch.zeros_like(groups, dtype=torch.bool).scatter_(2, top4, True)
        out[f"{name}_2of4"] = keep24.reshape(score.shape)
    return out


def profile(model: Any, names: Sequence[str], windows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    import torch

    modules = {name: model.get_submodule(name) for name in names}
    inputs = {name: _Inputs(module.in_features, module.weight.device) for name, module in modules.items()}

    def run(hook_for: Any) -> None:
        handles = [module.register_forward_pre_hook(hook_for(name)) for name, module in modules.items()]
        try:
            with torch.no_grad():
                for ids in windows:
                    model(ids)
        finally:
            for handle in handles:
                handle.remove()

    run(lambda name: lambda _module, args: inputs[name].add(args[0]))
    kept = {name: masks(module.weight.detach(), inputs[name].square.sqrt())
            for name, module in modules.items()}
    errors = {name: {rule: [0.0, 0.0] for rule in kept[name]} for name in names}

    def measure(name: str) -> Any:
        weight = modules[name].weight.detach().double()

        def hook(_module: Any, args: Any) -> None:
            x = args[0].reshape(-1, args[0].shape[-1]).double()
            reference = x @ weight.T
            norm = reference.pow(2).sum().item()
            for rule, keep in kept[name].items():
                errors[name][rule][0] += (x @ (weight * ~keep).T).pow(2).sum().item()
                errors[name][rule][1] += norm
        return hook

    run(measure)
    return {name: {"weight": _weight_stats(modules[name].weight.detach()),
                   "activation": inputs[name].result(),
                   "sparsity": {rule: (num / den) ** 0.5 if den else None
                                for rule, (num, den) in errors[name].items()}}
            for name in names}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.layer_profile", description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from app import autotune as at
    from tricast.calibration import _dataset_texts, _windows
    from tricast.eval.envinfo import capture_env

    started = time.perf_counter()
    dtype = "bfloat16" if args.device == "cuda" else "float32"
    at._seed(args.seed)
    env = capture_env(args.model, {"seed": args.seed, "dtype": dtype, "device": args.device})
    model, tokenizer = at._load(args.model, dtype, args.device, args.revision or env.get("model_sha"))
    actual = getattr(model.config, "_commit_hash", None)  # the revision loaded, as app.autotune.open_session
    if actual and env.get("model_kind") != "local":
        env["model_sha"] = actual
    layers, blocks = at.model_layers(model)
    texts = _dataset_texts("wikitext2", args.split)
    windows = _windows(tokenizer, texts, None, args.windows, args.seqlen, args.seed)
    digest = hashlib.sha256()
    for ids in windows:
        digest.update(ids.cpu().contiguous().numpy().astype("<i8").tobytes())
    stats = profile(model, [layer.name for layer in layers], windows)
    meta = {"model_id": args.model, "revision": args.revision, "model_sha": env.get("model_sha"),
            "dtype": dtype, "device": args.device, "gpu_names": env.get("gpu_names"), "windows": args.windows,
            "seqlen": args.seqlen, "seed": args.seed, "source": {"dataset": "wikitext2", "split": args.split},
            "dataset_fingerprint": digest.hexdigest(), "git_sha": env.get("git_sha"),
            "git_dirty": env.get("git_dirty"), "model_blocks": blocks, **at.provenance(),
            "wall_s": time.perf_counter() - started, "env": env}
    rows = [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
             "in_features": layer.in_features, "out_features": layer.params // layer.in_features,
             **stats[layer.name]} for layer in layers]
    payload = {"schema": SCHEMA, "status": "complete", "stat_notes": STAT_NOTES, "meta": meta, "layers": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"LAYER_PROFILE_DONE layers={len(rows)} out={args.out} wall_s={meta['wall_s']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
