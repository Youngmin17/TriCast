"""Z-Image-Turbo per-layer weight bit-widths: the offline demo dataset of the tuning page (text to image).

"기존 방법" = one weight quantization on every Linear of the diffusion transformer (uniform INT8 / INT4 /
INT3) or a common rule (INT8 on the embedders, the adaLN modulations and the final layer, INT4 elsewhere);
"우리 방법" = TriCast Auto-tune: a method per Linear chosen by the exact multiple-choice knapsack
(app.autotune.solve_mckp) over a measured per-layer table, at exactly the average bits of the setup it is
compared with.

Stages (each ends with one log marker: ZT_<STAGE>_DONE, or ZT_FAIL stage=<stage>):
  refs      text embeddings of every prompt (the Qwen3 text encoder in FP32), and the transformer inputs of
            the 8 forwards of each table prompt's unquantized generation -> refs/
  table     D(layer, method): one Linear quantized at a time, every recorded forward replayed ->
            table_<k>of<n>.jsonl (--shard k/n splits the layers over GPUs; a rerun skips the layers already
            written)
  merge     the shards -> table.json
  solve     knapsack at the reference setups' average bits -> solutions.json
  evaluate  per setup: the 8-forward generation of every evaluation prompt, PSNR / SSIM of the decoded
            image against the unquantized model's, and the measured D -> eval_<setup>.json, images/<setup>/
  export    app/web/demo/autotune/zimage.json + app/web/demo/media/zimage/

Numerics: weight-only fake-quant (tricast.quant.api.fake_quant). INT8: per output row, symmetric, fp32
scale. INT4 / INT3 / INT2: per 128 inputs, fp16 scale and an integer zero point (the formats of the LLM
tables' GPTQ methods, here with round-to-nearest, no calibration). The transformer computes in --dtype: FP32
by default, the checkpoint's own precision (on a V100 the FP16 run gave non-finite latents and BF16 is only
emulated); a fake-quantized weight is rounded to that dtype like every other weight. The text encoder and
the VAE stay unquantized. Linears whose input width is not a multiple of 128 (the 64-wide patch embedder)
stay unquantized and count with the other parameters. Run on one GPU through gpu_cap; seed 42.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import socket
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "tricast.image_tune/1"
TABLE_SCHEMA = "tricast.image_tune.table/1"
REPO = "Tongyi-MAI/Z-Image-Turbo"
REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
SEED = 42
# 8 transformer forwards, the model card's 8 NFEs: in diffusers 0.39 the default sigmas are
# linspace(1, 1/N, N), so N steps call the transformer N times (the card's num_inference_steps=9 counts an
# older schedule)
STEPS = 8
GROUP = 128
# element bits; the order is the knapsack's option order
# "bf16" = the weight left as it is: the checkpoint stores FP32 tensors whose every value is a BF16 value (all
# 521 tensors checked 2026-11-14; load_transformer checks again), so it costs 16 bits and changes nothing
METHODS = {"bf16": 16, "int8": 8, "int4": 4, "int3": 3, "int2": 2}
QUANT_METHODS = ("int8", "int4", "int3", "int2")
UNIFORM = {"u_int8": "int8", "u_int4": "int4", "u_int3": "int3"}
HEURISTIC = "heur_8_4"
AUTO = {"auto_iso_u4": "u_int4", "auto_iso_h84": HEURISTIC, "auto_iso_u3": "u_int3"}  # auto -> size reference
SETUPS = ("native", *UNIFORM, HEURISTIC, *AUTO)
LABELS = {
    "u_int8": "INT8 균일",
    "u_int4": "INT4 균일",
    "u_int3": "INT3 균일",
    HEURISTIC: "INT8 임베더·adaLN·마지막 층 + INT4",
    "auto_iso_u4": "Auto-tune · INT4 균일 크기",
    "auto_iso_h84": "Auto-tune · INT8/4 경험칙 크기",
    "auto_iso_u3": "Auto-tune · INT3 균일 크기",
}
# the common rule's INT8 layers: what a diffusion transformer's quantizer usually keeps wider
HEURISTIC_PATTERNS = ("t_embedder.", "cap_embedder.", "adaLN_modulation.", "all_final_layer.")
QUANTIZATION = (
    "weight-only fake-quant (RTN): INT8 per output row (symmetric, fp32 scale); INT4/3/2 per 128 inputs "
    "with an fp16 scale and an integer zero point; activations, text encoder and VAE unquantized"
)
OBJECTIVE = (
    "D = sum over the recorded forwards of ||v_q - v||^2 / sum ||v||^2, where v is the unquantized "
    "transformer's output (the flow velocity) on the inputs it received during the 8-forward generation of "
    "each table prompt, and v_q the quantized one on the same inputs. Table: one Linear quantized at a "
    "time; the knapsack sums D over layers (additive assumption); the measured D quantizes every layer of "
    "a setup at once"
)
SIZE_RULE = (
    "bytes = sum over the quantizable Linear weights of params * bits / 8, bits = 8 + 32/K for INT8 (one "
    "fp32 scale per output row, K = in_features) and b + (16 + b)/128 for INT4/3/2 (an fp16 scale and a "
    "b-bit zero point per 128), 16 for unquantized; plus every other transformer parameter at 2 bytes. "
    "Original = every transformer parameter at 2 bytes (the BF16 model). The text encoder and the VAE are "
    "not counted"
)
# prompts written for this dataset (no person, brand or work is named); the table and evaluation sets do
# not overlap
TABLE_PROMPTS = (  # --table-prompts N uses the first N (a product photo, a busy scene, lettering, a face)
    "A ceramic teapot on a wooden table next to a window, soft afternoon light, product photograph",
    "A street food stall at night with steam rising from noodle bowls and glowing signs, cinematic photo",
    "A shop sign that reads 'OPEN DAILY 9-6' in white letters on a green board",
    "Close-up portrait of an old fisherman in a knitted cap, natural light, photograph",
    "An aerial view of a winding river through an autumn forest, detailed landscape photograph",
    "A watercolor illustration of a fox sleeping under a maple tree",
    "A futuristic electric train crossing a bridge over a bay at sunset, wide angle",
    "A flat vector icon set of kitchen tools on a pastel background",
)
DEMO_PROMPTS = (
    "A golden retriever puppy sitting in fresh snow, shallow depth of field, photograph",
    "A minimal poster with the bold title 'TriCast' and the Korean subtitle '양자화 실험실', blue and white",
    "A cozy library with tall wooden shelves and a reading lamp, warm light, detailed illustration",
)
EVAL_PROMPTS = DEMO_PROMPTS + (
    "A bowl of ramen with a soft-boiled egg and green onions, top view, food photograph",
    "A lighthouse on a rocky cliff during a storm, dramatic waves, oil painting",
    "A red vintage car parked on a cobblestone street, morning fog, photograph",
    "A hand-drawn map of an imaginary island with mountains, rivers and a compass rose",
    "A glass of iced coffee on a cafe counter with condensation drops, macro photograph",
    "A snowy mountain village at dusk with warm lights in the windows, digital art",
    "A chalkboard menu that says 'SOUP OF THE DAY' with a drawing of a bowl",
    "A hummingbird hovering near a purple flower, high-speed photograph",
    "A pixel art scene of a small farm with a barn, cows and a windmill",
    "A modern living room with a large window, a grey sofa and indoor plants, interior photograph",
    "An astronaut planting a small tree on the surface of the moon, illustration",
    "A stack of colorful books with a pair of round glasses on top, studio photograph",
    "A traditional Korean house with a tiled roof in spring with cherry blossoms, photograph",
    "A robot made of cardboard boxes waving hello, children's book illustration",
    "A plate of sushi on a black slate board, restaurant photograph",
    "Neon text 'NIGHT MARKET' above a crowded alley, rainy night photo",
    "A sailing boat on a calm lake reflecting the mountains, early morning, photograph",
    "A close-up of a mechanical watch movement with gears, macro photograph",
    "A cat wearing a tiny wizard hat reading a book, cute 3D render",
    "A desert road leading to distant red rock formations, wide landscape photograph",
    "A birthday cake with candles that spell '25', bright party background",
)


class TuneError(RuntimeError):
    """A failed protocol check: the stage stops and prints ZT_FAIL."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TuneError(message)


# ---------- pure accounting ----------
@dataclass(frozen=True)
class Layer:
    name: str
    params: int
    k: int  # in_features
    out: int


def stored_bits(method: str, k: int) -> Fraction:
    """Stored bits per weight: the element plus its share of the scale (and zero point)."""
    bits = METHODS[method]
    if method == "bf16":
        return Fraction(bits)
    if method == "int8":
        return bits + Fraction(32, k)
    return bits + Fraction(16 + bits, GROUP)


def layer_cost(layer: Layer, method: str) -> Fraction:
    return layer.params * stored_bits(method, layer.k)


def total_bits(layers: Sequence[Layer], assignment: Mapping[str, str]) -> Fraction:
    require(
        set(assignment) == {layer.name for layer in layers}, "assignment must name every quantizable layer"
    )
    return sum((layer_cost(layer, assignment[layer.name]) for layer in layers), Fraction(0))


def avg_bits(layers: Sequence[Layer], assignment: Mapping[str, str]) -> Fraction:
    return total_bits(layers, assignment) / sum(layer.params for layer in layers)


def model_bytes(layers: Sequence[Layer], assignment: Mapping[str, str], other_params: int) -> int:
    return math.ceil((total_bits(layers, assignment) + 16 * other_params) / 8)


def original_bytes(layers: Sequence[Layer], other_params: int) -> int:
    return 2 * (sum(layer.params for layer in layers) + other_params)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(text: str) -> Fraction:
    numerator, denominator = text.split("/")
    return Fraction(int(numerator), int(denominator))


def method_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(assignment.values())
    return {method: counts.get(method, 0) for method in METHODS}


def heuristic_names(layers: Sequence[Layer]) -> list[str]:
    return [layer.name for layer in layers if any(p in layer.name for p in HEURISTIC_PATTERNS)]


def baseline_assignment(setup: str, layers: Sequence[Layer]) -> dict[str, str]:
    if setup == "native":
        return {layer.name: "bf16" for layer in layers}
    if setup in UNIFORM:
        return {layer.name: UNIFORM[setup] for layer in layers}
    require(setup == HEURISTIC, f"{setup} is not a uniform/heuristic setup")
    keep = set(heuristic_names(layers))
    return {layer.name: "int8" if layer.name in keep else "int4" for layer in layers}


def predicted_d(assignment: Mapping[str, str], table: Mapping[str, Mapping[str, float]]) -> float:
    return sum(0.0 if method == "bf16" else float(table[name][method]) for name, method in assignment.items())


@dataclass(frozen=True)
class Solution:
    choice: tuple[int, ...]
    cost: int  # exact stored bits of the choice
    value: float  # its predicted D
    bound: float  # the solver's proven lower bound on the optimum
    gap: float  # value - bound


def solve(
    layers: Sequence[Layer], table: Mapping[str, Mapping[str, float]], budget_bits: Fraction
) -> tuple[dict[str, str], Solution]:
    """Minimum predicted D with the Linear weights' total stored bits <= ``budget_bits``: the multiple-choice
    knapsack as an integer program (scipy milp / HiGHS, mip_rel_gap 0). The exact DP of app.autotune does not
    fit here (275 layers x 2e8 budget units). The costs are whole numbers, so the chosen assignment's cost is
    re-checked exactly, and the solver's dual bound must equal the value (gap <= 1e-9 relative)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    methods = list(METHODS)
    costs = [[layer_cost(layer, method) for method in methods] for layer in layers]
    require(all(c.denominator == 1 for row in costs for c in row), "stored bits are not whole numbers")
    require(budget_bits.denominator == 1, "the budget is not a whole number of bits")
    unit = math.gcd(*[int(c) for row in costs for c in row], int(budget_bits))
    values = np.array(
        [[0.0 if m == "bf16" else float(table[layer.name][m]) for m in methods] for layer in layers]
    )
    require(np.isfinite(values).all() and (values >= 0).all(), "table values must be finite and >= 0")
    scale = 1.0 / max(float(values.max()), 1e-300)  # objective of order 1 for the solver's tolerances
    n, m = len(layers), len(methods)
    one_each = np.zeros((n, n * m))
    for i in range(n):
        one_each[i, i * m : (i + 1) * m] = 1
    size = np.array([int(c) // unit for row in costs for c in row], dtype=np.float64)
    result = milp(
        c=(values * scale).ravel(),
        integrality=np.ones(n * m),
        bounds=Bounds(0, 1),
        constraints=[
            LinearConstraint(one_each, 1, 1),
            LinearConstraint(size[None, :], -np.inf, int(budget_bits) // unit),
        ],
        options={"mip_rel_gap": 0.0, "disp": False},
    )
    require(result.status == 0, f"no optimal knapsack solution ({result.message})")
    x = result.x.reshape(n, m)
    choice = tuple(int(j) for j in np.argmax(x, axis=1))
    require(np.allclose(x, np.eye(m)[list(choice)], atol=1e-6), "the solver returned a non-integral choice")
    cost = sum(int(costs[i][j]) for i, j in enumerate(choice))
    require(cost <= budget_bits, "knapsack exceeded its budget")
    value = float(sum(values[i, j] for i, j in enumerate(choice)))
    bound = float(getattr(result, "mip_dual_bound", result.fun)) / scale
    require(value - bound <= 1e-9 * max(1.0, value), f"optimality not proven (value {value}, bound {bound})")
    assignment = {layer.name: methods[j] for layer, j in zip(layers, choice, strict=True)}
    return assignment, Solution(choice, cost, value, bound, value - bound)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def sign_flip_p(diffs: Sequence[float]) -> float:
    """Exact two-sided sign-flip p of paired differences: the share of the 2^n sign patterns whose |sum| is at
    least the observed one (zeros count as either sign). n <= 24 (16.7 M patterns, enumerated in blocks)."""
    import numpy as np

    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    require(1 <= n <= 24, "sign-flip enumeration needs 1 <= n <= 24")
    observed = abs(d.sum()) - 1e-12 * max(1.0, float(np.abs(d).sum()))
    low = min(n, 20)  # 2^20 patterns per block
    bits = ((np.arange(1 << low)[:, None] >> np.arange(low)) & 1).astype(np.float64) * 2 - 1
    partial = bits @ d[:low]
    hits = 0
    for high in range(1 << (n - low)):
        signs = np.array([1.0 if (high >> j) & 1 else -1.0 for j in range(n - low)])
        hits += int((np.abs(partial + float(signs @ d[low:])) >= observed).sum())
    return hits / float(1 << n)


def psnr(a: Any, b: Any) -> float | None:
    """PSNR in dB of two uint8 images (None when they are equal: JSON has no infinity)."""
    import numpy as np

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse == 0 else 10 * math.log10(255.0**2 / mse)


def ssim(a: Any, b: Any) -> float:
    """Mean SSIM of two uint8 RGB images [H, W, 3]: 11x11 Gaussian window (sigma 1.5), K1 0.01, K2 0.03,
    L 255,
    per channel then averaged (valid region, float64)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(a.astype(np.float64)).permute(2, 0, 1).unsqueeze(1)
    y = torch.from_numpy(b.astype(np.float64)).permute(2, 0, 1).unsqueeze(1)
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x**2
    syy = F.conv2d(y * y, w) - mu_y**2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


# ---------- environment ----------
def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_env() -> dict[str, Any]:
    import torch

    from tricast.eval.envinfo import source_identity

    identity = source_identity()
    import diffusers

    return {
        "git_sha": identity["git_sha"],
        "git_dirty": identity["git_dirty"],
        "script_sha256": sha256_file(Path(__file__)),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": _version("transformers"),
        "model": REPO,
        "revision": REVISION,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def setup_torch() -> None:
    import os

    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}


def torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, DTYPES[name])


# ---------- model ----------
def model_dir() -> str:
    """The checkpoint's local snapshot at REVISION (offline, from_pretrained with a repo id and a revision
    asks the hub API)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, local_files_only=True)


def load_transformer(dtype: str) -> Any:
    from diffusers import ZImageTransformer2DModel

    # straight onto the GPU: a CPU copy first commits 25 GB of host memory per process (FP32), which a strict
    # overcommit host refuses when two jobs load at once (MemoryError on athens, 2026-11-14)
    model = ZImageTransformer2DModel.from_pretrained(
        model_dir(), subfolder="transformer", torch_dtype=torch_dtype(dtype), device_map="cuda"
    )
    model.eval()
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            require(
                torch.equal(param.bfloat16().to(param.dtype), param), f"{name} holds values that are not BF16"
            )
    return model


def linear_layers(model: Any) -> tuple[list[Layer], dict[str, Any], int]:
    """The quantizable Linears (input width a multiple of GROUP) in module order, and the count of every
    other parameter."""
    import torch

    layers, modules = [], {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.in_features % GROUP == 0:
            layers.append(Layer(name, module.weight.numel(), module.in_features, module.out_features))
            modules[name] = module
    quantized = {id(m.weight) for m in modules.values()}
    others = sum(p.numel() for p in model.parameters() if id(p) not in quantized)
    return layers, modules, others


def weight_spec(method: str) -> Any:
    """The QuantSpec of a method (None for bf16). Every method dequantizes to FP32 exactly: (q - z) * s, with
    no rounding onto BF16 (the "dequant" default) between the codes and the FP32 computation."""
    from dataclasses import replace

    from tricast.quant.spec import FP16, FP32, QuantSpec, ScaleSpec, get_scheme

    if method == "bf16":
        return None
    if method == "int8":
        return replace(get_scheme("int8_row"), mma_input="dequant", dequant_format=FP32)
    return QuantSpec(
        f"uint{METHODS[method]}",
        "group",
        group_size=GROUP,
        scale=ScaleSpec(FP16),
        zero_point="int",
        mma_input="dequant",
        dequant_format=FP32,
    )


def fake_quant_weight(weight: Any, method: str) -> Any:
    """[out, in] weight -> its fake-quantized copy in the weight's dtype (computed in fp32)."""
    from tricast.quant.api import fake_quant

    spec = weight_spec(method)
    if spec is None:
        return weight.detach().clone()
    return fake_quant(weight.detach().float(), spec, backend="auto").to(weight.dtype)


def weights_digest(modules: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(modules):
        digest.update(name.encode())
        digest.update(modules[name].weight.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def quantized(modules: Mapping[str, Any], assignment: Mapping[str, str]) -> Iterator[None]:
    """Replace the assigned Linears' weights by their fake-quantized copies (originals kept on the CPU)."""
    import torch

    saved = {}
    try:
        with torch.no_grad():
            for name, method in assignment.items():
                if method == "bf16":
                    continue
                weight = modules[name].weight
                saved[name] = weight.detach().to("cpu", copy=True)
                weight.copy_(fake_quant_weight(weight, method))
        yield
    finally:
        with torch.no_grad():
            for name, weight in saved.items():
                modules[name].weight.copy_(weight.to(modules[name].weight.device))


def replay(model: Any, refs: Mapping[str, Any], embeds: Sequence[Any], batch: int) -> list[Any]:
    """The transformer's outputs (fp32, CPU) on every recorded forward, in batches."""
    import torch

    dtype = next(model.parameters()).dtype
    outs = []
    n = refs["latents"].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = range(start, min(n, start + batch))
            x = [refs["latents"][i].to("cuda", dtype) for i in idx]
            t = refs["t"][list(idx)].to("cuda")
            caps = [embeds[int(refs["cap"][i])].to("cuda", dtype) for i in idx]
            out = model(x, t, caps, return_dict=False)[0]
            outs.extend(o.float().cpu() for o in out)
    return outs


def relative_mse(outs: Sequence[Any], base: Sequence[Any]) -> float:
    """sum ||a - b||^2 / sum ||b||^2 in float64; refused unless finite."""
    num = sum(float(((a.double() - b.double()) ** 2).sum()) for a, b in zip(outs, base, strict=True))
    den = sum(float((b.double() ** 2).sum()) for b in base)
    require(math.isfinite(num) and math.isfinite(den) and den > 0, f"non-finite D ({num} / {den})")
    return num / den


class Generator:
    """The pipeline without its text encoder: generation from stored embeddings, fixed latents per seed."""

    def __init__(self, model: Any, size: int) -> None:
        import torch
        from diffusers import AutoencoderKL, ZImagePipeline

        self.vae = AutoencoderKL.from_pretrained(model_dir(), subfolder="vae", torch_dtype=torch.float32).to(
            "cuda"
        )
        self.pipe = ZImagePipeline.from_pretrained(
            model_dir(),
            text_encoder=None,
            tokenizer=None,
            transformer=model,
            vae=self.vae,
            torch_dtype=next(model.parameters()).dtype,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.size = size

    def latents(self, seed: int) -> Any:
        import torch

        g = torch.Generator("cpu").manual_seed(seed)
        return torch.randn((1, 16, self.size // 8, self.size // 8), generator=g, dtype=torch.float32)

    def __call__(self, embed: Any, seed: int, record: list | None = None) -> tuple[Any, Any]:
        """(final latent fp32 CPU, uint8 image [H, W, 3]); ``record`` collects the transformer's inputs."""
        import torch

        model = self.pipe.transformer
        hook = None
        if record is not None:

            def pre_hook(_module: Any, args: tuple, _kwargs: dict) -> None:
                x, t = args[0], args[1]
                for k in range(len(x)):
                    record.append((x[k].detach().cpu().clone(), float(t[k])))

            hook = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                lat = self.pipe(
                    prompt_embeds=[embed.to("cuda", next(model.parameters()).dtype)],
                    height=self.size,
                    width=self.size,
                    num_inference_steps=STEPS,
                    guidance_scale=0.0,
                    latents=self.latents(seed).to("cuda"),
                    output_type="latent",
                ).images
                require(bool(torch.isfinite(lat).all()), "non-finite latent")
                img = self.vae.decode(
                    lat.float() / self.vae.config.scaling_factor + self.vae.config.shift_factor
                ).sample
        finally:
            if hook is not None:
                hook.remove()
        arr = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return lat.float().cpu(), arr


# ---------- stages ----------
def stage_refs(args: argparse.Namespace) -> None:
    import torch
    from diffusers import ZImagePipeline

    setup_torch()
    out = args.work / "refs"
    require(not (out / "refs.json").exists(), f"{out} exists; use a fresh --work")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pipe = ZImagePipeline.from_pretrained(model_dir(), transformer=None, vae=None, torch_dtype=torch.float32)
    pipe.text_encoder.to("cuda")
    embeds = {}
    with torch.no_grad():
        table_prompts = TABLE_PROMPTS[: args.table_prompts]
        for split, prompts in (("table", table_prompts), ("eval", EVAL_PROMPTS)):
            items, _ = pipe.encode_prompt(
                prompt=list(prompts), device="cuda", do_classifier_free_guidance=False
            )
            embeds[split] = [e.float().cpu() for e in items]
            require(
                all(bool(torch.isfinite(e).all()) for e in embeds[split]), f"non-finite {split} embedding"
            )
    torch.save(embeds, out / "embeds.pt")
    import gc

    pipe.text_encoder.to("cpu")  # deleting the pipeline alone left the 16 GB encoder on the GPU (V100 probe)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    require(
        torch.cuda.memory_allocated() < 2**30, f"{torch.cuda.memory_allocated()} bytes left after encoding"
    )
    print(f"ZT_REFS_ENCODED table={len(table_prompts)} eval={len(EVAL_PROMPTS)}", flush=True)
    model = load_transformer(args.dtype)
    gen = Generator(model, args.size)
    record: list = []
    caps = []
    for k, embed in enumerate(embeds["table"]):
        before = len(record)
        gen(embed, SEED + k, record)
        caps.extend([k] * (len(record) - before))
    require(len(record) == len(table_prompts) * STEPS, f"recorded {len(record)} forwards")
    refs = {
        "latents": torch.stack([x for x, _ in record]),
        "t": torch.tensor([t for _, t in record], dtype=torch.float32),
        "cap": torch.tensor(caps),
    }
    torch.save(refs, out / "replay.pt")
    write_json(
        out / "refs.json",
        {
            "dtype": args.dtype,
            "size": args.size,
            "steps": STEPS,
            "seed": SEED,
            "table_prompts": list(table_prompts),
            "eval_prompts": list(EVAL_PROMPTS),
            "demo_prompts": len(DEMO_PROMPTS),
            "caption_tokens": {s: [int(e.shape[0]) for e in v] for s, v in embeds.items()},
            "forwards": len(record),
            "embeds_sha256": sha256_file(out / "embeds.pt"),
            "replay_sha256": sha256_file(out / "replay.pt"),
            "seconds": round(time.monotonic() - started, 1),
            "env": run_env(),
        },
    )
    print(f"ZT_REFS_DONE forwards={len(record)} seconds={time.monotonic() - started:.0f}", flush=True)


def load_refs(work: Path) -> tuple[dict, dict, dict]:
    import torch

    info = json.loads((work / "refs" / "refs.json").read_text())
    require(
        sha256_file(work / "refs" / "replay.pt") == info["replay_sha256"], "replay.pt differs from refs.json"
    )
    require(
        sha256_file(work / "refs" / "embeds.pt") == info["embeds_sha256"], "embeds.pt differs from refs.json"
    )
    return info, torch.load(work / "refs" / "replay.pt"), torch.load(work / "refs" / "embeds.pt")


def fingerprint(info: Mapping[str, Any], batch: int) -> dict[str, Any]:
    """What a table row depends on besides the layer: refs, numerics, methods, replay batching. A resumed
    shard, the merge, an evaluation reused and the export must all see the same one."""
    identity = {
        "replay_sha256": info["replay_sha256"],
        "embeds_sha256": info["embeds_sha256"],
        "dtype": info["dtype"],
        "size": info["size"],
        "steps": STEPS,
        "batch": batch,
        "methods": {m: repr(weight_spec(m)) for m in METHODS},
        "group": GROUP,
    }
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def stage_table(args: argparse.Namespace) -> None:
    import torch

    setup_torch()
    info, refs, embeds = load_refs(args.work)
    require(info["dtype"] == args.dtype, f"refs were recorded in {info['dtype']}")
    k, n = (int(x) for x in args.shard.split("/"))
    require(1 <= k <= n, "--shard k/n with 1 <= k <= n")
    out = args.work / f"table_{k}of{n}.jsonl"
    mark = args.work / f"table_{k}of{n}.fingerprint.json"
    fp = fingerprint(info, args.batch)
    done = set()
    if out.exists():
        require(mark.exists(), f"{out.name} has rows but no fingerprint; start a fresh --work")
        require(
            json.loads(mark.read_text())["sha256"] == fp["sha256"], f"{out.name} was measured differently"
        )
        done = {json.loads(line)["name"] for line in out.read_text().splitlines() if line.strip()}
    else:
        write_json(mark, fp)
    started = time.monotonic()
    model = load_transformer(args.dtype)
    layers, modules, others = linear_layers(model)
    native = weights_digest(modules)
    base = replay(model, refs, embeds["table"], args.batch)
    again = replay(model, refs, embeds["table"], args.batch)
    require(
        all(torch.equal(a, b) for a, b in zip(base, again, strict=True)), "native replay is not deterministic"
    )
    mine = [(i, layer) for i, layer in enumerate(layers) if i % n == k - 1 and layer.name not in done]
    print(
        f"ZT_TABLE_START shard={k}/{n} layers={len(layers)} todo={len(mine)} done={len(done)} "
        f"forwards={len(base)} other_params={others}",
        flush=True,
    )
    with out.open("a") as stream:
        for count, (index, layer) in enumerate(mine):
            tick = time.monotonic()
            d = {}
            for method in QUANT_METHODS:
                with quantized(modules, {layer.name: method}):
                    d[method] = relative_mse(replay(model, refs, embeds["table"], args.batch), base)
            require(all(math.isfinite(v) and v >= 0 for v in d.values()), f"{layer.name}: bad D {d}")
            row = {"index": index, **asdict(layer), "d": d, "seconds": round(time.monotonic() - tick, 1)}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(
                f"ZT_TABLE_LAYER {count + 1}/{len(mine)} #{index} {layer.name} "
                + " ".join(f"{m}={v:.3e}" for m, v in d.items())
                + f" t={row['seconds']}s",
                flush=True,
            )
    require(weights_digest(modules) == native, "native weights were not restored after the table")
    meta = {
        "shard": f"{k}/{n}",
        "layers": len(layers),
        "other_params": others,
        "heuristic_int8": heuristic_names(layers),
        "native_digest": native,
        "fingerprint": fp,
        "seconds": round(time.monotonic() - started, 1),
        "env": run_env(),
    }
    write_json(args.work / f"table_{k}of{n}.meta.json", meta)
    print(f"ZT_TABLE_DONE shard={k}/{n} seconds={time.monotonic() - started:.0f}", flush=True)


def stage_merge(args: argparse.Namespace) -> None:
    info = json.loads((args.work / "refs" / "refs.json").read_text())
    metas = sorted(args.work.glob("table_*of*.meta.json"))
    require(metas, "no finished table shards")
    rows: dict[str, dict] = {}
    first = json.loads(metas[0].read_text())
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        require(meta["native_digest"] == first["native_digest"], f"{meta_path.name}: another model")
        require(meta["fingerprint"] == first["fingerprint"], f"{meta_path.name}: measured differently")
        require(
            meta["fingerprint"]["replay_sha256"] == info["replay_sha256"], f"{meta_path.name}: other refs"
        )
        for line in (
            meta_path.with_name(meta_path.name.replace(".meta.json", ".jsonl")).read_text().splitlines()
        ):
            row = json.loads(line)
            require(
                row["name"] not in rows or rows[row["name"]]["d"] == row["d"],
                f"{row['name']} twice, differently",
            )
            rows[row["name"]] = row
    require(len(rows) == first["layers"], f"{len(rows)} of {first['layers']} layers measured")
    names = sorted(rows, key=lambda name: rows[name]["index"])
    require(
        [rows[name]["index"] for name in names] == list(range(len(names))), "layer indices are not 0..n-1"
    )
    write_json(
        args.work / "table.json",
        {
            "schema": TABLE_SCHEMA,
            "model": REPO,
            "revision": REVISION,
            "dtype": info["dtype"],
            "size": info["size"],
            "steps": STEPS,
            "layers": [{k: rows[name][k] for k in ("name", "params", "k", "out", "d")} for name in names],
            "other_params": first["other_params"],
            "heuristic_int8": first["heuristic_int8"],
            "native_digest": first["native_digest"],
            "fingerprint": first["fingerprint"],
            "calibration": {
                "prompts": info["table_prompts"],
                "seed": SEED,
                "forwards": info["forwards"],
                "rule": f"prompt k uses seed {SEED}+k; every transformer forward of its {STEPS}-forward "
                "generation",
            },
            "shards": [json.loads(m.read_text()) for m in metas],
            "env": run_env(),
        },
    )
    print(f"ZT_MERGE_DONE layers={len(rows)}", flush=True)


def table_layers(table: Mapping[str, Any]) -> tuple[list[Layer], dict[str, dict[str, float]]]:
    layers = [Layer(row["name"], row["params"], row["k"], row["out"]) for row in table["layers"]]
    return layers, {row["name"]: row["d"] for row in table["layers"]}


def stage_solve(args: argparse.Namespace) -> None:
    table_path = args.work / "table.json"
    out = args.work / "solutions.json"
    require(not out.exists(), f"{out} exists")
    table = json.loads(table_path.read_text())
    require(table["schema"] == TABLE_SCHEMA, "not an image_tune table")
    layers, d = table_layers(table)
    require(heuristic_names(layers) == table["heuristic_int8"], "the heuristic layers changed")
    params = sum(layer.params for layer in layers)
    solutions = {}
    for auto, reference in AUTO.items():
        target = avg_bits(layers, baseline_assignment(reference, layers))
        assignment, solution = solve(layers, d, target * params)
        bits = avg_bits(layers, assignment)
        require(bits <= target, f"{auto}: {bits} bits exceed the budget {target}")
        solutions[auto] = {
            "reference": reference,
            "budget_fraction": fraction_text(target),
            "bits_fraction": fraction_text(bits),
            "predicted_d": solution.value,
            "solver": {"method": "milp (scipy HiGHS)", "bound": solution.bound, "gap": solution.gap},
            "cost_bits": solution.cost,
            "methods": assignment,
            "method_counts": method_counts(assignment),
        }
        print(
            f"ZT_SOLVE {auto} budget={float(target):.6f} bits={float(bits):.6f} "
            f"predicted_d={solution.value:.4e} "
            f"counts={method_counts(assignment)}",
            flush=True,
        )
    write_json(out, {"table_sha256": sha256_file(table_path), "solutions": solutions, "env": run_env()})
    print(f"ZT_SOLVE_DONE out={out}", flush=True)


def stage_evaluate(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    setup_torch()
    info, refs, embeds = load_refs(args.work)
    require(info["dtype"] == args.dtype, f"refs were recorded in {info['dtype']}")
    table = json.loads((args.work / "table.json").read_text())
    layers, d = table_layers(table)
    solutions = {}
    if (args.work / "solutions.json").exists():
        solved = json.loads((args.work / "solutions.json").read_text())
        require(
            solved["table_sha256"] == sha256_file(args.work / "table.json"),
            "solutions are from another table",
        )
        solutions = solved["solutions"]
    model = load_transformer(args.dtype)
    found, modules, _ = linear_layers(model)
    require(
        [x.name for x in found] == [x.name for x in layers], "the model's Linears differ from the table's"
    )
    native_digest = weights_digest(modules)
    require(native_digest == table["native_digest"], "another model than the table's")
    now = fingerprint(info, table["fingerprint"]["batch"])  # this code's methods and steps on these refs
    require(now == table["fingerprint"], "the table was measured on other refs, numerics or methods")
    gen = Generator(model, info["size"])
    base = replay(model, refs, embeds["table"], table["fingerprint"]["batch"])  # the table's batching
    for setup in [s.strip() for s in args.setups.split(",") if s.strip()]:
        require(setup in SETUPS, f"unknown setup {setup}")
        out = args.work / f"eval_{setup}.json"
        if out.exists():
            old = json.loads(out.read_text())
            require(
                old["fingerprint"] == table["fingerprint"]["sha256"]
                and old["table_sha256"] == sha256_file(args.work / "table.json"),
                f"{out.name} exists from another table or refs; move it aside",
            )
            print(f"ZT_EVAL_SKIP {setup} (exists)", flush=True)
            continue
        if setup in AUTO:
            require(setup in solutions, f"{setup}: run solve first")
            assignment = solutions[setup]["methods"]
        else:
            assignment = baseline_assignment(setup, layers)
        native_dir = args.work / "images" / "native"
        require(setup == "native" or native_dir.is_dir(), "evaluate native first")
        tick = time.monotonic()
        folder = args.work / "images" / setup
        folder.mkdir(parents=True, exist_ok=True)
        scores = []
        with quantized(modules, assignment):
            measured = (
                0.0
                if setup == "native"
                else relative_mse(replay(model, refs, embeds["table"], table["fingerprint"]["batch"]), base)
            )
            for k, embed in enumerate(embeds["eval"]):
                _, arr = gen(embed, SEED + 1000 + k)
                Image.fromarray(arr).save(folder / f"{k:02d}.png")
                if setup != "native":
                    ref = np.asarray(Image.open(native_dir / f"{k:02d}.png").convert("RGB"))
                    scores.append({"psnr": psnr(arr, ref), "ssim": ssim(arr, ref)})
        require(weights_digest(modules) == native_digest, f"{setup}: native weights were not restored")
        finite = [s["psnr"] for s in scores if s["psnr"] is not None]
        metrics = (
            None
            if setup == "native"
            else {
                # mean over the prompts whose image changed; how many did not is said beside it
                "psnr": sum(finite) / len(finite) if finite else None,
                "ssim": sum(s["ssim"] for s in scores) / len(scores),
                "identical": len(scores) - len(finite),
            }
        )
        write_json(
            out,
            {
                "setup": setup,
                "methods": assignment,
                "bits_fraction": fraction_text(avg_bits(layers, assignment)),
                "measured_d": measured,
                "metrics": metrics,
                "per_prompt": scores,
                "prompts": info["eval_prompts"],
                "seeds": [SEED + 1000 + k for k in range(len(EVAL_PROMPTS))],
                "size": info["size"],
                "dtype": args.dtype,
                "fingerprint": table["fingerprint"]["sha256"],
                "table_sha256": sha256_file(args.work / "table.json"),
                "seconds": round(time.monotonic() - tick, 1),
                "env": run_env(),
            },
        )
        print(
            f"ZT_EVAL_SETUP {setup} bits={float(avg_bits(layers, assignment)):.4f} measured_d={measured:.4e} "
            + (
                f"psnr={metrics['psnr']} ssim={metrics['ssim']:.4f} identical={metrics['identical']} "
                if metrics
                else ""
            )
            + f"t={time.monotonic() - tick:.0f}s",
            flush=True,
        )
    print("ZT_EVAL_DONE", flush=True)


def compare(setups: Sequence[dict[str, Any]]) -> None:
    """The verdict, fixed before any evaluation (docs/evidence/2026-11-14_zimage_turbo.md §1.5): each
    Auto-tune setup against the existing way whose size it was given, on per-prompt SSIM, exact sign-flip
    test, Holm over the Auto-tune setups. Adds ``paired`` to every auto setup."""
    by_id = {e["id"]: e for e in setups}
    autos = [e for e in setups if e["kind"] == "auto"]
    for entry in autos:
        ref = by_id[AUTO[entry["id"]]]
        diffs = [a - b for a, b in zip(entry["per_prompt"]["ssim"], ref["per_prompt"]["ssim"], strict=True)]
        entry["paired"] = {
            "vs": ref["id"],
            "metric": "ssim",
            "n": len(diffs),
            "wins": sum(x > 0 for x in diffs),
            "losses": sum(x < 0 for x in diffs),
            "mean_diff": sum(diffs) / len(diffs),
            "p_sign_flip": sign_flip_p(diffs),
        }
    for entry, adjusted in zip(autos, holm([e["paired"]["p_sign_flip"] for e in autos]), strict=True):
        paired = entry["paired"]
        paired["p_holm"] = adjusted
        paired["verdict"] = verdict(adjusted, paired["mean_diff"])


def verdict(p_holm: float, mean_diff: float) -> str:
    return (
        "ours" if p_holm < 0.05 and mean_diff > 0 else "theirs" if p_holm < 0.05 and mean_diff < 0 else "same"
    )


def stage_export(args: argparse.Namespace) -> None:
    from PIL import Image

    table = json.loads((args.work / "table.json").read_text())
    solved = json.loads((args.work / "solutions.json").read_text())
    require(
        solved["table_sha256"] == sha256_file(args.work / "table.json"), "solutions are from another table"
    )
    evals = {}
    for setup in SETUPS:
        path = args.work / f"eval_{setup}.json"
        require(path.exists(), f"eval_{setup}.json not found")
        evals[setup] = json.loads(path.read_text())
        require(
            evals[setup]["table_sha256"] == solved["table_sha256"], f"{setup}: evaluated with another table"
        )
        require(
            evals[setup]["fingerprint"] == table["fingerprint"]["sha256"], f"{setup}: other refs or numerics"
        )
        for key in ("prompts", "seeds", "size", "dtype"):
            require(evals[setup][key] == evals["native"][key], f"{setup}: another {key} than the native run")
        require(
            evals[setup]["prompts"] == list(EVAL_PROMPTS), f"{setup}: not this script's evaluation prompts"
        )
    info = json.loads((args.work / "refs" / "refs.json").read_text())
    now = fingerprint(info, table["fingerprint"]["batch"])
    require(now == table["fingerprint"], "the table was measured on other refs, numerics or methods")
    layers, d = table_layers(table)
    others = table["other_params"]
    setups = []
    for setup in SETUPS[1:]:
        result = evals[setup]
        assignment = result["methods"]
        bits = avg_bits(layers, assignment)
        entry = {
            "id": setup,
            "kind": "auto" if setup in AUTO else "uniform" if setup in UNIFORM else "heuristic",
            "label": LABELS[setup],
            "bits": float(bits),
            "bits_fraction": fraction_text(bits),
            "bytes": model_bytes(layers, assignment, others),
            "methods": assignment,
            "method_counts": method_counts(assignment),
            "predicted_d": predicted_d(assignment, d),
            "measured_d": result["measured_d"],
            "metrics": result["metrics"],
            "env": result["env"],
        }
        entry["per_prompt"] = {
            "ssim": [x["ssim"] for x in result["per_prompt"]],
            "psnr": [x["psnr"] for x in result["per_prompt"]],
        }
        if setup in AUTO:
            require(solved["solutions"][setup]["methods"] == assignment, f"{setup}: evaluated methods differ")
            entry["budget_bits"] = solved["solutions"][setup]["budget_fraction"]
        setups.append(entry)
    compare(setups)
    media = args.media
    media.mkdir(parents=True, exist_ok=True)
    images = []
    for k, prompt in enumerate(DEMO_PROMPTS):
        files = {}
        for setup in SETUPS:
            source = args.work / "images" / setup / f"{k:02d}.png"
            target = media / f"{setup}_{k}.jpg"
            Image.open(source).convert("RGB").resize((args.thumb, args.thumb), Image.LANCZOS).save(
                target, quality=88, optimize=True
            )
            files[setup] = f"media/zimage/{target.name}"
        images.append(
            {
                "prompt": prompt,
                "seed": SEED + 1000 + k,
                "files": files,
                "psnr": {s: evals[s]["per_prompt"][k]["psnr"] for s in SETUPS[1:]},
            }
        )
    native = evals["native"]
    data = {
        "schema": SCHEMA,
        "model": {
            "id": "z-image-turbo",
            "label": "Z-Image-Turbo",
            "repo": REPO,
            "revision": REVISION,
            "dtype": table["dtype"],
            "size": table["size"],
            "steps": STEPS,
            "linear_layers": len(layers),
            "linear_params": sum(layer.params for layer in layers),
            "other_params": others,
            "original_bytes": original_bytes(layers, others),
            "size_rule": SIZE_RULE,
        },
        "quantization": QUANTIZATION,
        "objective": OBJECTIVE,
        "calibration": {**table["calibration"], "env": table["env"]},
        "evaluation": {
            "prompts": len(native["prompts"]),
            "rule": "the evaluation prompts (none is a table prompt), seed 1042+k, the same latents for "
            "every setup",
            "metric": "PSNR (dB) and SSIM of the decoded image against the unquantized model's image, mean "
            "over prompts",
        },
        "methods": [{"id": method, "bits": bits} for method, bits in METHODS.items()],
        "layers": [
            {"name": layer.name, "params": layer.params, "k": layer.k, "out": layer.out, "d": d[layer.name]}
            for layer in layers
        ],
        "setups": setups,
        # the original against itself: the same image (SSIM 1, no finite PSNR)
        "native": {
            "metrics": {"ssim": 1.0, "psnr": None},
            "bytes": original_bytes(layers, others),
            "env": native["env"],
        },
        "images": images,
        "env": run_env(),
    }
    problems = validate(data)
    require(not problems, "exported dataset is inconsistent: " + "; ".join(problems))
    write_json(args.out, data)
    print(f"ZT_EXPORT_DONE out={args.out} bytes={args.out.stat().st_size}", flush=True)


def _metric_problems(setup: Mapping[str, Any], prompts: int) -> list[str]:
    """A setup's mean SSIM / PSNR are the means of its per-prompt values (PSNR over the changed images)."""
    per = setup.get("per_prompt") or {}
    ssim_values, psnr_values = per.get("ssim") or [], per.get("psnr") or []
    if len(ssim_values) != prompts or len(psnr_values) != prompts:
        return [f"{setup['id']}: per-prompt values"]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and -1 <= v <= 1 for v in ssim_values):
        return [f"{setup['id']}: SSIM values"]
    changed = [v for v in psnr_values if v is not None]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in changed):
        return [f"{setup['id']}: PSNR values"]
    m = setup["metrics"]
    problems = []
    if not abs(m["ssim"] - sum(ssim_values) / prompts) <= 1e-12:  # also true for NaN
        problems.append(f"{setup['id']}: mean SSIM")
    if (m["psnr"] is None) != (not changed) or (
        changed and not abs(m["psnr"] - sum(changed) / len(changed)) <= 1e-9
    ):
        problems.append(f"{setup['id']}: mean PSNR")
    if m.get("identical") != prompts - len(changed):
        problems.append(f"{setup['id']}: identical count")
    return problems


def _same_paired(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if set(a) != set(b):
        return False
    for key, value in a.items():
        if isinstance(value, float):
            if not abs(value - b[key]) <= 1e-12 * max(1.0, abs(b[key])):
                return False
        elif value != b[key]:
            return False
    return True


def validate(data: Mapping[str, Any]) -> list[str]:
    """Internal consistency of an exported dataset (also run by app/tests/test_zimage_tune.py)."""
    problems = []
    layers = [Layer(x["name"], x["params"], x["k"], x["out"]) for x in data["layers"]]
    if data["schema"] != SCHEMA:
        problems.append(f"schema {data['schema']}")
    if data["model"]["linear_layers"] != len(layers):
        problems.append("linear_layers differs from the layer list")
    if data["model"]["original_bytes"] != original_bytes(layers, data["model"]["other_params"]):
        problems.append("original_bytes")
    names = {x.name for x in layers}
    for row in data["layers"]:
        if set(row["d"]) != set(QUANT_METHODS) or not all(
            math.isfinite(v) and v >= 0 for v in row["d"].values()
        ):
            problems.append(f"{row['name']}: table row")
    by_id = {s["id"]: s for s in data["setups"]}
    if sorted(by_id) != sorted(SETUPS[1:]):
        problems.append(f"setups {sorted(by_id)}")
    for s in data["setups"]:
        if set(s["methods"]) != names:
            problems.append(f"{s['id']}: methods do not name every layer")
            continue
        bits = avg_bits(layers, s["methods"])
        if fraction_text(bits) != s["bits_fraction"] or s["bytes"] != model_bytes(
            layers, s["methods"], data["model"]["other_params"]
        ):
            problems.append(f"{s['id']}: bits or bytes")
        if s["kind"] != "auto" and s["methods"] != baseline_assignment(s["id"], layers):
            problems.append(f"{s['id']}: not its rule")
        if s["kind"] == "auto":
            if bits > parse_fraction(s["budget_bits"]):
                problems.append(f"{s['id']}: over budget")
            ref = by_id.get(AUTO[s["id"]])
            if ref is None or parse_fraction(s["budget_bits"]) != avg_bits(layers, ref["methods"]):
                problems.append(f"{s['id']}: budget is not its reference's size")
        want = predicted_d(s["methods"], {r["name"]: r["d"] for r in data["layers"]})
        if not (math.isfinite(s["predicted_d"]) and abs(s["predicted_d"] - want) <= 1e-12 * max(1.0, want)):
            problems.append(f"{s['id']}: predicted_d")
        if not (
            isinstance(s["measured_d"], (int, float))
            and math.isfinite(s["measured_d"])
            and s["measured_d"] >= 0
        ):
            problems.append(f"{s['id']}: measured_d")
        problems += _metric_problems(s, data["evaluation"]["prompts"])
        if s["kind"] == "auto":
            paired = s.get("paired") or {}
            if paired.get("vs") != AUTO[s["id"]] or not 0 <= paired.get("p_holm", -1) <= 1:
                problems.append(f"{s['id']}: paired comparison")
            elif paired["verdict"] != verdict(paired["p_holm"], paired["mean_diff"]):
                problems.append(f"{s['id']}: verdict")
    autos = [s for s in data["setups"] if s["kind"] == "auto"]
    if autos and not any("paired comparison" in x or "per-prompt" in x for x in problems):
        again = json.loads(json.dumps(data["setups"]))  # the verdict recomputed from the per-prompt values
        compare(again)
        for s, redo in zip(data["setups"], again, strict=True):
            if s["kind"] == "auto" and not _same_paired(s["paired"], redo["paired"]):
                problems.append(f"{s['id']}: paired statistics do not follow from the per-prompt values")
    if data["native"].get("metrics") != {"ssim": 1.0, "psnr": None}:
        problems.append("native metrics")
    for image in data["images"]:
        if sorted(image["files"]) != sorted(SETUPS):
            problems.append("an image lacks a setup")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("refs", "table", "merge", "solve", "evaluate", "export"):
        stage = sub.add_parser(name)
        stage.add_argument("--work", type=Path, required=True, help="directory of the intermediate files")
        if name in ("refs", "table", "evaluate"):
            # FP32 by default: the checkpoint's own precision; on a V100 FP16 overflows (non-finite latents)
            stage.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
            stage.add_argument("--batch", type=int, default=8, help="recorded forwards per transformer call")
        if name == "refs":
            stage.add_argument("--size", type=int, default=512)
            stage.add_argument(
                "--table-prompts", type=int, default=4, choices=range(1, len(TABLE_PROMPTS) + 1)
            )
        if name == "table":
            stage.add_argument("--shard", default="1/1")
        if name == "evaluate":
            stage.add_argument("--setups", required=True, help=f"comma list of {','.join(SETUPS)}")
        if name == "export":
            stage.add_argument(
                "--out", type=Path, default=ROOT / "app" / "web" / "demo" / "autotune" / "zimage.json"
            )
            stage.add_argument(
                "--media", type=Path, default=ROOT / "app" / "web" / "demo" / "media" / "zimage"
            )
            stage.add_argument("--thumb", type=int, default=384)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    stages = {
        "refs": stage_refs,
        "table": stage_table,
        "merge": stage_merge,
        "solve": stage_solve,
        "evaluate": stage_evaluate,
        "export": stage_export,
    }
    try:
        stages[args.stage](args)
    except Exception as error:
        traceback.print_exc()
        print(f"ZT_FAIL stage={args.stage} error={type(error).__name__}: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    sys.exit(main())
