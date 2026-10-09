"""Export real FP8 operands of a transformer layer, with TriCast's own GEMM output bits, for the browser lab.

Run on a CUDA machine (the cluster): one prompt goes through the native model, forward pre-hooks capture
the input activation of ``model.layers.0.self_attn.q_proj`` and ``model.layers.0.mlp.down_proj``, the
activation and the weight are quantized with TriCast's ``fp8_tensor`` QuantSpec (``tricast.quantize``),
and ``tricast.gemm`` output bits (out_format fp32) are recorded for each MMA setting and backend. The
browser then runs the same operands and compares its bits with the server's.

    HF_HUB_OFFLINE=1 python -m app.export_operands --device cuda \\
        [--models qwen3-0.6b llama-3.2-1b] [--backends triton reference] [--out app/web/demo/webgpu/operands]

Files per pack ``<id>``: ``<id>.a.u8.bin`` (activation codes ``[M, K]``, raw E4M3 bytes, row-major),
``<id>.b.u8.bin`` (weight codes ``[N, K]``), ``<id>.<setting>.<backend>.u32.bin`` (output bits ``[M, N]``,
little-endian u32), ``<id>.json`` (shapes, scales as fp32 hex, model, revision, prompt, layer, env).
``index.json`` lists the packs for the GUI (``server`` maps the GUI's canonical-mma key to a bits file).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

import tricast
from tricast.mma.spec import MMASpec, get_preset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "web" / "demo" / "webgpu" / "operands"
SCHEME = "fp8_tensor"
LAYERS = ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj")
DEFAULT_LAYERS = LAYERS[:1]  # Llama-3.2-1B down_proj alone is 16.8 MB of codes; opt in with --layers
PROMPT = "The history of the printing press begins"
# Revisions (pinned) and chat flags come from app/catalog.py.
MODELS = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "llama-3.2-1b": "meta-llama/Llama-3.2-1B"}


def _settings() -> dict[str, tuple[str, MMASpec]]:
    """MMA settings with their source; outputs are fp32 so the browser (fp32 only) can compare bits."""
    def recipe(name: str) -> MMASpec:
        return tricast.load_recipe(name).defaults.mma.with_(out_format="fp32")

    return {"f7_fused": ("recipe fp8_f7_lowacc", recipe("fp8_f7_lowacc")),
            "hopper": ("preset nvidia_hopper_fp8", get_preset("nvidia_hopper_fp8").with_(out_format="fp32")),
            "f7_decoupled": ("recipe fp8_f7_decoupled", recipe("fp8_f7_decoupled"))}


def canonical_mma(spec: MMASpec) -> dict:
    """The browser's form of a CoFDA setting (f2_bits only when decoupled)."""
    if spec.algorithm != "cofda" or spec.promote_interval:
        raise ValueError(f"the browser path runs cofda without promotion, not {spec}")
    mma = {"algorithm": "cofda", "f_bits": spec.f_bits, "chunk_size": spec.chunk_size, "c_mode": spec.c_mode,
           "promote_interval": 0, "norm_rounding": spec.norm_rounding}
    if spec.c_mode == "decoupled":
        mma["f2_bits"] = spec.f2_bits
    return mma


def preset_key(mma: dict) -> str:
    """The GUI's lookup key: ``JSON.stringify(Object.keys(mma).sort().map(k => [k, mma[k]]))``."""
    return json.dumps([[key, mma[key]] for key in sorted(mma)], separators=(",", ":"))


def capture_inputs(model: torch.nn.Module, input_ids: torch.Tensor,
                   layers: Sequence[str]) -> dict[str, torch.Tensor]:
    """Input activation of each named linear for one native forward pass."""
    modules = dict(model.named_modules())
    captured: dict[str, torch.Tensor] = {}

    def hook(name: str) -> Callable:
        def store(_module: torch.nn.Module, args: tuple) -> None:
            captured[name] = args[0].detach()
        return store

    handles = [modules[name].register_forward_pre_hook(hook(name)) for name in layers]
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _codes(q: Any) -> torch.Tensor:
    """E4M3 codes of a QTensor's grid values (an exact cast: the values are on the grid)."""
    values = q.values.float()
    codes = values.to(torch.float8_e4m3fn).view(torch.uint8)
    if not torch.equal(codes.view(torch.float8_e4m3fn).float(), values):
        raise ValueError("quantized values are not on the fp8_e4m3 grid")
    return codes.cpu()


def _f32_hex(x: torch.Tensor) -> str:
    return f"{x.float().reshape(()).cpu().view(torch.int32).item() & 0xFFFFFFFF:08x}"


def _u32_bytes(out: torch.Tensor) -> bytes:
    return out.float().contiguous().cpu().view(torch.int32).numpy().astype("<i4").tobytes()


def _update_index(out_dir: Path, packs: list[dict]) -> None:
    path = out_dir / "index.json"
    existing = json.loads(path.read_text())["packs"] if path.exists() else []
    fresh = {pack["id"]: pack for pack in packs}
    merged = [fresh.pop(pack["id"], pack) for pack in existing] + list(fresh.values())
    path.write_text(json.dumps({"packs": merged}, ensure_ascii=False, indent=1) + "\n")


def export_model(model: torch.nn.Module, input_ids: torch.Tensor, *, model_key: str, model_id: str,
                 label: str, revision: str | None, prompt: str, out_dir: Path,
                 backends: Sequence[str] = ("triton", "reference"),
                 layers: Sequence[str] = LAYERS) -> list[dict]:
    """Write one pack per layer into ``out_dir`` and merge them into ``index.json``."""
    from tricast.eval.envinfo import capture_env

    out_dir.mkdir(parents=True, exist_ok=True)
    captured = capture_inputs(model, input_ids, layers)
    modules = dict(model.named_modules())
    env = capture_env(None, {"model_id": model_id, "model_revision": revision,
                             "tricast": tricast.__version__})
    settings = _settings()
    packs = []
    for layer in layers:
        x = captured[layer]
        qa = tricast.quantize(x.reshape(-1, x.shape[-1]).float(), SCHEME)
        qb = tricast.quantize(modules[layer].weight.detach().float(), SCHEME)
        a_codes, b_codes = _codes(qa), _codes(qb)
        (M, K), N = a_codes.shape, b_codes.shape[0]
        short = layer.replace("model.layers.", "l").replace("self_attn.", "").replace("mlp.", "")
        pack_id = f"{model_key}.{short}"
        files = {"a": f"{pack_id}.a.u8.bin", "b": f"{pack_id}.b.u8.bin"}
        (out_dir / files["a"]).write_bytes(a_codes.numpy().tobytes())
        (out_dir / files["b"]).write_bytes(b_codes.numpy().tobytes())
        server, results = {}, {}
        for name, (source, spec) in settings.items():
            mma = canonical_mma(spec)
            bits = {}
            for backend in backends:
                data = _u32_bytes(tricast.gemm(qa, qb, spec, backend=backend))
                bits[backend] = f"{pack_id}.{name}.{backend}.u32.bin"
                (out_dir / bits[backend]).write_bytes(data)
            blobs = {backend: (out_dir / f).read_bytes() for backend, f in bits.items()}
            results[name] = {"source": source, "mma": mma, "key": preset_key(mma), "bits": bits,
                             "backends_equal": len(set(blobs.values())) == 1}
            server[preset_key(mma)] = bits[backends[0]]
        meta = {"id": pack_id, "model": model_id, "revision": revision, "prompt": prompt, "layer": layer,
                "scheme": SCHEME, "M": M, "N": N, "K": K, "a": {"file": files["a"], "shape": [M, K]},
                "b": {"file": files["b"], "shape": [N, K]}, "scale_a": _f32_hex(qa.scale),
                "scale_b": _f32_hex(qb.scale), "out_format": "fp32", "settings": results, "env": env}
        (out_dir / f"{pack_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
        packs.append({"id": pack_id, "label": f"{label} · {short}", "model": model_id,
                      "layer": layer, "M": M, "N": N, "K": K, "a_file": files["a"], "b_file": files["b"],
                      "scale_a": qa.scale.float().item(), "scale_b": qb.scale.float().item(),
                      "meta_file": f"{pack_id}.json", "server": server})
    _update_index(out_dir, packs)
    return packs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backends", nargs="+", default=["triton", "reference"],
                        choices=["triton", "reference"])
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS), choices=list(LAYERS))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before huggingface_hub reads it
    from .catalog import MODELS as CATALOG
    from .runners.llm import encode_prompt, load_llm

    for key in args.models:
        entry = CATALOG[MODELS[key]]
        model, tokenizer = load_llm(entry.id, args.device, entry.revision)
        input_ids = encode_prompt(tokenizer, args.prompt, entry.chat).to(args.device)
        packs = export_model(model, input_ids, model_key=key, model_id=entry.id, label=entry.label,
                             revision=entry.revision, prompt=args.prompt, out_dir=args.out,
                             backends=args.backends, layers=args.layers)
        for pack in packs:
            print(f"EXPORT_PACK id={pack['id']} M={pack['M']} N={pack['N']} K={pack['K']}")
    print(f"EXPORT_OPERANDS_DONE out={args.out}")


if __name__ == "__main__":
    main()
