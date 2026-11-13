"""What the tuning app (app/web/tune.html, step 5) hands over after the evaluation: the model's size under a
setup and the recipe bundle (``POST /api/autotune/export``). Stdlib plus ``app.autotune``: no torch, no GPU.

- Size (``model_size``) is computed, not a file measured: the original checkpoint's tensors (``MODEL_FILES``,
  read from its safetensors header) with the Linear weights at the setup's average stored bits (element +
  scale bits, ``app.autotune.weight_bits``, as the page's size is counted) and every other tensor (token
  embedding, output layer, norms) left in BF16, as the recipes leave them. It is the size of a low-bit file;
  TriCast does not write one yet (the evaluated model is a fake-quant model in BF16).
- The bundle (``bundle_zip``) is a zip: the recipe file (the very bytes the recipe route serves), the GPU
  measurement recorded for it (its PPL with the record that measured that PPL: a session proposal's KL and
  PPL were measured on different trees), the evaluation this server just ran on it when there is one, the size
  and a README with the commands that apply the recipe.
"""

from __future__ import annotations

import io
import json
import zipfile
from fractions import Fraction
from typing import Any

from app import autotune as at

# The original checkpoint per model, read from the safetensors header of its pinned revision: 311 BF16
# tensors — Linear weights 440,401,920 (the 196 Linear layers of the table), token embedding and output layer
# 2 × 151,936 × 1,024 (stored separately although the config ties them), RMSNorm weights 65,536 — so the file
# is 2 bytes per parameter plus the 35,552-byte header and its 8-byte length.
MODEL_FILES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-0.6B": {
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "file": "model.safetensors", "file_bytes": 1_503_300_328, "header_bytes": 35_560,
        "linear_params": 440_401_920, "embedding_params": 311_164_928, "norm_params": 65_536,
        "source": "model.safetensors header of revision c1899de (embed_tokens + lm_head, norms, 196 Linear)",
    },
    # 399 BF16 tensors in 5 shards: Linear 6,945,767,424 (36 blocks × 7), embed_tokens + lm_head (untied)
    # 2 × 151,936 × 4,096, RMSNorm 308,224; 2 bytes each + 46,056 header bytes (5 headers and their lengths)
    "Qwen/Qwen3-8B": {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "file": "model-0000{1..5}-of-00005.safetensors", "file_bytes": 16_381_516_776, "header_bytes": 46_056,
        "linear_params": 6_945_767_424, "embedding_params": 1_244_659_712, "norm_params": 308_224,
        "source": "safetensors headers of revision b968826 (5 shards: embed_tokens + lm_head, norms, "
                  "252 Linear)",
        # the recipes fit GPTQ on every layer at once (sequential: false); the fp64 Hessians of 8B take
        # ~58 GB, over tricast's 4 GiB default budget, so the measurement raised it (tune_outputs.py)
        "calibration_note": "GPTQ fits every layer at once (`sequential: false`, as measured): its fp64 "
                            "Hessians need ~58 GB, over tricast's 4 GiB default, so run it through "
                            "`tricast run` with `hessian_max_bytes: 103079215104` on a GPU with that much "
                            "free memory (the measurement used an H200).",
    },
}


def _rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Every measured row of the bundle: knapsack solutions, uniform recipes, option transfers, proposals."""
    rows: list[dict[str, Any]] = []
    for key in ("solutions", "baselines", "transfer"):
        rows += bundle.get(key) or []
    for round_ in (bundle.get("session") or {}).get("rounds") or []:
        rows += round_.get("proposals") or []
    return rows


def _row(bundle: dict[str, Any], ident: str | None) -> dict[str, Any] | None:
    if ident is None:
        return None
    return next((r for r in _rows(bundle) if (r.get("id") or r.get("name")) == ident), None)


def ppl_record(bundle: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any] | None:
    """The PPL of the recipe's measured row with the record of that measurement: the row's ``ppl_record``,
    or for a session proposal the session record its ``ppl_id`` names."""
    row = _row(bundle, (recipe.get("measured") or {}).get("id"))
    if row is None or row.get("ppl") is None:
        return None
    records = (bundle.get("session") or {}).get("records") or {}
    record = row.get("ppl_record") or records.get(row.get("ppl_id"))
    keys = ("gpu", "git_sha", "git_dirty", "n_tokens", "source")
    return {"ppl": row["ppl"], **{key: (record or {}).get(key) for key in keys}}


def linear_bits(bundle: dict[str, Any], recipe: dict[str, Any], grid: list[list[str]] | None) -> float | None:
    """Average stored bits of the Linear weights of a setup: the measured row's when the recipe is a recorded
    one (its options can change a group size), else computed from the grid on the table."""
    row = _row(bundle, (recipe.get("measured") or {}).get("id"))
    if row is not None:
        bits = row.get("bits") if row.get("bits") is not None else row.get("avg_bits")
        if bits is not None:
            return float(bits)
    if grid is None:
        return None
    table = bundle["table"]
    layers = [at.Layer(e["name"], e["block"], e["kind"], e["params"], e["in_features"])
              for e in table["layers"]]
    methods = [grid[layer.block][table["kinds"].index(layer.kind)] for layer in layers]
    return float(at.avg_bits(layers, methods))


def model_size(model_id: str, bits: float) -> dict[str, Any] | None:
    """The original file and this setup's low-bit file, by part, in bytes (computed); None for a model whose
    checkpoint is not in ``MODEL_FILES``."""
    model = MODEL_FILES.get(model_id)
    if model is None:
        return None
    linear, embedding, norm = model["linear_params"], model["embedding_params"], model["norm_params"]
    original = [{"part": "linear", "params": linear, "bits": 16, "bytes": 2 * linear},
                {"part": "embedding", "params": embedding, "bits": 16, "bytes": 2 * embedding},
                {"part": "norm", "params": norm, "bits": 16, "bytes": 2 * norm}]
    quantized = round(Fraction(linear) * Fraction(bits) / 8)
    setup = [{**original[0], "bits": bits, "bytes": quantized}, *original[1:]]
    total = sum(part["bytes"] for part in setup) + model["header_bytes"]
    return {"model": model_id, "revision": model["revision"], "original_bytes": model["file_bytes"],
            "original": original, "setup": setup, "setup_bytes": total, "ratio": total / model["file_bytes"],
            "linear_ratio": bits / 16, "computed": True, "source": model["source"]}


def _readme(recipe: dict[str, Any], ppl: dict[str, Any] | None, live: dict[str, Any] | None,
            size: dict[str, Any] | None, model_id: str) -> str:
    file = recipe["file"]
    lines = [f"# {recipe['name']} — TriCast recipe bundle", "",
             f"Model: {model_id}. Recipe: `{file}` (SHA-256 `{recipe['sha256']}`).", ""]
    if ppl is not None:
        same = ("the very file measured" if (recipe.get("measured") or {}).get("same_file")
                else "NOT the file measured")
        where = f"{ppl.get('gpu') or 'GPU'}, git {str(ppl.get('git_sha') or '')[:7]}"
        lines += [f"Measured on GPU: WikiText-2 test PPL {ppl['ppl']:.2f} ({where}); the recipe file is "
                  f"{same}. See `evaluation.json`.", ""]
    if live is not None:  # PPL when the job measured it, the job's tasks either way (scores in the json)
        gpu = (live.get("env") or {}).get("gpu") or "GPU"
        run = (live.get("results") or {}).get("run") or {}
        ppl = f"WikiText-2 test PPL {run['ppl']:.2f}, " if run.get("ppl") is not None else ""
        tasks = ", ".join(live.get("tasks") or [])
        lines += [f"Evaluated on this server's GPU ({gpu}): {ppl}tasks {tasks}, job `{live['job']}` "
                  "(`evaluation.json`, `this_server_run`).", ""]
    if size:
        lines += [f"Size (computed, `size.json`): {size['setup_bytes'] / 1e9:.2f} GB as a low-bit file vs "
                  f"{size['original_bytes'] / 1e9:.2f} GB for the original BF16 checkpoint "
                  f"({100 * size['ratio']:.0f}%; Linear weights at {size['setup'][0]['bits']:.3f} bits, "
                  "embedding and norms in BF16). TriCast does not write that low-bit file yet: applying the "
                  "recipe gives a fake-quant model in BF16.", ""]
    lines += ["## Apply", "", "```bash", "pip install -e \".[eval]\"   # in the TriCast repository",
              f"tricast recipe-check {file}", f"tricast ppl --model {model_id} --recipe {file}",
              f"tricast eval --model {model_id} --recipe {file} --tasks hellaswag,arc_easy   # lm-eval",
              "```", "", "```python", "import tricast", f"recipe = tricast.load_recipe(\"{file}\")",
              "tricast.patch_model(model, recipe)   # model: transformers AutoModelForCausalLM",
              "tricast.calibrate(model, recipe, tokenizer=tokenizer)   # GPTQ / AWQ recipes need calibration",
              "```", ""]
    note = (MODEL_FILES.get(model_id) or {}).get("calibration_note")
    if note:
        lines += [note, ""]
    return "\n".join(lines)


def bundle_zip(bundle: dict[str, Any], recipe: dict[str, Any], bits: float | None,
               live: dict[str, Any] | None = None) -> bytes:
    """The zip the page downloads for one setup; ``live``: a finished GPU job of this server on this very
    recipe text ({job, results, env, tasks}), or None."""
    model_id = bundle["table"]["meta"]["model_id"]
    size = None if bits is None else model_size(model_id, bits)
    ppl = ppl_record(bundle, recipe)
    native = bundle.get("native") or {}
    evaluation = {"setup": recipe["name"], "recipe_file": recipe["file"], "recipe_sha256": recipe["sha256"],
                  "same_file_as_measured": (recipe.get("measured") or {}).get("same_file"),
                  "measured_ppl": ppl, "native_ppl": native.get("ppl"),
                  "native_record": {k: native.get(k) for k in ("gpu", "git_sha", "n_tokens", "source")},
                  "metric": "WikiText-2 test perplexity, 2048-token windows (not the data picked on)",
                  "this_server_run": live}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(recipe["file"], recipe["yaml"])
        zf.writestr("evaluation.json", json.dumps(evaluation, ensure_ascii=False, indent=1) + "\n")
        if size is not None:
            zf.writestr("size.json", json.dumps(size, ensure_ascii=False, indent=1) + "\n")
        zf.writestr("README.md", _readme(recipe, ppl, live, size, model_id))
    return buffer.getvalue()
