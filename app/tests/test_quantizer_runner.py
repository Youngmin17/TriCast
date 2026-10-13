"""CPU checks of app/runners/quantizer.py with a tiny random Qwen3 injected through the model factory: the
records are CLI-shaped and the results page-shaped, calibration settings reach tricast the way the CLI's
--calib-* flags do, and the page's Python API snippet only calls functions that exist. No Qwen3-0.6B, no GPU,
no dataset download (the calibrating test wraps tricast.calibrate to read its windows from the test text).
Like test_runners_cpu.py, the model tests need transformers.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

import tricast
from app import export_layers
from app import quantizer_recipe as qr
from app.runners.quantizer import QuantizerArgs, run_quantizer
from tricast.recipe import load_recipe

REPO = Path(qr.__file__).resolve().parents[1]
TEXT = " ".join(f"token{i % 37} word{i % 11}." for i in range(120))


class ByteTokenizer:
    """Enough of a tokenizer for perplexity, layer_report and calibration: text -> {"input_ids": [[...]]}."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([[ord(c) % 128 for c in text]])}


def tiny_qwen3(model_id: str, dtype: str, device: str, revision: str | None):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    cfg = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    return transformers.Qwen3ForCausalLM(cfg).eval(), ByteTokenizer()


def test_runner_writes_cli_shaped_records_and_page_shaped_results(tmp_path: Path) -> None:
    grid = qr.uniform("mxfp4")
    for block in range(qr.BLOCKS):
        grid[block][qr.KINDS.index("down_proj")] = "mxfp8"  # one kind rule: "*.mlp.down_proj"
    recipe = qr.build("qz_cpu", "mxfp4", grid)
    model_dir = tmp_path / "tiny"  # an existing path: capture_env records it as a local model, no Hub lookup
    model_dir.mkdir()
    stages: list[tuple[str, float]] = []
    args = QuantizerArgs(
        name="qz_cpu",
        recipe=recipe,
        tasks=("ppl", "report"),
        limit=None,
        device="cpu",
        out_dir=tmp_path / "out",
        model_id=str(model_dir),
        ppl_seqlen=32,
        report_samples=2,
        report_seqlen=32,
        text=TEXT,
    )
    result = run_quantizer(
        args, lambda stage, fraction, partial=None: stages.append((stage, fraction)), model_factory=tiny_qwen3
    )
    json.dumps(result, allow_nan=False)  # what the job queue stores
    assert [stage for stage, _ in stages] == [
        "모델 불러오기",
        "레이어별 리포트",
        "패치",
        "WikiText-2 PPL",
        "정리",
    ]
    assert all(a <= b for (_, a), (_, b) in zip(stages, stages[1:], strict=False))
    run, report = result["metrics"]["run"], result["metrics"]["report"]
    assert result["metrics"]["lmeval"] is None
    assert math.isfinite(run["ppl"]) and run["ppl"] > 1 and run["n_windows"] == len(TEXT) // 32
    sha = load_recipe(recipe).sha256
    assert run["recipe_hash"] == report["recipe_hash"] == result["env"]["recipe_hash"] == sha
    measured = [i for i, v in enumerate(report["sqnr_db"]) if v is not None]
    assert len(report["sqnr_db"]) == 196 and measured == list(range(14))  # the tiny model has blocks 0 and 1
    assert report["measured"] == sorted(qr.layer_name(b, k) for b in range(2) for k in qr.KINDS)
    record = json.loads((tmp_path / "out" / "eval" / "qz_cpu.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete" and record["recipe"]["name"] == "qz_cpu"
    assert record["patch_report"]["patched"] and record["env"]["model_kind"] == "local"
    formats = {line.split("weight=")[1].split(",")[0] for _, line in record["patch_report"]["patched"]}
    assert formats == {"fp4_e2m1", "fp8_e4m3"}  # mxfp4 everywhere, mxfp8 on down_proj: the patch fired
    report_path = tmp_path / "out" / "report" / "qz_cpu" / "report.json"
    assert export_layers.load_report(report_path)["status"] == "complete"  # the layer-map exporter takes it
    assert result["evidence"]["layer_map_command"] == (
        f"python -m app.export_layers {tmp_path / 'out' / 'report'} --out app/web/demo/layers"
    )


def test_python_api_snippet_names_exist_with_these_arguments() -> None:
    """Every call in the page's Python snippet (app/web/js/quantizer.js API_SNIPPET) binds to a real function.
    """
    from tricast.eval import evaluate, perplexity

    snippet = (REPO / "app" / "web" / "js" / "quantizer.js").read_text(encoding="utf-8")
    for text in (
        "from tricast import load_recipe, patch_model, calibrate",
        "from tricast.eval import perplexity, evaluate",
        "recipe.needs_calibration",
        'calibrate(model, recipe, tokenizer=tok, device="cuda")',
        'perplexity(model, tok, device="cuda")["ppl"]',
        "evaluate(model, tok, [",
        'limit=300)["results"]',
        'torch_dtype="bfloat16"',
    ):
        assert text in snippet, text
    model, tok, recipe = object(), object(), object()
    inspect.signature(tricast.load_recipe).bind("my_recipe.yaml")
    inspect.signature(tricast.patch_model).bind(model, recipe)
    inspect.signature(tricast.calibrate).bind(model, recipe, tokenizer=tok, device="cuda")
    inspect.signature(perplexity).bind(model, tok, device="cuda")
    inspect.signature(evaluate).bind(model, tok, ["arc_easy"], limit=300)
    assert isinstance(inspect.getattr_static(tricast.recipe.Recipe, "needs_calibration"), property)


def test_snippet_model_load_accepts_a_string_dtype(tmp_path: Path) -> None:
    """from_pretrained(..., torch_dtype="bfloat16") as the snippet writes it, on a saved tiny Qwen3."""
    transformers = pytest.importorskip("transformers")
    model, _ = tiny_qwen3("tiny", "float32", "cpu", None)
    model.save_pretrained(tmp_path)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype="bfloat16")
    assert next(loaded.parameters()).dtype == torch.bfloat16


@pytest.mark.parametrize("task", ["report", "ppl"])
def test_one_task_only(tmp_path: Path, task: str) -> None:
    model_dir = tmp_path / "tiny"
    model_dir.mkdir()
    args = QuantizerArgs(
        name="qz_one",
        recipe=qr.build("qz_one", "nvfp4", qr.uniform("nvfp4")),
        tasks=(task,),
        limit=None,
        device="cpu",
        out_dir=tmp_path / "out",
        model_id=str(model_dir),
        ppl_seqlen=32,
        report_samples=1,
        report_seqlen=32,
        text=TEXT,
    )
    result = run_quantizer(args, lambda *a, **k: None, model_factory=tiny_qwen3)
    assert (result["metrics"]["run"] is None) == (task != "ppl")
    assert (result["metrics"]["report"] is None) == (task != "report")


# ---------- calibration settings (the CLI's --calib-* flags) ----------
def args_for(tmp_path: Path, recipe: dict, calibration: dict | None) -> QuantizerArgs:
    model_dir = tmp_path / "tiny"
    model_dir.mkdir(exist_ok=True)
    return QuantizerArgs(
        name=recipe["name"],
        recipe=recipe,
        tasks=("ppl",),
        limit=None,
        device="cpu",
        out_dir=tmp_path / "out",
        model_id=str(model_dir),
        ppl_seqlen=32,
        text=TEXT,
        calibration=calibration,
    )


def merged(recipe: dict, calibration: dict) -> tricast.recipe.Recipe:
    loaded = load_recipe(recipe)
    return load_recipe({**loaded.to_dict(), "calibration": {**(loaded.calibration or {}), **calibration}})


def test_settings_merge_over_the_recipe_calibration(tmp_path: Path) -> None:
    recipe = qr.build("qz_cal", "mxfp4", qr.uniform("mxfp4"), calibration={"samples": 64})
    settings = {"samples": 3, "seqlen": 16, "seed": 5}
    result = run_quantizer(
        args_for(tmp_path, recipe, settings), lambda *a, **k: None, model_factory=tiny_qwen3
    )
    record = json.loads((tmp_path / "out" / "eval" / "qz_cal.json").read_text(encoding="utf-8"))
    want = merged(recipe, settings)
    assert record["recipe"]["calibration"] == {"samples": 3, "seed": 5, "seqlen": 16}
    assert record["calibration_config"] == want.calibration_options == record["env"]["calibration_config"]
    assert result["env"]["recipe_hash"] == record["recipe_hash"] == want.sha256 != load_recipe(recipe).sha256
    assert record["calibration"] is None  # mxfp4 needs no calibration: the settings are recorded, not used


def test_a_static_observer_calibrates_with_the_requested_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tricast.calibrate
    calls: list[dict] = []

    def from_text(model, recipe, **kwargs):
        calls.append(recipe.calibration_options)
        return real(model, recipe, texts=[TEXT], **kwargs)

    monkeypatch.setattr(tricast, "calibrate", from_text)
    options = {
        "int8": {
            "activation.granularity": "tensor",
            "activation.observer.kind": "ema",
            "activation.observer.decay": 0.5,
        }
    }
    recipe = qr.build("qz_ema", "int8", qr.uniform("int8"), options=options)
    assert recipe["calibration"] == {"seed": 42}  # the observer turned calibration on by itself
    settings = {"samples": 4, "seqlen": 24, "sequential": True}
    result = run_quantizer(
        args_for(tmp_path, recipe, settings), lambda *a, **k: None, model_factory=tiny_qwen3
    )
    record = json.loads((tmp_path / "out" / "eval" / "qz_ema.json").read_text(encoding="utf-8"))
    assert len(calls) == 1 and (calls[0]["samples"], calls[0]["seqlen"], calls[0]["seed"]) == (4, 24, 42)
    report = record["calibration"]
    assert (report["samples"], report["n_tokens"], report["sequential"]) == (4, 96, True)
    assert record["patch_report"]["patched"] and result["metrics"]["run"]["ppl"] > 1
