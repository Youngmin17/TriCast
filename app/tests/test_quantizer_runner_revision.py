"""SPEC R7 finding (2026-11-03): a Quantizer job on a hub model recorded model_sha None while `tricast run`
recorded the loaded revision; the job's records must carry the revision the model was loaded at (AGENTS.md
§3.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

import tricast.eval
from app import quantizer_recipe as qr
from app.runners.quantizer import QuantizerArgs, run_quantizer
from app.tests.test_quantizer_runner import TEXT, tiny_qwen3


def test_a_hub_model_job_records_the_loaded_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture = tricast.eval.capture_env

    def offline_env(model_id, extra):  # offline hub: the capture cannot resolve the revision
        env = capture(str(tmp_path), extra)
        return {**env, "model_id": model_id, "model_kind": "hf", "model_sha": None}

    def loaded(*args):
        model, tokenizer = tiny_qwen3(*args)
        model.config._commit_hash = "c" * 40
        return model, tokenizer

    monkeypatch.setattr(tricast.eval, "capture_env", offline_env)
    recipe = qr.build("qz_rev", "nvfp4", qr.uniform("nvfp4"))
    args = QuantizerArgs(name="qz_rev", recipe=recipe, tasks=("ppl",), limit=None, device="cpu",
                         out_dir=tmp_path / "out", model_id="Qwen/Qwen3-0.6B",
                         ppl_seqlen=32, report_samples=1, report_seqlen=32, text=TEXT)
    result = run_quantizer(args, lambda *a, **k: None, model_factory=loaded)
    assert result["env"]["model_sha"] == "c" * 40
    assert result["metrics"]["run"]["env"]["model_sha"] == "c" * 40
