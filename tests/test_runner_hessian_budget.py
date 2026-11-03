"""`tricast run` config key hessian_max_bytes reaches tricast.calibration.calibrate (GPTQ's live-Hessian
storage budget), is refused when it is not a nonnegative integer, and leaves the run identity unchanged."""

import pytest

from tests.test_official_prune import _tiny, _tokenizer
from tricast import calibration

REAL_CALIBRATE = calibration.calibrate

runner = pytest.importorskip("tricast.eval.runner")

GPTQ = {"name": "gptq", "defaults": {"weight": "int4_g128_zp", "activation": None, "weight_algo": "gptq",
                                     "mma": {"preset": "fp64", "out_format": "fp32"}},
        "include": ["model.layers.0.self_attn.q_proj"]}


def _run(tmp_path, monkeypatch, **extra):
    from tricast.eval.envinfo import capture_env

    model, tokenizer = _tiny("llama"), _tokenizer()
    monkeypatch.setattr(runner, "_load_model", lambda *args: (model, tokenizer))
    env = capture_env(None, {"model_id": "offline/tiny", "model_sha": "model-commit", "git_sha": "a" * 40,
                             "git_dirty": False, "src_sha256": None, "seed": 42, "dtype": "fp32",
                             "device": "cpu"})
    monkeypatch.setattr(runner, "capture_env", lambda *args: env.copy())
    calls = []

    def spy(model, recipe, **kwargs):  # the real calibration on an inline text instead of WikiText-2
        calls.append(kwargs)
        text = " ".join(f"t{i % 200}" for i in range(64))
        return REAL_CALIBRATE(model, recipe, texts=[text], samples=2, seqlen=8, **kwargs)

    monkeypatch.setattr(calibration, "calibrate", spy)  # run_config imports it at call time
    cfg = {"model": "offline/tiny", "dtype": "fp32", "device": "cpu", "seed": 42, "recipes": [GPTQ],
           "tasks": {"ppl": {"texts": ["t0 t1 t2 t3"], "seqlen": 4}}, "output_dir": str(tmp_path), **extra}
    return runner.run_config(cfg), calls


def test_the_budget_reaches_calibrate_and_not_the_run_identity(tmp_path, monkeypatch):
    default, calls = _run(tmp_path / "a", monkeypatch)
    assert "hessian_max_bytes" not in calls[0]  # calibrate's own default (4 GiB) applies
    raised, calls = _run(tmp_path / "b", monkeypatch, hessian_max_bytes=16 << 30)
    assert calls[0]["hessian_max_bytes"] == 16 << 30
    assert raised[0]["calibration"]["hessian_max_bytes"] == 16 << 30
    assert raised[0]["run_hash"] == default[0]["run_hash"]  # a memory limit, not a numerical setting


def test_the_budget_is_enforced_by_calibrate(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exceeding hessian_max_bytes=1 "):
        _run(tmp_path, monkeypatch, hessian_max_bytes=1)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "4GiB"])
def test_a_bad_budget_is_refused(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError, match="hessian_max_bytes must be a nonnegative integer"):
        _run(tmp_path, monkeypatch, hessian_max_bytes=bad)
