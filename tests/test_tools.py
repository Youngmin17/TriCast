"""Agent tools return deterministic, attributable JSON evidence."""

import json

import pytest

from tricast.tools import catalog, errors, runs


def test_catalog_describes_presets_and_unknown_names() -> None:
    result = catalog.describe("nvidia_hopper_fp8")
    assert result["is_error"] is False
    assert result["kind"] == "preset"
    assert result["details"]["f_bits"] == 13
    assert "NADPE" in result["details"]["provenance"]
    assert result["sources"]
    assert catalog.describe("not_a_format")["is_error"] is True
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["format", "scheme", "preset", "recipe"])
def test_list_options_and_describe(kind: str) -> None:
    options = catalog.list_options(kind)
    assert options["is_error"] is False
    assert options["options"] == sorted(options["options"])
    for name in options["options"]:
        assert catalog.describe(name)["is_error"] is False
    assert catalog.list_options("missing")["is_error"] is True


def test_quantization_error_is_deterministic_and_local_rng() -> None:
    import torch

    before = torch.random.get_rng_state().clone()
    first = errors.quantization_error("mxfp4", n=128, seed=42)
    second = errors.quantization_error("mxfp4", n=128, seed=42)
    assert first == second
    assert torch.equal(before, torch.random.get_rng_state())
    assert first["bits_per_element"] == 4.25
    assert first["max_abs_err"] > 0
    assert first["env"]["backend"] == "reference"
    assert errors.quantization_error("mxfp8_e4m3", n=128, seed=42)["sqnr_db"] > first["sqnr_db"]
    json.dumps(first, allow_nan=False)


def test_quantization_storage_includes_tail_and_two_level_scales() -> None:
    assert errors.quantization_error("mxfp4", n=33)["bits_per_element"] == (33 * 4 + 2 * 8) / 33
    assert errors.quantization_error("nvfp4", n=32)["bits_per_element"] == 5.5
    exact = errors.quantization_error("fp32", n=8)
    assert exact["sqnr_db"] is None
    assert exact["sqnr_status"] == "infinite"
    assert exact["max_abs_err"] == 0
    json.dumps(exact, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"scheme": "missing"}, {"scheme": "mxfp4", "n": 0},
                                   {"scheme": "mxfp4", "n": 65537},
                                   {"scheme": "mxfp4", "source": "remote_dataset"}])
def test_quantization_error_rejects_invalid_inputs(kwargs: dict) -> None:
    assert errors.quantization_error(**kwargs)["is_error"] is True


def test_dry_run_never_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    from tricast.eval import runner

    def forbidden(_cfg: dict) -> list:
        pytest.fail("dry run executed the evaluator")

    monkeypatch.setattr(runner, "run_config", forbidden)
    cfg = {"model": "Qwen/Qwen3-0.6B", "recipes": ["hopper_fp8_w8a8"],
           "tasks": {"ppl": {"max_windows": 1}}}
    result = runs.run_eval(cfg)
    assert result["is_error"] is False
    assert result["execute"] is False
    assert result["plan"]["recipes"][0]["name"] == "hopper_fp8_w8a8"
    assert "cost_estimate" in result
    json.dumps(result, allow_nan=False)


def _record() -> dict:
    return {"status": "complete", "run_hash": "run", "recipe_hash": "recipe",
            "recipe": {"name": "test"}, "metrics": {"ppl": {"ppl": 2.0}},
            "env": {"git_sha": "abc", "model_sha": "model"}, "wall_time_s": 0.1}


def test_run_eval_preserves_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from tricast.eval import runner

    record = _record()
    monkeypatch.setattr(runner, "run_config", lambda _cfg: [record])
    cfg = {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"ppl": {}},
           "output_dir": str(tmp_path)}
    result = runs.run_eval(cfg, execute=True)
    assert result["is_error"] is False
    assert result["results"][0]["env"] == record["env"]
    assert result["results"][0]["metrics"] == record["metrics"]
    assert result["paths"] == [str(tmp_path / "test.json")]


def test_run_eval_executes_exactly_the_materialized_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    from tricast.eval import runner

    calls = []
    monkeypatch.setattr(runner, "run_config", lambda cfg: calls.append(cfg) or [_record()])
    cfg = {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"ppl": {}},
           "dtype": "fp16", "device": "cpu", "seed": 17}
    result = runs.run_eval(cfg, execute=True)
    assert not result["is_error"]
    assert calls == [result["plan"]]
    assert result["plan"]["dtype"] == "fp16" and result["plan"]["seed"] == 17
    assert result["plan"]["device"] == "cpu"
    assert isinstance(result["plan"]["recipes"][0], dict)
    assert cfg["recipes"] == ["fp64_reference"]


@pytest.mark.parametrize("changes", [
    {"recipes": [{"name": "../escape"}]}, {"recipes": [{"name": "env"}]},
    {"dtype": "garbage"}, {"tasks": {"ppl": {"max_windows": 0}}},
    {"tasks": {"ppl": {"seqlen": 1}}}, {"tasks": {"ppl": {"device": "cpu"}}},
    {"tasks": {"lm_eval": {"tasks": []}}},
    {"tasks": {"lm_eval": {"tasks": ["piqa"], "limit": -1}}}, {"seed": -1},
    {"tasks": {"ppl": {"batch_size": None}}},
])
def test_dry_run_rejects_invalid_execution_inputs(changes: dict) -> None:
    config = {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"ppl": {}}, **changes}
    result = runs.run_eval(config)
    assert result["is_error"] is True


def test_compare_runs_preserves_env_and_does_not_assert_fairness(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_record()))
    result = runs.compare_runs([str(path)])
    assert result["is_error"] is False
    assert result["runs"][0]["env"] == _record()["env"]
    assert result["comparable"] is False
    assert result["warnings"]
    path.write_text(json.dumps({"token": "secret_should_not_be_echoed"}))
    failed = runs.compare_runs([str(path)])
    assert failed["is_error"] is True
    assert "secret_should_not_be_echoed" not in json.dumps(failed)


@pytest.mark.parametrize("metrics", [{"ppl": {}}, {"ppl": {"ppl": -1}}, {"ppl": {"ppl": True}},
                                     {"lm_eval": "not-metrics"}, {"lm_eval": {"results": {}}}])
def test_compare_runs_rejects_broken_metrics(tmp_path, metrics: dict) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({**_record(), "metrics": metrics}))
    assert runs.compare_runs([str(path)])["is_error"] is True


def test_run_eval_does_not_echo_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from tricast.eval import runner

    def fail(_config: dict) -> list:
        raise RuntimeError("Authorization: Bearer private_sentinel")

    monkeypatch.setattr(runner, "run_config", fail)
    cfg = {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"ppl": {}}}
    result = runs.run_eval(cfg, execute=True)
    assert result["is_error"] is True
    assert "RuntimeError" in result["error"]
    assert "private_sentinel" not in json.dumps(result)


def test_exposed_tool_schemas_are_strict() -> None:
    from tricast.rag import SEARCH_SCHEMA

    for schema in (catalog.DESCRIBE_SCHEMA, catalog.LIST_OPTIONS_SCHEMA,
                   errors.QUANTIZATION_ERROR_SCHEMA, runs.COMPARE_RUNS_SCHEMA, SEARCH_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def _report_config(**changes: object) -> dict:
    return {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"report": {}}, **changes}


def test_layer_report_dry_run_never_imports_execution_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from tricast.tools import reports

    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name in ("analysis", "tricast.analysis", "transformers", "datasets", "eval.envinfo",
                    "eval.runner"):
            pytest.fail(f"dry run imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    config = _report_config()
    result = reports.layer_report(config)
    assert not result["is_error"] and result["execute"] is False and result["needs_inputs"]
    assert result["plan"]["tasks"] == {"report": {"samples": 8, "seqlen": 128}}
    assert result["plan"]["recipes"][0]["name"] == "fp64_reference"
    assert "results" not in result
    assert config == _report_config()
    json.dumps(result, allow_nan=False)
    missing = reports.layer_report(config, execute=True)
    assert missing["is_error"] and "supply texts or input_ids" in missing["error"]


@pytest.mark.parametrize("changes", [
    {"model": " "}, {"recipes": []}, {"recipes": ["fp64_reference", "fp64_reference"]},
    {"dtype": "bad"}, {"seed": True}, {"seed": -1}, {"tasks": {"ppl": {}}},
    {"tasks": {"report": {"texts": []}}}, {"tasks": {"report": {"texts": [" "]}}},
    {"tasks": {"report": {"texts": ["ok"], "input_ids": [[1, 2]]}}},
    {"tasks": {"report": {"input_ids": [[1, 2], [1, 2, 3]]}}},
    {"tasks": {"report": {"input_ids": [[1.0, 2]]}}},
    {"tasks": {"report": {"input_ids": [[True, 2]]}}},
    {"tasks": {"report": {"input_ids": [[-1, 2]]}}},
    {"tasks": {"report": {"dataset": "implicit_download"}}},
    {"tasks": {"report": {"samples": 0}}}, {"tasks": {"report": {"seqlen": 1}}},
    {"tasks": {"report": {"samples": 1.0}}}, {"tasks": {"report": {"seqlen": True}}},
    {"tasks": {"report": {"input_ids": [[1, 2]], "seqlen": 3}}},
])
def test_layer_report_rejects_invalid_plans(changes: dict) -> None:
    from tricast.tools import reports

    assert reports.layer_report(_report_config(**changes))["is_error"]


@pytest.mark.parametrize("task", [{"texts": ["tiny text"]}, {"input_ids": [[1, 2, 3]]}])
def test_layer_report_dispatch_preserves_inputs_and_evidence(
    monkeypatch: pytest.MonkeyPatch, task: dict,
) -> None:
    import sys
    from types import SimpleNamespace

    import torch

    from tricast.eval import envinfo, runner
    from tricast.tools import reports

    calls = []
    model, tokenizer = SimpleNamespace(config=SimpleNamespace(_commit_hash="model-sha")), object()
    analysis_result = {"layers": [{"name": "projection", "output": {"mse": 0.25}}],
                       "env": {"git_sha": "source-sha", "model_sha": "model-sha"}}

    def load(model_id: str, dtype: str, device: str) -> tuple:
        calls.append((model_id, dtype, device))
        return model, tokenizer

    def analyze(actual_model: object, recipe: object, **kwargs: object) -> dict:
        assert actual_model is model and recipe.name == "fp64_reference"
        assert kwargs["tokenizer"] is tokenizer
        if "texts" in task:
            assert kwargs["texts"] == task["texts"] and "input_ids" not in kwargs
            assert kwargs["samples"] == 8 and kwargs["seqlen"] == 128
        else:
            assert kwargs["input_ids"].dtype == torch.long
            assert kwargs["input_ids"].tolist() == task["input_ids"] and "texts" not in kwargs
            assert kwargs["samples"] == 1 and kwargs["seqlen"] == 3
        return analysis_result

    monkeypatch.setattr(runner, "_load_model", load)
    monkeypatch.setitem(sys.modules, "tricast.analysis", SimpleNamespace(layer_report=analyze))
    monkeypatch.setattr(envinfo, "capture_env", lambda **kw: pytest.fail("existing environment lost"))
    result = reports.layer_report(_report_config(tasks={"report": task}, dtype="fp32", device="cpu", seed=42),
                                  execute=True)
    assert not result["is_error"] and not result["needs_inputs"]
    assert calls == [("org/tiny", "fp32", "cpu")]
    record = result["results"][0]
    assert record["recipe"]["name"] == "fp64_reference"
    assert record["metrics"]["report"]["layers"] == analysis_result["layers"]
    assert record["env"] == analysis_result["env"] and "env" in analysis_result
    assert "env" not in record["metrics"]["report"]
    json.dumps(result, allow_nan=False)


def test_layer_report_captures_missing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from tricast.eval import envinfo, runner
    from tricast.tools import reports

    captured = []

    def capture(**kwargs: object) -> dict:
        captured.append(kwargs)
        return {"git_sha": "source-sha", **kwargs["extra"]}

    monkeypatch.setattr(runner, "_load_model", lambda *a: (
        SimpleNamespace(config=SimpleNamespace(_commit_hash="loaded-sha")), object(),
    ))
    monkeypatch.setattr(envinfo, "capture_env", capture)
    monkeypatch.setitem(sys.modules, "tricast.analysis", SimpleNamespace(layer_report=lambda *a, **kw: {
        "layers": [], "logits_kl": 0.0,
    }))
    result = reports.layer_report(_report_config(tasks={"report": {"input_ids": [[1, 2]]}}), execute=True)
    assert not result["is_error"] and len(captured) == 1
    env = result["results"][0]["env"]
    assert env["model_id"] == "org/tiny" and env["model_sha"] == "loaded-sha"
    assert env["seed"] == 42 and len(env["input_fingerprint"]) == 64
    assert env["recipe_hash"] == result["results"][0]["recipe_hash"]


def test_layer_report_execution_errors_do_not_echo_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from tricast.eval import runner
    from tricast.tools import reports

    def fail(*args: object) -> None:
        raise RuntimeError("Authorization: Bearer private_sentinel")

    monkeypatch.setattr(runner, "_load_model", fail)
    monkeypatch.setitem(sys.modules, "tricast.analysis", SimpleNamespace(layer_report=lambda *a, **kw: {}))
    result = reports.layer_report(_report_config(tasks={"report": {"texts": ["tiny text"]}}), execute=True)
    assert result["is_error"] and "RuntimeError" in result["error"]
    assert "private_sentinel" not in json.dumps(result) and "results" not in result


def test_layer_report_tool_schema_is_strict_and_locally_bounded() -> None:
    from jsonschema import Draft202012Validator

    from tricast.tools.reports import LAYER_REPORT_SCHEMA

    assert LAYER_REPORT_SCHEMA["additionalProperties"] is False
    assert set(LAYER_REPORT_SCHEMA["required"]) == set(LAYER_REPORT_SCHEMA["properties"])
    validator = Draft202012Validator(LAYER_REPORT_SCHEMA)
    valid = {"model": "org/tiny", "recipe": "fp64_reference", "texts": None, "input_ids": [[1, 2]]}
    assert validator.is_valid(valid)
    assert not validator.is_valid({**valid, "input_ids": [[-1, 2]]})
    assert not validator.is_valid({**valid, "texts": ["x"] * 65})


@pytest.mark.parametrize("use_texts", [False, True])
def test_layer_report_tool_runs_tiny_model_without_download(
    monkeypatch: pytest.MonkeyPatch, use_texts: bool,
) -> None:
    pytest.importorskip("tricast.analysis")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tricast.eval import envinfo, runner
    from tricast.tools import reports

    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=1, max_position_embeddings=16)).eval()
    class Tokenizer:
        def __call__(self, text: str, *, return_tensors: str) -> dict:
            assert text == "a short tiny input" and return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    monkeypatch.setattr(runner, "_load_model", lambda *a: (model, Tokenizer() if use_texts else None))
    monkeypatch.setattr(envinfo, "capture_env", lambda **kw: {"git_sha": "test", **kw.get("extra", {})})
    task = ({"texts": ["a short tiny input"], "samples": 1, "seqlen": 4} if use_texts
            else {"input_ids": [[1, 2, 3, 4]]})
    result = reports.layer_report(_report_config(tasks={"report": task},
                                                 dtype="fp32", device="cpu"), execute=True)
    assert not result["is_error"], result.get("error")
    record = result["results"][0]
    assert record["metrics"]["report"] and record["env"]
    assert result["plan"]["tasks"]["report"]["samples"] == 1
    assert result["plan"]["tasks"]["report"]["seqlen"] == 4
    assert record["metrics"]["report"]["samples"] == 1
    assert record["metrics"]["report"]["seqlen"] == 4
    assert record["metrics"]["report"]["layers"]
    assert record["metrics"]["report"]["model"]["ppl_reference"] > 0
    json.dumps(result, allow_nan=False)


def test_layer_report_materializes_explicit_and_derived_windows() -> None:
    from tricast.tools import reports

    ids = [[1, 2, 3, 4], [4, 3, 2, 1]]
    for changes, expected in (({}, (2, 4)), ({"seqlen": 2}, (4, 2)),
                              ({"samples": 1, "seqlen": 2}, (1, 2))):
        result = reports.layer_report(_report_config(tasks={"report": {"input_ids": ids, **changes}}))
        assert not result["is_error"]
        task = result["plan"]["tasks"]["report"]
        assert (task["samples"], task["seqlen"]) == expected


@pytest.mark.parametrize("analysis_result", [{}, {"env": {"git_sha": "source"}}, {"is_error": False}])
def test_layer_report_rejects_empty_analysis(monkeypatch: pytest.MonkeyPatch, analysis_result: dict) -> None:
    import sys
    from types import SimpleNamespace

    from tricast.eval import runner
    from tricast.tools import reports

    monkeypatch.setattr(runner, "_load_model", lambda *a: (object(), object()))
    monkeypatch.setitem(sys.modules, "tricast.analysis", SimpleNamespace(
        layer_report=lambda *a, **kw: analysis_result,
    ))
    result = reports.layer_report(_report_config(tasks={"report": {"input_ids": [[1, 2]]}}), execute=True)
    assert result["is_error"] and "results" not in result


def test_layer_report_rejects_seed_not_used_by_analysis() -> None:
    from tricast.tools import reports

    result = reports.layer_report(_report_config(seed=7))
    assert result["is_error"]
    assert "fixes analysis RNG to 42" in result["error"]


def test_default_run_directories_are_unique_even_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from datetime import datetime, timezone
    from pathlib import Path
    from types import SimpleNamespace

    instant = datetime(2026, 9, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(runs, "datetime", SimpleNamespace(now=lambda _: instant))
    monkeypatch.chdir(tmp_path)
    config = {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"ppl": {}}}
    first, second = runs.run_eval(config), runs.run_eval(config)
    assert not first["is_error"] and not second["is_error"]
    assert first["plan"]["output_dir"] != second["plan"]["output_dir"]
    assert not Path(first["plan"]["output_dir"]).exists()
    assert not Path(second["plan"]["output_dir"]).exists()
    assert "output_dir" not in config
    assert runs.run_eval(first["plan"])["plan"]["output_dir"] == first["plan"]["output_dir"]


def test_separate_default_runs_preserve_record_environment_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from pathlib import Path

    from tricast.eval import runner

    monkeypatch.chdir(tmp_path)

    def evaluate(config: dict) -> list[dict]:
        record = _record()
        record["recipe"]["name"] = config["recipes"][0]["name"]
        record["env"]["model_id"] = config["model"]
        record["run_hash"] = config["model"]
        directory = Path(config["output_dir"])
        directory.mkdir(parents=True)
        (directory / f"{record['recipe']['name']}.json").write_text(json.dumps(record))
        (directory / "env.json").write_text(json.dumps(record["env"]))
        (directory / "summary.md").write_text(config["model"])
        return [record]

    monkeypatch.setattr(runner, "run_config", evaluate)
    config = {"model": "Org-A/model-a", "recipes": ["fp64_reference"], "tasks": {"ppl": {}}}
    first = runs.run_eval(config, execute=True)
    first_directory = Path(first["plan"]["output_dir"])
    preserved = {path.name: path.read_text() for path in first_directory.iterdir()}
    second = runs.run_eval({**config, "model": "Org-B/model-b"}, execute=True)
    assert not first["is_error"] and not second["is_error"]
    assert first["paths"] != second["paths"]
    assert {path.name: path.read_text() for path in first_directory.iterdir()} == preserved
    compared = runs.compare_runs(first["paths"] + second["paths"])
    assert not compared["is_error"]
    assert [record["env"]["model_id"] for record in compared["runs"]] == ["Org-A/model-a", "Org-B/model-b"]


def test_explicit_run_directory_remains_unchanged(tmp_path) -> None:
    config = {"model": "org/tiny", "recipes": ["fp64_reference"], "tasks": {"ppl": {}},
              "output_dir": str(tmp_path / "chosen")}
    result = runs.run_eval(config)
    assert not result["is_error"]
    assert result["plan"]["output_dir"] == config["output_dir"]
