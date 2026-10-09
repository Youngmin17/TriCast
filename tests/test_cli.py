"""CLI inspection commands run without downloading models or loading CUDA."""

import json
from types import SimpleNamespace

import pytest
import torch

from tricast.cli import main


@pytest.mark.parametrize(("argv", "expected"), [
    (["formats"], "fp8_e4m3"),
    (["formats", "--name", "bf16"], "bf16"),
    (["schemes"], "nvfp4"),
    (["presets"], "nvidia_hopper_fp8"),
    (["cast", "--format", "int4", "--rounding", "rne", "1.5", "2.5"], "2.0"),
    (["cast", "--format", "fp16", "--no-saturate", "1e10"], "inf"),
])
def test_inspection_commands(argv, expected, capsys):
    assert main(argv) == 0
    assert expected in capsys.readouterr().out


def test_recipe_check(capsys):
    pytest.importorskip("tricast.recipe")
    assert main(["recipe-check", "fp64_reference"]) == 0
    assert "fp64_reference" in capsys.readouterr().out


def test_invalid_format(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["formats", "--name", "invalid"])
    assert exc.value.code == 2
    assert "unknown format" in capsys.readouterr().err


@pytest.fixture
def cli_model(tmp_path, monkeypatch):
    from tricast.eval import ppl, runner

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 8)
            self.config = SimpleNamespace(_commit_hash=None)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.lm_head(self.proj(self.embed(input_ids))))

    def tokenizer(text, **kwargs):
        return {"input_ids": torch.tensor([[int(word) % 8 for word in text.split()]])}

    torch.manual_seed(42)
    model = TinyLM().eval()
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({
        "name": "cli_tiny", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
        "include": ["proj"], "backend": "reference",
    }))
    monkeypatch.setattr(runner, "_load_model", lambda *args, **kwargs: (model, tokenizer))
    monkeypatch.setattr(ppl, "_dataset_texts", lambda *args: (["0 1 2 3 4 5 6 7"], "offline-data"))
    return model_path, recipe_path


def test_ppl_cli_saves_environment(cli_model, tmp_path, capsys):
    model_path, recipe_path = cli_model
    output = tmp_path / "ppl"
    assert main(["ppl", "--model", str(model_path), "--recipe", str(recipe_path),
                 "--seqlen", "4", "--max-windows", "1", "--out", str(output), "--device", "cpu"]) == 0
    record = json.loads((output / "cli_tiny.json").read_text())
    assert record["status"] == "complete"
    assert record["env"]["git_sha"]
    assert record["env"]["model_id"] == str(model_path)
    assert record["recipe_hash"]
    assert record["metrics"]["ppl"]["dataset_fingerprint"] == "offline-data"
    assert record["metrics"]["ppl"]["n_windows"] == 1
    assert (output / "env.json").is_file()
    printed = capsys.readouterr().out
    assert str(output / "cli_tiny.json") in printed
    assert "Dataset fingerprint: offline-data" in printed


def test_eval_cli_saves_environment(cli_model, tmp_path, monkeypatch, capsys):
    from tricast.eval import lmeval

    def evaluate(*args, **kwargs):
        assert kwargs["tasks"] == ["leaf"]
        return {"results": {"leaf": {"acc,none": 1.0}},
                "n-samples": {"leaf": {"original": 1, "effective": 1}},
                "dataset_fingerprints": {"leaf": {"test": "offline-leaf"}},
                "config": {"dtype": torch.float32}}

    monkeypatch.setattr(lmeval, "evaluate", evaluate)
    model_path, recipe_path = cli_model
    output = tmp_path / "eval"
    assert main(["eval", "--model", str(model_path), "--recipe", str(recipe_path),
                 "--tasks", "leaf", "--out", str(output)]) == 0
    record = json.loads((output / "cli_tiny.json").read_text())
    assert record["env"]["git_sha"]
    assert record["metrics"]["lm_eval"]["config"]["dtype"] == "torch.float32"
    assert record["metrics"]["lm_eval"]["dataset_fingerprints"]["leaf"]["test"] == "offline-leaf"
    printed = capsys.readouterr().out
    assert str(output / "cli_tiny.json") in printed
    assert "offline-leaf" in printed


@pytest.mark.parametrize("command", ["ppl", "eval", "run"])
def test_cli_calibration_overrides(command, monkeypatch):
    from tricast.eval import runner

    captured = []

    def run_config(config, **kwargs):
        captured.append((config, kwargs))
        return []

    monkeypatch.setattr(runner, "run_config", run_config)
    argv = [command, "config.yaml"] if command == "run" else [
        command, "--model", "offline", "--recipe", "fp64_reference"
    ]
    if command == "eval":
        argv += ["--tasks", "leaf"]
    argv += ["--calib-dataset", "c4", "--calib-samples", "2", "--calib-seqlen", "4",
             "--calib-seed", "7", "--sequential"]
    assert main(argv) == 0
    config, kwargs = captured[0]
    overrides = kwargs["calibration"] if command == "run" else config["calibration"]
    assert overrides == {"dataset": "c4", "samples": 2, "seqlen": 4, "seed": 7, "sequential": True}
    if command == "run":
        assert config == "config.yaml"


def test_cli_default_output_is_timestamped(monkeypatch, tmp_path):
    from pathlib import Path

    from tricast.eval import runner

    captured = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "run_config", lambda cfg: captured.append(cfg) or [])
    assert main(["ppl", "--model", "offline", "--recipe", "fp64_reference"]) == 0
    output = Path(captured[0]["output_dir"])
    assert output.parent == tmp_path / "runs"
    assert output.name.endswith("Z") and "T" in output.name
    assert captured[0]["calibration"] == {}


def test_cli_failed_run_returns_nonzero(cli_model, tmp_path, capsys):
    model_path, recipe_path = cli_model
    data = json.loads(recipe_path.read_text())
    data["include"] = ["missing"]
    recipe_path.write_text(json.dumps(data))
    output = tmp_path / "failed"
    assert main(["ppl", "--model", str(model_path), "--recipe", str(recipe_path),
                 "--seqlen", "4", "--out", str(output)]) == 1
    record = json.loads((output / "cli_tiny.json").read_text())
    assert record["status"] == "failed"
    assert record["error"]
    assert "failed" in capsys.readouterr().out


def test_eval_help_documents_registration_entrypoint(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    assert "python -m tricast.eval.lmeval" in capsys.readouterr().out


def test_report_cli_writes_metrics_and_markdown(cli_model, tmp_path, capsys):
    pytest.importorskip("tricast.analysis")
    model_path, recipe_path = cli_model
    text = tmp_path / "sample.txt"
    text.write_text("0 1 2 3 4 5 6 7")
    output = tmp_path / "report"
    assert main(["report", "--model", str(model_path), "--recipe", str(recipe_path),
                 "--text", str(text), "--samples", "1", "--seqlen", "4", "--out", str(output),
                 "--calib-dataset", "wikitext2", "--calib-samples", "2", "--calib-seqlen", "4",
                 "--calib-seed", "7", "--sequential", "--device", "cpu"]) == 0
    record = json.loads((output / "report.json").read_text())
    assert record["status"] == "complete"
    assert record["env"]["git_sha"]
    assert record["dataset_fingerprint"]
    assert record["source"]["fingerprint"]
    assert record["recipe"]["calibration"] == {
        "dataset": "wikitext2", "samples": 2, "seqlen": 4, "seed": 7, "sequential": True
    }
    assert record["model"]["logits_kl"] >= 0
    assert [layer["name"] for layer in record["layers"]] == ["proj"]
    assert (output / "report.md").read_text() == record["markdown"]
    assert (output / "env.json").is_file()
    assert str(output / "report.json") in capsys.readouterr().out


def test_report_rejects_multiple_sources(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["report", "--model", "offline", "--recipe", "fp64_reference", "--dataset", "wikitext2",
              "--text", "sample.txt"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_report_preserves_local_revision_and_start_environment(cli_model, tmp_path, monkeypatch):
    analysis = pytest.importorskip("tricast.analysis")
    from tricast.eval import envinfo, runner

    model_path, recipe_path = cli_model
    revision = envinfo.local_model_revision(model_path)
    calls = []
    captured = {}
    original_capture = envinfo.capture_env
    original_load = runner._load_model
    original_report = analysis.layer_report

    def capture(*args, **kwargs):
        calls.append("env")
        result = original_capture(*args, **kwargs)
        captured.update(result)
        return result

    def load(*args, **kwargs):
        calls.append("load")
        model, tokenizer = original_load(*args, **kwargs)
        model.config._commit_hash = "stale-upstream-commit"
        return model, tokenizer

    def report(*args, **kwargs):
        calls.append("report")
        return original_report(*args, **kwargs)

    monkeypatch.setattr(envinfo, "capture_env", capture)
    monkeypatch.setattr(runner, "_load_model", load)
    monkeypatch.setattr(analysis, "layer_report", report)
    output = tmp_path / "report"
    assert main(["report", "--model", str(model_path), "--recipe", str(recipe_path),
                 "--dataset", "wikitext2", "--samples", "1", "--seqlen", "4", "--out", str(output),
                 "--device", "cpu"]) == 0
    record = json.loads((output / "report.json").read_text())
    assert record["env"]["model_sha"] == revision
    assert record["env"]["src_sha256"] == captured["src_sha256"]
    assert record["source"]["fingerprint"] == "offline-data"
    assert calls == ["env", "load", "report"]


@pytest.mark.parametrize("command", ["ppl", "eval", "report"])
@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("device", [None, "cpu", "cuda:1"])
def test_cli_device_defaults_match_runner(command, cuda_available, device, monkeypatch):
    from tricast import cli
    from tricast.eval import runner

    captured = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(runner, "run_config", lambda config: captured.append(config["device"]) or [])
    monkeypatch.setattr(cli, "_report", lambda args: captured.append(args.device) or 0)
    argv = [command, "--model", "offline", "--recipe", "fp64_reference"]
    if command == "eval":
        argv += ["--tasks", "hellaswag"]
    if device is not None:
        argv += ["--device", device]
    assert main(argv) == 0
    assert captured == [device or ("cuda" if cuda_available else "cpu")]


@pytest.mark.parametrize("command", ["ppl", "eval", "report", "run"])
@pytest.mark.parametrize(("flag", "expected"), [
    (None, {}), ("--sequential", {"sequential": True}),
    ("--no-sequential", {"sequential": False}),
])
def test_cli_sequential_can_be_disabled_without_changing_defaults(command, flag, expected, monkeypatch):
    from tricast import cli
    from tricast.eval import runner

    captured = []

    def run_config(config, **kwargs):
        captured.append(kwargs["calibration"] if command == "run" else config["calibration"])
        return []

    monkeypatch.setattr(runner, "run_config", run_config)
    monkeypatch.setattr(cli, "_report", lambda args: captured.append(cli._calibration_overrides(args)) or 0)
    argv = [command, "config.yaml"] if command == "run" else [
        command, "--model", "offline", "--recipe", "fp64_reference"
    ]
    if command == "eval":
        argv += ["--tasks", "hellaswag"]
    if flag:
        argv.append(flag)
    assert main(argv) == 0
    assert captured == [expected]


def test_recipe_schema_accepts_and_normalizes_quant_granularity_synonyms():
    from importlib.resources import files

    from tricast.quant.spec import _GRANULARITY_SYNONYMS
    from tricast.recipe import load_recipe

    schema = json.loads(files("tricast").joinpath("schemas/recipe.schema.json").read_text())
    assert set(schema["$defs"]["granularity_kind"]["enum"]) == {"tensor", "row", "group", "block"}
    for synonym, canonical in _GRANULARITY_SYNONYMS.items():
        recipe = load_recipe({"name": "synonym", "defaults": {
            "weight": {"format": "fp8_e4m3", "granularity": synonym},
            "activation": {"format": "fp8_e4m3", "granularity": synonym},
        }})
        assert recipe.defaults.weight.granularity == canonical
        assert recipe.defaults.activation.granularity == canonical
        assert recipe.to_dict()["defaults"]["weight"]["granularity"] == canonical
        assert recipe.to_dict()["defaults"]["activation"]["granularity"] == canonical


@pytest.mark.parametrize("query", ["mxfp4 PPL max_windows 2", "mxfp4 hellaswag"])
def test_cli_agent_marks_unknown_cost_unavailable(query, capsys):
    assert main(["agent", query, "--llm", "offline"]) == 0
    output = capsys.readouterr().out
    assert "비용: 추정 불가" in output
    assert "비용 (추정):" not in output


def test_cli_agent_preserves_available_cost_estimate(monkeypatch, capsys):
    from tricast.agent import loop

    report = SimpleNamespace(
        request=None, source="offline", assumptions=[], questions=[], errors=[], results=[],
        summary="Dry run.",
        plan={"recipes": [{"name": "test"}], "tasks": {"ppl": {}}},
        cost={"estimated_seconds": 2.0, "emulated_macs": 2000},
    )
    monkeypatch.setattr(loop, "run_agent", lambda *args, **kwargs: report)
    assert main(["agent", "mxfp4 PPL", "--llm", "offline"]) == 0
    output = capsys.readouterr().out
    assert '비용 (추정): {"estimated_seconds": 2.0, "emulated_macs": 2000}' in output
    assert "추정 불가" not in output
