"""Offline evaluation integration using directly constructed tiny language models."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tricast.eval.envinfo import capture_env
from tricast.eval.ppl import perplexity


@pytest.fixture
def tiny_eval_model():
    transformers = pytest.importorskip("transformers")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = transformers.LlamaConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, vocab_size=256, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    yield transformers.LlamaForCausalLM(config).eval()
    torch.set_num_threads(previous_threads)


@pytest.fixture
def tiny_eval_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i + 4 for i in range(252)})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]",
        unk_token="[UNK]", model_max_length=64,
    )


def test_perplexity_token_nll_and_tail(tiny_eval_model, tiny_eval_tokenizer):
    texts = ["t0 t1 t2 t3 t4", "t5 t6 t7 t8 t9"]
    ids = tiny_eval_tokenizer("\n\n".join(texts), return_tensors="pt")["input_ids"][0, :8].reshape(2, 4)
    with torch.no_grad():
        logits = tiny_eval_model(input_ids=ids, use_cache=False).logits[:, :-1]
        expected = F.cross_entropy(logits.double().reshape(-1, 256), ids[:, 1:].reshape(-1), reduction="sum")
    tiny_eval_model.train()
    result = perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=texts, seqlen=4, batch_size=2)
    assert result["nll"] == expected.item() / 6
    assert result["ppl"] == math.exp(result["nll"])
    assert result["n_tokens"] == 6
    assert result["n_windows"] == 2
    assert len(result["dataset_fingerprint"]) == 64
    assert tiny_eval_model.training
    limited = perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=texts, seqlen=4, max_windows=1)
    assert limited["n_windows"] == 1
    assert limited["n_tokens"] == 3
    assert limited["dataset_fingerprint"] == result["dataset_fingerprint"]


def test_perplexity_streaming_matches_full_window(tiny_eval_model, tiny_eval_tokenizer):
    texts = ["t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11"]
    full = perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=texts, seqlen=6)
    streamed = perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=texts, seqlen=6, streaming=True)
    assert (full["forward_mode"], streamed["forward_mode"]) == ("full_window", "streaming_cache")
    assert streamed["n_tokens"] == full["n_tokens"] == 10
    assert streamed["nll"] == pytest.approx(full["nll"], rel=1e-5)


@pytest.mark.parametrize("activation", ["fp8_tensor", "nvfp4", "mxfp4"])
def test_perplexity_does_not_depend_on_batch_size(tiny_eval_model, tiny_eval_tokenizer, activation):
    from tricast.nn.patch import iter_emulinear, patch_model, unpatch_model
    from tricast.recipe import load_recipe

    recipe = load_recipe({"name": "act", "backend": "reference", "defaults": {
        "activation": activation, "mma": {"preset": "fp64", "out_format": "fp32"}}})
    patch_model(tiny_eval_model, recipe)
    try:
        texts = [" ".join(f"t{(i * 7) % 250}" for i in range(40))]
        one = perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=texts, seqlen=8, batch_size=1)
        many = perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=texts, seqlen=8, batch_size=5)
        # Batching changes only the order of the native (unemulated) reductions; Linux CPU BLAS
        # rounds their last bit differently by batch shape (nvfp4: 256.5753119826756 vs ...758).
        # A scale that spanned the batch would change the quantized operands, far beyond 1e-12.
        floats = ("ppl", "nll")
        assert {k: v for k, v in many.items() if k not in floats} == {
            k: v for k, v in one.items() if k not in floats}
        assert all(math.isclose(many[k], one[k], rel_tol=1e-12, abs_tol=0.0) for k in floats)
        per_sequence = {layer.per_sequence for _, layer in iter_emulinear(tiny_eval_model)}
        assert per_sequence == {activation != "mxfp4"}  # MX blocks lie along K within one token
    finally:
        unpatch_model(tiny_eval_model)


def test_lmeval_adapter_evaluates_token_spanning_scales_one_request_at_a_time(tiny_eval_model,
                                                                             tiny_eval_tokenizer):
    lmeval = pytest.importorskip("tricast.eval.lmeval")
    from tricast.nn.patch import patch_model, unpatch_model
    from tricast.recipe import load_recipe

    for activation, expected in (("fp8_tensor", 1), ("mxfp4", 4)):
        recipe = load_recipe({"name": "act", "backend": "reference", "defaults": {
            "activation": activation, "mma": {"preset": "fp64", "out_format": "fp32"}}})
        patch_model(tiny_eval_model, recipe)
        try:
            adapter = lmeval.TriCastLM(pretrained=tiny_eval_model, tokenizer=tiny_eval_tokenizer,
                                       batch_size=4)
            assert adapter.batch_size == expected
            assert (adapter.batch_size_reason is None) == (expected == 4)
        finally:
            unpatch_model(tiny_eval_model)


def test_perplexity_short_and_invalid(tiny_eval_model, tiny_eval_tokenizer):
    with pytest.raises(ValueError, match="one complete"):
        perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=["t0"], seqlen=4)
    with pytest.raises(ValueError, match="seqlen"):
        perplexity(tiny_eval_model, tiny_eval_tokenizer, texts=["t0"], seqlen=1)


def test_capture_env(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub.HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="model-commit"))
    env = capture_env("offline/tiny", {"seed": 42})
    assert env["model_sha"] == "model-commit"
    assert env["model_id"] == "offline/tiny"
    assert env["seed"] == 42
    assert isinstance(env["git_dirty"], bool)
    assert len(env["git_sha"]) == 40
    assert {"utc", "hostname", "versions", "gpu_names", "gpu_driver", "cuda"} <= env.keys()
    assert {"python", "torch", "triton", "transformers", "lm_eval", "datasets"} <= env["versions"].keys()
    json.dumps(env, allow_nan=False)


def test_runner_resume_and_changed_config(tmp_path, monkeypatch, tiny_eval_model, tiny_eval_tokenizer):
    pytest.importorskip("tricast.mma.api")
    pytest.importorskip("tricast.reference.mma")
    pytest.importorskip("tricast.quant.api")
    pytest.importorskip("tricast.transforms")
    runner = pytest.importorskip("tricast.eval.runner")
    calls = []

    def load(*args):
        calls.append(args)
        return tiny_eval_model, tiny_eval_tokenizer

    monkeypatch.setattr(runner, "_load_model", load)
    env = capture_env(None, {"model_id": "offline/tiny", "model_sha": "model-commit",
                             "git_sha": "a" * 40, "git_dirty": False, "src_sha256": None,
                             "seed": 42, "dtype": "fp32", "device": "cpu"})
    monkeypatch.setattr(runner, "capture_env", lambda *args: env.copy())
    cfg = {
        "model": "offline/tiny", "dtype": "fp32", "device": "cpu", "seed": 42,
        "recipes": [{"name": "tiny", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
                     "include": ["model.layers.0.self_attn.q_proj"]}],
        "tasks": {"ppl": {"texts": ["t0 t1 t2 t3"], "seqlen": 4}}, "output_dir": str(tmp_path),
    }
    first = runner.run_config(cfg)
    assert math.isfinite(first[0]["metrics"]["ppl"]["ppl"])
    assert len(calls) == 1
    assert (tmp_path / "env.json").is_file()
    assert (tmp_path / "summary.md").is_file()
    assert json.loads((tmp_path / "tiny.json").read_text())["status"] == "complete"
    assert runner.run_config(cfg) == first
    assert len(calls) == 1
    invalid_ppls = [None, {}, {**first[0]["metrics"]["ppl"], "ppl": float("nan")},
                    {**first[0]["metrics"]["ppl"], "n_tokens": True},
                    {**first[0]["metrics"]["ppl"], "nll": "5"},
                    {**first[0]["metrics"]["ppl"], "dataset_fingerprint": None}]
    for invalid in invalid_ppls:
        corrupted = {**first[0], "metrics": {"ppl": invalid}}
        (tmp_path / "tiny.json").write_text(json.dumps(corrupted))
        repaired = runner.run_config(cfg)
        assert math.isfinite(repaired[0]["metrics"]["ppl"]["ppl"])
    assert len(calls) == 1 + len(invalid_ppls)
    (tmp_path / "tiny.json").write_text("[]")
    runner.run_config(cfg)
    assert len(calls) == 2 + len(invalid_ppls)
    cfg["seed"] = 43
    second = runner.run_config(cfg)
    assert len(calls) == 3 + len(invalid_ppls)
    assert first[0]["run_hash"] != second[0]["run_hash"]
    from tricast.nn.patch import iter_emulinear

    assert not list(iter_emulinear(tiny_eval_model))


def test_sweep_cartesian_product():
    runner = pytest.importorskip("tricast.eval.runner")
    recipes = runner.expand_sweep({
        "base_recipe": {"name": "sweep", "defaults": {"mma": "nvidia_hopper_fp8"}},
        "axes": {"mma.f_bits": [7, 13], "mma.c_mode": ["fused", "decoupled"]},
    })
    assert len(recipes) == 4
    assert len({r.name for r in recipes}) == 4
    assert {(r.defaults.mma.f_bits, r.defaults.mma.c_mode) for r in recipes} == {
        (7, "fused"), (7, "decoupled"), (13, "fused"), (13, "decoupled"),
    }
    with pytest.raises(ValueError, match="axis"):
        runner.expand_sweep({"base_recipe": recipes[0].to_dict(), "axes": {"mma.typo": [1]}})


def test_lmeval_registration_and_offline_task(tiny_eval_model, tiny_eval_tokenizer):
    pytest.importorskip("lm_eval")
    import pyarrow as pa
    from datasets import Dataset, DatasetDict
    from lm_eval.api.registry import get_model
    from lm_eval.api.task import ConfigurableTask

    from tricast.eval.lmeval import TriCastLM, evaluate

    class OfflineTask(ConfigurableTask):
        EVAL_HARNESS_NAME = "tricast_offline"

        def download(self, *args, **kwargs):
            table = pa.table({"text": ["t0 t1"], "target": [" t2"]})
            self.dataset = DatasetDict({"test": Dataset(table, fingerprint="tricast-offline-test")})

    task = OfflineTask(config={
        "task": "tricast_offline", "dataset_path": "offline", "test_split": "test",
        "output_type": "loglikelihood", "doc_to_text": "text", "doc_to_target": "target",
        "num_fewshot": 0,
    })
    assert get_model("tricast") is TriCastLM
    result = evaluate(tiny_eval_model, tiny_eval_tokenizer, [task], limit=1, batch_size=1)
    from tricast.eval.runner import _valid_metrics

    assert _valid_metrics({"lm_eval": result}, {"lm_eval": {}})
    assert math.isfinite(result["results"]["tricast_offline"]["perplexity,none"])
    assert result["n-samples"]["tricast_offline"]["effective"] == 1
    assert result["dataset_fingerprints"]["tricast_offline"]["test"] == "tricast-offline-test"


def test_tricast_adapter_patches_preloaded_model(tiny_eval_model, tiny_eval_tokenizer):
    pytest.importorskip("lm_eval")
    pytest.importorskip("tricast.reference.mma")
    pytest.importorskip("tricast.quant.api")
    pytest.importorskip("tricast.transforms")
    from lm_eval.api.instance import Instance

    from tricast.eval.lmeval import TriCastLM
    from tricast.nn.patch import iter_emulinear, unpatch_model

    recipe = {"name": "adapter", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
              "include": ["model.layers.0.self_attn.q_proj"]}
    adapter = TriCastLM(tiny_eval_model, tokenizer=tiny_eval_tokenizer, recipe=recipe,
                       backend="reference", calibrate=False, batch_size=1)
    assert [name for name, _ in iter_emulinear(adapter.model)] == ["model.layers.0.self_attn.q_proj"]
    requests = [Instance(request_type="loglikelihood", doc={}, arguments=("t0 t1", " t2"), idx=0)]
    try:
        scores = adapter.loglikelihood(requests)
        assert len(scores) == 1
        assert math.isfinite(scores[0][0])
    finally:
        unpatch_model(adapter.model)


@pytest.mark.parametrize("recipe_name", ["fp8_ema_static", "w4a16_g128_zp_gptq", "nvfp4_smoothquant"])
def test_calibration_paths(recipe_name, tiny_eval_model, tiny_eval_tokenizer, monkeypatch):
    pytest.importorskip("tricast.transforms")
    pytest.importorskip("tricast.reference.mma")
    pytest.importorskip("tricast.quant.api")
    import tricast.nn.linear as linear_module
    from tricast.calibration import calibrate
    from tricast.nn import patch_model
    from tricast.recipe import load_recipe

    recipe_data = load_recipe(recipe_name).to_dict()
    recipe_data["include"] = ["model.layers.0.self_attn.q_proj"]
    recipe = load_recipe(recipe_data)
    tiny_eval_model.train()
    tiny_eval_model.model.embed_tokens.eval()
    patch_model(tiny_eval_model, recipe, backend="reference")
    layer = tiny_eval_model.model.layers[0].self_attn.q_proj
    previous_operand = layer._weight_operand
    algorithms = []
    original_quantize_weight = linear_module.quantize_weight

    def tracked(*args, **kwargs):
        algorithms.append((args[2].kind, kwargs["hessian"].clone()))
        return original_quantize_weight(*args, **kwargs)

    monkeypatch.setattr(linear_module, "quantize_weight", tracked)
    result = calibrate(tiny_eval_model, recipe, tiny_eval_tokenizer,
                       texts=["t0 t1 t2 t3 t4 t5 t6 t7 t8 t9"], samples=2, seqlen=4, seed=42)
    assert result["samples"] == 2 and result["layers"] == ["model.layers.0.self_attn.q_proj"]
    assert layer.mode == "frozen" and layer._weight_operand is not previous_operand
    assert tiny_eval_model.training and not tiny_eval_model.model.embed_tokens.training
    assert layer._calibration_inputs is None and layer._stats is None
    if recipe_name == "fp8_ema_static":
        assert layer.observer.count == 2 and layer.observer.static_amax is not None
        assert layer.observer.mode == "frozen"
    elif recipe_name == "w4a16_g128_zp_gptq":
        assert len(algorithms) == 1 and algorithms[0][0] == "gptq"
        assert algorithms[0][1].shape == (64, 64)
        assert torch.isfinite(algorithms[0][1]).all()
    else:
        assert layer.transform.kind == "smoothquant"
        assert torch.isfinite(layer.transform.diag).all()
    tiny_eval_model.eval()
    with torch.no_grad():
        assert torch.isfinite(tiny_eval_model(torch.tensor([[4, 5, 6, 7]])).logits).all()


def test_calibration_replay_and_failure_restore(tiny_eval_model, monkeypatch):
    pytest.importorskip("tricast.transforms")
    pytest.importorskip("tricast.reference.mma")
    from tricast.calibration import calibrate
    from tricast.nn import patch_model
    from tricast.recipe import load_recipe

    recipe = load_recipe({"name": "replay", "defaults": {
        "activation": "fp8_tensor_ema", "weight": "fp8_row", "transform": "smoothquant",
        "mma": {"preset": "fp64", "out_format": "fp32"}},
        "include": ["model.layers.0.self_attn.q_proj"]})
    patch_model(tiny_eval_model, recipe, backend="reference")
    layer = tiny_eval_model.model.layers[0].self_attn.q_proj
    ids = torch.tensor([[4, 5, 6, 7]])
    captured = []
    hook = layer.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    calibrate(tiny_eval_model, recipe, input_ids=ids, samples=1, seqlen=4)
    hook.remove()
    expected_amax = layer.transform.apply_activation(captured[0]).abs().amax()
    assert layer.observer.static_amax == expected_amax
    assert layer._calibration_inputs is None
    observer, transform, operand = layer.observer, layer.transform, layer._weight_operand

    def fail(*args, **kwargs):
        raise ValueError("calibration failed deliberately")

    monkeypatch.setattr(tiny_eval_model, "forward", fail)
    with pytest.raises(ValueError, match="deliberately"):
        calibrate(tiny_eval_model, recipe, input_ids=ids, samples=1, seqlen=4)
    assert layer.mode == "frozen" and layer.observer is observer
    assert layer.transform is transform and layer._weight_operand is operand
    assert layer._calibration_inputs is None and layer._stats is None


def test_calibration_window_seed_and_validation():
    pytest.importorskip("tricast.transforms")
    from tricast.calibration import _windows

    ids = torch.arange(40).reshape(2, 20)
    first = _windows(None, None, ids, 4, 5, 42)
    second = _windows(None, None, ids, 4, 5, 42)
    third = _windows(None, None, ids, 4, 5, 43)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(first, third, strict=True))
    assert all(torch.equal(row.diff(), torch.ones(1, 4, dtype=torch.long)) for row in first)
    with pytest.raises(ValueError, match="shorter"):
        _windows(None, None, ids, 2, 21, 42)


@pytest.mark.parametrize("name", ["env", "ENV"])
def test_runner_rejects_reserved_recipe_name(name, tmp_path, monkeypatch):
    pytest.importorskip("tricast.nn.patch")
    runner = pytest.importorskip("tricast.eval.runner")

    def unexpected_load(*args):
        pytest.fail("reserved names must fail before loading a model")

    monkeypatch.setattr(runner, "_load_model", unexpected_load)
    env_path = tmp_path / "env.json"
    env_path.write_text('{"original": true}')
    config = {"model": "offline/tiny", "recipes": [{"name": name, "defaults": {}}], "tasks": {"ppl": {}},
              "output_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="reserved"):
        runner.run_config(config)
    assert json.loads(env_path.read_text()) == {"original": True}


def test_runner_validates_lmeval_cache_metrics():
    runner = pytest.importorskip("tricast.eval.runner")
    valid = {"results": {"offline": {"alias": "offline", "perplexity,none": 3.0}},
             "n-samples": {"offline": {"original": 1, "effective": 1}}}
    assert runner._valid_metrics({"lm_eval": valid}, {"lm_eval": {}})
    invalid = [None, {}, {**valid, "results": {}}, {**valid, "results": {"offline": None}},
               {**valid, "results": {"offline": {"perplexity,none": float("inf")}}},
               {**valid, "results": {"offline": {"perplexity,none": True}}},
               {**valid, "n-samples": None}]
    for entry in invalid:
        assert not runner._valid_metrics({"lm_eval": entry}, {"lm_eval": {}})


def test_lmeval_entrypoint_registers_in_fresh_process():
    import subprocess
    import sys

    pytest.importorskip("lm_eval")
    script = """
import runpy
import lm_eval.__main__
from lm_eval.api.registry import MODEL_REGISTRY
assert 'tricast' not in MODEL_REGISTRY

def cli():
    assert 'tricast' in MODEL_REGISTRY
    print('TRICAST_REGISTERED')

lm_eval.__main__.cli_evaluate = cli
runpy.run_module('tricast.eval.lmeval', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "TRICAST_REGISTERED" in result.stdout


def test_runner_json_metadata_types(tmp_path):
    from pathlib import Path

    import numpy as np

    from tricast.eval.runner import _write_json

    class Metadata:
        def __str__(self) -> str:
            return "metadata"

    path = tmp_path / "result.json"
    _write_json(path, {"dtype": torch.float32, "device": torch.device("cpu"), "scalar": np.int64(7),
                       "array": np.array([1, 2]), "tensor": torch.tensor([3, 4]), "path": Path("result"),
                       "set": {"b", "a"}, "object": Metadata()})
    assert json.loads(path.read_text()) == {
        "dtype": "torch.float32", "device": "cpu", "scalar": 7, "array": [1, 2], "tensor": [3, 4],
        "path": "result", "set": ["a", "b"], "object": "metadata",
    }


def test_runner_group_results_validate_only_leaf_sample_counts():
    from tricast.eval.runner import _valid_metrics

    scores = {"acc,none": 0.5, "acc_stderr,none": 0.1}
    result = {"results": {"mmlu": scores, "mmlu_math": scores, "mmlu_history": scores},
              "groups": {"mmlu": scores}, "group_subtasks": {"mmlu": ["mmlu_math", "mmlu_history"]},
              "n-samples": {"mmlu_math": {"original": 2, "effective": 2},
                            "mmlu_history": {"original": 3, "effective": 3}}}
    assert _valid_metrics({"lm_eval": result}, {"lm_eval": {}})
    del result["n-samples"]["mmlu_math"]
    assert not _valid_metrics({"lm_eval": result}, {"lm_eval": {}})
    result["n-samples"] = {}
    assert not _valid_metrics({"lm_eval": result}, {"lm_eval": {}})


def test_source_identity_hashes_untracked_source_and_ignores_bytecode(tmp_path, monkeypatch):
    import tricast.eval.envinfo as envinfo

    monkeypatch.setattr(envinfo, "_command",
                        lambda args, cwd=None: "a" * 40 if "rev-parse" in args else "?? src/")
    source = tmp_path / "src"
    source.mkdir()
    module = source / "module.py"
    module.write_text("value = 1\n")
    first = envinfo.source_identity(tmp_path)
    assert first["git_sha"] == "a" * 40 and first["git_dirty"]
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"bytecode")
    assert envinfo.source_identity(tmp_path) == first
    module.write_text("value = 2\n")
    assert envinfo.source_identity(tmp_path)["src_sha256"] != first["src_sha256"]
    module.unlink()
    assert envinfo.source_identity(tmp_path)["src_sha256"] != first["src_sha256"]


def test_local_model_revision_tracks_config_and_weight_metadata(tmp_path):
    import os

    from tricast.eval.envinfo import local_model_revision

    config = tmp_path / "config.json"
    config.write_text('{"hidden_size": 4}')
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")
    first = local_model_revision(tmp_path)
    timestamp = weight.stat().st_mtime_ns
    os.utime(weight, ns=(timestamp, timestamp + 1))
    second = local_model_revision(tmp_path)
    assert second != first
    weight.rename(tmp_path / "weights.safetensors")
    third = local_model_revision(tmp_path)
    assert third != second
    config.write_text('{"hidden_size": 8}')
    assert local_model_revision(tmp_path) != third


@pytest.fixture
def runner_case(tmp_path, monkeypatch, tiny_eval_model, tiny_eval_tokenizer):
    import tricast.eval.runner as runner

    env = capture_env(None, {"model_id": "offline/tiny", "model_sha": "model-commit",
                             "git_sha": "a" * 40, "git_dirty": False, "src_sha256": None,
                             "seed": 42, "dtype": "fp32", "device": "cpu"})
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return tiny_eval_model, tiny_eval_tokenizer

    monkeypatch.setattr(runner, "_load_model", load)
    monkeypatch.setattr(runner, "capture_env", lambda *args: env.copy())
    cfg = {"model": "offline/tiny", "dtype": "fp32", "device": "cpu", "seed": 42,
           "recipes": [{"name": "tiny", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
                        "include": ["model.layers.0.self_attn.q_proj"]}],
           "tasks": {"ppl": {"texts": ["t0 t1 t2 t3"], "seqlen": 4}}, "output_dir": str(tmp_path)}
    return runner, cfg, env, calls


def test_runner_rejects_empty_saved_environment(runner_case):
    from pathlib import Path

    runner, cfg, _, calls = runner_case
    first = runner.run_config(cfg)[0]
    record = {**first, "env": {}}
    Path(first["result_path"]).write_text(json.dumps(record))
    repaired = runner.run_config(cfg)[0]
    assert len(calls) == 2
    assert runner._valid_env(repaired["env"])
    assert repaired["env"]["recipe_hash"] == repaired["recipe_hash"]
    assert repaired["env"]["dataset_fingerprints"]["ppl"] == repaired["metrics"]["ppl"]["dataset_fingerprint"]
    assert repaired["patch_report"]["patched"][0][0] == "model.layers.0.self_attn.q_proj"


@pytest.mark.parametrize("change", ["git_sha", "git_dirty", "src_sha256", "model_sha"])
def test_runner_resume_invalidates_source_or_model_revision(runner_case, change):
    runner, cfg, env, calls = runner_case
    if change == "src_sha256":
        env.update(git_dirty=True, src_sha256="source-one")
    first = runner.run_config(cfg)[0]
    assert runner.run_config(cfg)[0] == first
    assert len(calls) == 1
    if change == "git_dirty":
        env.update(git_dirty=True, src_sha256="source-two")
    else:
        env[change] = "changed-revision"
    second = runner.run_config(cfg)[0]
    assert len(calls) == 2
    assert first["run_hash"] != second["run_hash"]


def test_runner_does_not_resume_unknown_model_revision(runner_case):
    runner, cfg, env, calls = runner_case
    env["model_sha"] = None
    first = runner.run_config(cfg)[0]
    second = runner.run_config(cfg)[0]
    assert first["status"] == second["status"] == "complete"
    assert len(calls) == 2
    assert second["env"]["model_sha"] is None


def test_runner_local_weights_change_invalidates_resume(runner_case, tmp_path, monkeypatch):
    import os

    from tricast.eval.envinfo import capture_env

    runner, cfg, _, calls = runner_case
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden_size": 4}')
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights")
    cfg["model"] = str(model_dir)
    monkeypatch.setattr(runner, "capture_env", capture_env)
    import tricast.eval.envinfo as envinfo

    monkeypatch.setattr(envinfo, "source_identity", lambda: {
        "git_sha": "a" * 40, "git_dirty": False, "src_sha256": None,
    })
    first = runner.run_config(cfg)[0]
    assert runner.run_config(cfg)[0] == first
    assert len(calls) == 1
    timestamp = weights.stat().st_mtime_ns
    os.utime(weights, ns=(timestamp, timestamp + 1))
    second = runner.run_config(cfg)[0]
    assert len(calls) == 2
    assert second["run_hash"] != first["run_hash"]


@pytest.mark.parametrize("change", ["versions", "gpu_names", "gpu_driver", "cuda"])
def test_runner_does_not_resume_across_software_or_gpu_changes(runner_case, change):
    runner, cfg, env, calls = runner_case
    runner.run_config(cfg)
    env[change] = {**env["versions"], "torch": "0.0.0-other"} if change == "versions" else ["other"]
    runner.run_config(cfg)
    assert len(calls) == 2


def test_installed_copy_reports_no_commit(monkeypatch):
    import tricast.eval.envinfo as envinfo

    monkeypatch.setattr(envinfo, "_command", lambda args, cwd=None: "/somewhere/else")
    identity = envinfo.source_identity()
    assert identity["git_sha"] is None and identity["git_dirty"] is None
    assert len(identity["src_sha256"]) == 64


def test_lmeval_adapter_records_recipe_patch_and_source(tiny_eval_model, tiny_eval_tokenizer):
    lmeval = pytest.importorskip("tricast.eval.lmeval")
    from tricast.nn.patch import unpatch_model
    from tricast.recipe import load_recipe

    recipe = load_recipe({"name": "act", "backend": "reference", "defaults": {
        "activation": "mxfp4", "mma": {"preset": "fp64", "out_format": "fp32"}}})
    adapter = lmeval.TriCastLM(pretrained=tiny_eval_model, tokenizer=tiny_eval_tokenizer,
                               recipe=recipe.to_dict())
    try:
        info = adapter.get_model_info()["tricast"]
        assert info["recipe_hash"] == recipe.sha256 and info["recipe"] == recipe.to_dict()
        assert info["patch_report"]["patched"] and info["calibration"] is None
        assert {"git_sha", "git_dirty", "src_sha256"} <= info["source"].keys()
    finally:
        unpatch_model(tiny_eval_model)


def test_runner_zero_patches_records_failure(runner_case):
    from pathlib import Path

    runner, cfg, _, _ = runner_case
    cfg["recipes"][0]["include"] = ["no_matching_module"]
    result = runner.run_config(cfg)[0]
    assert result["status"] == "failed"
    assert "zero patches" in result["error"]
    assert not result["patch_report"]["patched"]
    assert result["metrics"] == {}
    assert json.loads(Path(result["result_path"]).read_text())["status"] == "failed"


def test_runner_native_baseline_is_the_unpatched_model(runner_case, tiny_eval_model, tiny_eval_tokenizer):
    runner, cfg, _, calls = runner_case
    cfg["native_baseline"] = True
    native, tiny = runner.run_config(cfg)
    direct = perplexity(tiny_eval_model, tiny_eval_tokenizer, **cfg["tasks"]["ppl"])
    assert native["recipe"] == {"name": "native"} and native["recipe_hash"] == "native"
    assert native["status"] == "complete" and native["patch_report"] == {"native": True}
    assert native["metrics"]["ppl"] == direct
    assert tiny["recipe"]["name"] == "tiny" and tiny["patch_report"]["patched"]
    assert runner.run_config(cfg) == [native, tiny]
    assert len(calls) == 1
    cfg["recipes"][0]["name"] = "native"
    with pytest.raises(ValueError, match="reserved"):
        runner.run_config(cfg)


def test_runner_calibration_overrides_merge_and_record_actual_config(runner_case, monkeypatch):
    import tricast.calibration as calibration_module

    runner, cfg, _, _ = runner_case
    cfg["recipes"][0]["defaults"]["activation"] = "fp8_tensor_ema"
    cfg["recipes"][0]["calibration"] = {"dataset": "c4", "samples": 3, "seqlen": 4, "seed": 7}
    cfg["calibration"] = {"samples": 2, "seed": 8}
    observed = []

    def calibrate(model, recipe, **kwargs):
        observed.append((recipe.calibration, kwargs))
        return {"dataset_fingerprint": "calibration-tokens", "samples": 1, "seqlen": 4, "seed": 8}

    monkeypatch.setattr(calibration_module, "calibrate", calibrate)
    monkeypatch.setattr(runner, "perplexity", lambda *args, **kwargs: {
        "ppl": 2.0, "nll": math.log(2), "n_tokens": 3, "n_windows": 1, "dataset_fingerprint": "tokens",
    })
    result = runner.run_config(cfg, calibration={"samples": 1, "sequential": True})[0]
    assert observed[0][0] == {"dataset": "c4", "samples": 1, "seqlen": 4, "seed": 8, "sequential": True}
    assert "seed" not in observed[0][1]
    assert result["calibration_config"] == {
        "dataset": "c4", "samples": 1, "seqlen": 4, "seed": 8, "sequential": True, "split": "train",
    }
    assert result["calibration"]["dataset_fingerprint"] == "calibration-tokens"


@pytest.mark.parametrize("value, valid", [
    ("float", True), ("integer", True), ("nan", False), ("bool", False),
])
def test_runner_numpy_metric_values(value, valid):
    import numpy as np

    from tricast.eval.runner import _valid_metrics

    values = {"float": np.float32(0.5), "integer": np.int64(1), "nan": np.float32(float("nan")),
              "bool": np.bool_(True)}
    metrics = {"lm_eval": {"results": {"offline": {"acc,none": values[value]}},
                           "n-samples": {"offline": {"original": 1, "effective": 1}}}}
    assert _valid_metrics(metrics, {"lm_eval": {}}) is valid


def test_runner_kv_only_has_quantization_evidence(runner_case, monkeypatch):
    import tricast.kv.cache as cache_module

    runner, cfg, _, _ = runner_case
    cfg["recipes"][0]["include"] = ["no_matching_linear"]
    cfg["recipes"][0]["kv"] = {"key": {"scheme": "kivi2", "group_size": 2},
                               "value": {"scheme": "kivi2", "group_size": 2}, "residual": 2}
    calls = []
    original = cache_module.quantize_states

    def quantize(states, spec, axis):
        calls.append((tuple(states.shape), axis))
        return original(states, spec, axis)

    monkeypatch.setattr(cache_module, "quantize_states", quantize)
    result = runner.run_config(cfg)[0]
    assert result["status"] == "complete"
    assert not result["patch_report"]["patched"] and result["patch_report"]["kv"]
    assert result["metrics"]["ppl"]["forward_mode"] == "streaming_cache"
    assert calls and any(axis == "token" for _, axis in calls)


def test_runner_group_lmeval_json_and_fingerprints(runner_case, monkeypatch):
    from pathlib import Path

    import numpy as np

    import tricast.eval.lmeval as lmeval

    runner, cfg, _, _ = runner_case
    cfg["tasks"] = {"lm_eval": {"tasks": ["mmlu"]}}
    scores = {"acc,none": np.float32(0.5)}
    result = {"results": {"mmlu": scores, "mmlu_math": scores}, "groups": {"mmlu": scores},
              "group_subtasks": {"mmlu": ["mmlu_math"]},
              "n-samples": {"mmlu_math": {"original": 2, "effective": 2}},
              "config": {"dtype": torch.float32, "device": torch.device("cpu")},
              "dataset_fingerprints": {"mmlu_math": {"test": "offline-mmlu"}}}
    monkeypatch.setattr(lmeval, "evaluate", lambda *args, **kwargs: result)
    record = runner.run_config(cfg)[0]
    saved = json.loads(Path(record["result_path"]).read_text())
    assert saved["status"] == "complete"
    assert saved["metrics"]["lm_eval"]["config"]["dtype"] == "torch.float32"
    assert saved["env"]["dataset_fingerprints"]["lm_eval"] == {"mmlu_math": {"test": "offline-mmlu"}}
    assert runner.run_config(cfg)[0] == saved


def test_runner_pins_resolved_hf_revision(runner_case):
    runner, cfg, env, calls = runner_case
    env["model_kind"] = "hf"
    runner.run_config(cfg)
    assert calls[0][1] == {"revision": "model-commit"}


def test_lmeval_cache_model_call_matches_streaming_helper(tiny_eval_model, tiny_eval_tokenizer, monkeypatch):
    import tricast.kv.cache as cache_module
    from tricast.eval.lmeval import TriCastLM
    from tricast.eval.ppl import _model_logits
    from tricast.nn.patch import unpatch_model

    recipe = {"name": "kv_only", "defaults": {}, "include": ["no_matching_linear"],
              "kv": {"key": {"scheme": "kivi2", "group_size": 2},
                     "value": {"scheme": "kivi2", "group_size": 2}, "residual": 2}}
    calls = []
    original = cache_module.quantize_states

    def quantize(states, spec, axis):
        calls.append(axis)
        return original(states, spec, axis)

    monkeypatch.setattr(cache_module, "quantize_states", quantize)
    adapter = TriCastLM(tiny_eval_model, tokenizer=tiny_eval_tokenizer, recipe=recipe, batch_size=1)
    ids = torch.tensor([[4, 5, 6, 7]])
    try:
        actual = adapter._model_call(ids)
        assert calls and "token" in calls
        with torch.no_grad():
            expected = _model_logits(adapter.model, ids)
        assert torch.equal(actual, expected)
    finally:
        unpatch_model(adapter.model)


def test_local_model_revision_requires_config_and_weight_evidence(tmp_path):
    from tricast.eval.envinfo import local_model_revision

    assert local_model_revision(tmp_path) is None
    (tmp_path / "config.json").write_text("{}")
    assert local_model_revision(tmp_path) is None
    (tmp_path / "model.safetensors").write_bytes(b"weight-data")
    assert local_model_revision(tmp_path) is not None


def test_lmeval_adapter_rejects_zero_patches(tiny_eval_model, tiny_eval_tokenizer):
    from tricast.eval.lmeval import TriCastLM
    from tricast.nn.patch import iter_emulinear

    with pytest.raises(ValueError, match="no|zero"):
        TriCastLM(tiny_eval_model, tokenizer=tiny_eval_tokenizer,
                  recipe={"name": "empty", "defaults": {}, "include": ["no_matching_module"]})
    assert not list(iter_emulinear(tiny_eval_model))


@pytest.mark.parametrize("fail", [False, True])
def test_lmeval_evaluate_restores_training_modes(tiny_eval_model, tiny_eval_tokenizer, monkeypatch, fail):
    import tricast.eval.lmeval as lmeval

    tiny_eval_model.train()
    tiny_eval_model.model.embed_tokens.eval()
    modes = [(module, module.training) for module in tiny_eval_model.modules()]

    def simple_evaluate(**kwargs):
        assert not tiny_eval_model.training
        if fail:
            raise ValueError("deliberate evaluation failure")
        return {"results": {"offline": {"acc,none": 1.0}}}

    monkeypatch.setattr(lmeval.lm_eval, "simple_evaluate", simple_evaluate)
    if fail:
        with pytest.raises(ValueError, match="deliberate"):
            lmeval.evaluate(tiny_eval_model, tiny_eval_tokenizer, ["offline"])
    else:
        result = lmeval.evaluate(tiny_eval_model, tiny_eval_tokenizer, ["offline"])
        assert result["forward_modes"]["loglikelihood"] == "full_window"
    assert all(module.training == training for module, training in modes)


@pytest.mark.parametrize("override, expected", [(None, "reference"), ("auto", "auto")])
def test_lmeval_adapter_preserves_recipe_backend(
    tiny_eval_model, tiny_eval_tokenizer, override, expected,
):
    from tricast.eval.lmeval import TriCastLM
    from tricast.nn.patch import iter_emulinear, unpatch_model

    recipe = {"name": "backend", "backend": "reference", "defaults": {"mma": {"preset": "fp64"}},
              "include": ["model.layers.0.self_attn.q_proj"]}
    kwargs = {} if override is None else {"backend": override}
    adapter = TriCastLM(tiny_eval_model, tokenizer=tiny_eval_tokenizer, recipe=recipe, **kwargs)
    try:
        assert [layer.backend for _, layer in iter_emulinear(adapter.model)] == [expected]
    finally:
        unpatch_model(adapter.model)


@pytest.mark.parametrize("split", ["test", "validation", "train"])
@pytest.mark.parametrize("max_windows, expected_windows", [(None, 256), (3, 3), (512, 256)])
def test_c4_perplexity_gptq_bounded_documents_and_windows(monkeypatch, split, max_windows, expected_windows):
    import datasets

    consumed = []
    loaded = []
    tokenized = []
    forwarded = []
    documents = [f"doc{i}" for i in range(1100)]

    def load_dataset(*args, **kwargs):
        loaded.append((args, kwargs))

        def rows():
            for index, text in enumerate(documents):
                consumed.append(index)
                yield {"text": text}
            raise AssertionError("C4 evaluation consumed more than 1100 documents")

        return rows()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, use_cache):
            forwarded.append(input_ids.clone())
            logits = torch.zeros((*input_ids.shape, 4))
            return SimpleNamespace(logits=logits)

    def tokenize(text, return_tensors):
        tokenized.append(text)
        return {"input_ids": (torch.arange(1100) % 4).reshape(1, -1)}

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    result = perplexity(Model(), tokenize, dataset="c4", split=split, seqlen=4,
                        max_windows=max_windows, batch_size=64)
    resolved_split = "validation" if split == "test" else split
    shard_count = "01024" if split == "train" else "00008"
    assert loaded == [(("allenai/c4", "en"), {
        "data_files": {resolved_split: f"en/c4-{resolved_split}.00000-of-{shard_count}.json.gz"},
        "split": resolved_split, "streaming": True,
    })]
    assert consumed == list(range(1100))
    assert tokenized == [" ".join(documents)]
    assert result["n_windows"] == expected_windows
    assert result["n_tokens"] == expected_windows * 3
    # Summing token NLLs and applying exp can round the exact uniform-model PPL of 4.
    assert result["ppl"] == pytest.approx(4.0)
    assert torch.equal(torch.cat(forwarded).reshape(-1), torch.arange(expected_windows * 4) % 4)


def test_c4_explicit_texts_preserve_user_window_and_join_behavior(monkeypatch):
    import datasets

    captured = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, use_cache):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 4)))

    def tokenize(text, return_tensors):
        captured.append(text)
        return {"input_ids": torch.zeros((1, 1100), dtype=torch.long)}

    def unexpected_load(*args, **kwargs):
        raise AssertionError("explicit texts must not load C4")

    monkeypatch.setattr(datasets, "load_dataset", unexpected_load)
    result = perplexity(Model(), tokenize, dataset="c4", texts=iter(["first", "second"]),
                        seqlen=4, batch_size=64)
    assert captured == ["first\n\nsecond"]
    assert result["n_windows"] == 275
