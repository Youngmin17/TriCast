"""Hand-computed error metrics and offline report integration."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tricast.analysis import error_metrics, layer_report
from tricast.nn import EmuLinear


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(4, 4)
        self.proj = nn.Linear(4, 4, bias=False)
        self.unused = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.tensor([
                [0.25, 0.5, 0.75, 1.0], [1.25, -0.75, 0.25, 0.5],
                [-0.5, 0.25, 1.5, -0.75], [0.75, 1.25, -0.25, 0.5],
            ]))
            self.proj.weight.copy_(torch.tensor([
                [1.0, 0.25, -0.5, 0.75], [0.25, 1.0, 0.75, -0.5],
                [0.5, -0.75, 1.0, 0.25], [-0.25, 0.5, 0.75, 1.0],
            ]))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


class TinyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([[int(token) for token in text.split()]])}


@pytest.fixture
def model() -> TinyLM:
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return TinyLM()


def recipe(**changes: dict) -> dict:
    return {"name": "report", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}, **changes},
            "backend": "reference", "include": ["proj"]}


def test_error_metrics_hand_calculation() -> None:
    result = error_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
    assert result == {"mse": 0.5, "sqnr_db": 10 * math.log10(5), "max_abs_error": 1.0,
                      "relative_frobenius": math.sqrt(1 / 5), "cosine": 3 / math.sqrt(10)}
    assert error_metrics(torch.tensor([1.0]), torch.tensor([-1.0]))["cosine"] == -1.0


def test_error_metrics_zero_and_nonfinite() -> None:
    zero = torch.zeros(2)
    assert error_metrics(zero, zero) == {"mse": 0.0, "sqnr_db": "inf", "max_abs_error": 0.0,
                                       "relative_frobenius": 0.0, "cosine": 1.0}
    result = error_metrics(zero, torch.ones(2))
    assert result["sqnr_db"] == "-inf" and result["relative_frobenius"] == "inf"
    assert result["cosine"] == 0.0
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="non-finite"):
        error_metrics(torch.tensor([float("nan")]), torch.ones(1))
    with pytest.raises(ValueError, match="matching"):
        error_metrics(torch.zeros(2), torch.ones(1))


def test_report_identity_and_state_restoration(model: TinyLM) -> None:
    model.train()
    model.embed.eval()
    original = model.proj
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    result = layer_report(model, recipe(), input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    assert result["model"]["logits_kl"] == 0.0
    assert result["model"]["top1_agreement"] == 1.0
    assert result["model"]["ppl_reference"] == result["model"]["ppl_emulated"]
    assert result["model"]["n_tokens"] == 3
    for part in ("weight", "activation", "output"):
        assert result["layers"][0][part]["mse"] == 0.0
    assert model.proj is original and not isinstance(model.proj, EmuLinear)
    assert model.training and not model.embed.training
    assert not model.proj._forward_pre_hooks and not model.proj._forward_hooks
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    assert result["patch_report"]["patched"][0][0] == "proj"
    assert len(result["dataset_fingerprint"]) == 64
    json.dumps(result, allow_nan=False)


def test_report_quantization_changes_logits(model: TinyLM) -> None:
    result = layer_report(model, recipe(weight={"format": "int2", "scale": None}),
                          input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    assert result["model"]["logits_kl"] > 0.0
    assert result["layers"][0]["weight"]["mse"] > 0.0
    assert result["layers"][0]["activation"]["mse"] == 0.0
    assert result["layers"][0]["output"]["mse"] > 0.0


def test_report_transform_uses_same_coordinate_system(model: TinyLM) -> None:
    result = layer_report(model, recipe(transform={"kind": "hadamard", "block": 4}),
                          input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    assert result["model"]["logits_kl"] == 0.0
    assert all(result["layers"][0][part]["mse"] == 0.0 for part in ("weight", "activation", "output"))


def test_report_activation_includes_dequant_format_rounding(model: TinyLM) -> None:
    from tricast.reference.cast import round_to_format

    with torch.no_grad():
        model.embed.weight.add_(0.001)
    ids = torch.tensor([[0, 1, 2, 3]])
    activations = model.embed(ids).reshape(-1, 4)
    expected = error_metrics(activations, round_to_format(activations, "bf16"))
    result = layer_report(model, recipe(activation={"format": "fp32", "scale": None,
                                                   "mma_input": "dequant", "dequant_format": "bf16"}),
                          input_ids=ids, samples=1, seqlen=4)
    assert result["layers"][0]["activation"] == expected
    assert expected["mse"] > 0.0


def test_report_stochastic_activation_uses_actual_operand(model: TinyLM, monkeypatch) -> None:
    import tricast.nn.linear as linear_module
    from tricast.analysis import _operand_value

    actual = []
    original_gemm = linear_module.gemm

    def gemm(activation, *args, **kwargs):
        actual.append(_operand_value(activation).clone())
        return original_gemm(activation, *args, **kwargs)

    monkeypatch.setattr(linear_module, "gemm", gemm)
    ids = torch.tensor([[0, 1, 2, 3]])
    activations = model.embed(ids).reshape(-1, 4)
    torch.manual_seed(42)
    result = layer_report(model, recipe(activation={"format": "int2", "scale": None, "rounding": "sr"}),
                          input_ids=ids, samples=1, seqlen=4)
    assert len(actual) == 1
    assert result["layers"][0]["activation"] == error_metrics(activations, actual[0])


def test_report_stochastic_rounding_is_reproducible_without_rng_side_effects(model: TinyLM) -> None:
    data = recipe(weight={"format": "int2", "scale": None, "rounding": "sr"},
                  activation={"format": "int2", "scale": None, "rounding": "sr"})
    state = torch.random.get_rng_state().clone()
    first = layer_report(model, data, input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(7)
    second = layer_report(model, data, input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    assert first == second and first["seed"] == 42


def test_report_calibration_uses_recipe_dataset(model: TinyLM, monkeypatch) -> None:
    import importlib

    calibration_module = importlib.import_module("tricast.calibration")
    called = []

    def dataset(dataset: str, split: str) -> list[str]:
        called.append((dataset, split))
        return ["0 1 2 3"]

    monkeypatch.setattr(calibration_module, "_dataset_texts", dataset)
    data = recipe(activation="fp8_tensor_ema")
    data["calibration"] = {"dataset": "wikitext2", "samples": 1, "seqlen": 4, "seed": 42}
    result = layer_report(model, data, tokenizer=TinyTokenizer(), input_ids=[[3, 2, 1, 0]],
                          samples=1, seqlen=4)
    assert called == [("wikitext2", "train")]
    assert result["calibration"]["samples"] == 1 and result["calibration"]["seed"] == 42
    assert result["calibration_config"] == {"dataset": "wikitext2", "split": "train", "samples": 1,
                                             "seqlen": 4, "seed": 42, "sequential": False}
    assert result["calibration"]["dataset_fingerprint"] != result["dataset_fingerprint"]


def test_report_failure_removes_patch_and_hooks(model: TinyLM, monkeypatch) -> None:
    import tricast.nn.linear as linear_module

    original = model.proj
    replaced = []

    def fail(layer, x):
        replaced.append(layer)
        raise ValueError("deliberate emulation failure")

    monkeypatch.setattr(linear_module.EmuLinear, "forward", fail)
    with pytest.raises(ValueError, match="deliberate"):
        layer_report(model, recipe(), input_ids=[[0, 1]], samples=1, seqlen=2)
    assert model.proj is original and model.training
    assert not replaced[0]._forward_pre_hooks and not replaced[0]._forward_hooks


def test_report_texts_windows_and_fingerprint(model: TinyLM) -> None:
    first = layer_report(model, recipe(), texts=["0 1 2 3", "3 2 1 0"], tokenizer=TinyTokenizer(),
                         samples=2, seqlen=4)
    second = layer_report(model, recipe(), input_ids=[[0, 1, 2, 3], [3, 2, 1, 0]], samples=2, seqlen=4)
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["samples"] == 2 and first["model"]["n_tokens"] == 6
    assert first["layers"] == second["layers"]


def test_report_unused_layer_is_not_reported_as_zero_error(model: TinyLM) -> None:
    data = recipe()
    data["include"] = ["proj", "unused"]
    result = layer_report(model, data, input_ids=[[0, 1]], samples=1, seqlen=2)
    assert result["layers"][1]["name"] == "unused"
    assert result["layers"][1]["status"] == "not_observed"
    assert result["layers"][1]["activation"] is None and result["layers"][1]["output"] is None
    assert "unused (not observed)" in result["markdown"]


@pytest.mark.parametrize("kwargs, message", [
    ({"texts": "0 1"}, "tokenizer"),
    ({"input_ids": [[0.0, 1.0]]}, "integer"),
    ({"input_ids": [[0]]}, "complete window"),
    ({"input_ids": [[0, 1]], "texts": "0 1"}, "exactly one"),
])
def test_report_invalid_inputs(model: TinyLM, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        layer_report(model, recipe(), samples=1, seqlen=2, **kwargs)


def test_report_zero_patch_is_failure(model: TinyLM) -> None:
    data = recipe()
    data["include"] = ["missing"]
    with pytest.raises(ValueError, match="no Linear"):
        layer_report(model, data, input_ids=[[0, 1]], samples=1, seqlen=2)


def test_report_local_errors_do_not_include_upstream_errors(model: TinyLM, monkeypatch) -> None:
    with torch.no_grad():
        model.unused.weight.copy_(torch.eye(4))

    def forward(input_ids, use_cache=False):
        return SimpleNamespace(logits=model.unused(model.proj(model.embed(input_ids))))

    monkeypatch.setattr(model, "forward", forward)
    data = recipe()
    data["include"] = ["proj", "unused"]
    data["overrides"] = [{"match": "proj", "weight": {"format": "int2", "scale": None}}]
    result = layer_report(model, data, input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    assert result["model"]["logits_kl"] > 0.0
    assert [layer["name"] for layer in result["layers"]] == ["proj", "unused"]
    assert result["layers"][0]["output"]["mse"] > 0.0
    assert result["layers"][1]["output"]["mse"] == 0.0
    assert result["markdown"].index("| proj |") < result["markdown"].index("| unused |")


def test_report_history_observer_does_not_advance_twice(model: TinyLM, monkeypatch) -> None:
    import tricast.nn.linear as linear_module
    from tricast.analysis import _operand_value

    operands = []
    counts = []
    original = linear_module.gemm

    def gemm(activation, *args, **kwargs):
        operands.append(_operand_value(activation).clone())
        counts.append(model.proj.observer.count)
        return original(activation, *args, **kwargs)

    monkeypatch.setattr(linear_module, "gemm", gemm)
    ids = torch.tensor([[0, 0], [2, 2]])
    activation = model.embed(ids).reshape(-1, 4)
    result = layer_report(model, recipe(activation={"format": "fp8_e4m3", "granularity": "tensor",
                                                   "observer": {"kind": "history"}}),
                          input_ids=ids, samples=2, seqlen=2)
    assert counts == [1, 2]
    assert result["layers"][0]["activation"] == error_metrics(activation, torch.cat(operands))


@pytest.mark.parametrize("mode", ["fakequant", "cache"])
def test_report_kv_only_changes_logits_and_restores_attention(mode: str) -> None:
    transformers = pytest.importorskip("transformers")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(42)
            # initializer_range 0.5: with the default 0.02 the 2-bit cache moved the logits by
            # ~1e-8 and the KL was rounding noise (2e-17 on macOS, 0.0 on Linux); here KIVI-2 moves
            # the perplexity in the fourth digit and the KL is 2.7e-8 on both.
            model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
                hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, vocab_size=8, max_position_embeddings=16, initializer_range=0.5,
                bos_token_id=1, eos_token_id=2, pad_token_id=0,
            )).eval()
        attention = model.model.layers[0].self_attn
        config = attention.config
        data = {"name": "kv-report", "defaults": {}, "include": ["no-linear"],
                "kv": {"preset": "kivi2", "mode": mode,
                "residual": 2, "key": {"scheme": "kivi2", "group_size": 2},
                "value": {"scheme": "kivi2", "group_size": 2}}}
        result = layer_report(model, data, input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
        assert result["layers"] == [] and result["patch_report"]["patched"] == []
        assert result["patch_report"]["kv"] == ["model.layers.0.self_attn"]
        assert result["model"]["logits_kl"] > 1e-9
        assert math.isfinite(result["model"]["ppl_emulated"])
        assert attention.config is config
        assert not hasattr(model, "_tricast_kv_patch") and not hasattr(model, "_tricast_kv_patched")
    finally:
        torch.set_num_threads(previous_threads)
