# AI 작성 초안 — 2026-10-01 팀 승인
"""Hand-derived arithmetic and acceptance-criterion golden cases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from tricast.formats import E8M0, FP4_E2M1, FP8_E4M3, FP32, UE4M3, get_format
from tricast.mma.operand import Operand
from tricast.mma.spec import PRESETS, MMASpec
from tricast.quant.spec import QuantSpec
from tricast.recipe import load_recipe
from tricast.reference.cast import round_to_format
from tricast.reference.mma import gemm_reference
from tricast.reference.quantize import quantize_reference

ROOT = Path(__file__).resolve().parents[1]
CASES = yaml.safe_load((ROOT / "tests/harness/golden_cases.yaml").read_text(encoding="utf-8"))


def _parameters() -> list[Any]:
    parameters = []
    for case in CASES:
        marks = [getattr(pytest.mark, name) for name in case.get("marks", [])]
        files = sorted(ROOT.glob(case["input"]["glob"])) if case["kind"].startswith("nadpe") else []
        if files:
            for path in files:
                variant = {**case, "input": {**case["input"], "file": str(path)}}
                parameters.append(pytest.param(variant, id=f"{case['id']}/{path.name}", marks=marks))
        else:
            parameters.append(pytest.param(case, id=case["id"], marks=marks))
    return parameters


def _equal(actual: torch.Tensor | None, expected: Any) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    target = torch.tensor(expected, dtype=torch.float32, device=actual.device)
    assert actual.shape == target.shape
    value = actual.float()
    assert bool(((value == target) | (torch.isnan(value) & torch.isnan(target))).all()), (value, target)


def _cast(case: dict[str, Any]) -> None:
    data = case["input"]
    actual = round_to_format(
        torch.tensor(data["values"], dtype=torch.float32), data["format"], data["rounding"],
        saturate=data["saturate"],
    )
    _equal(actual, case["expected"])


def _quantize(case: dict[str, Any]) -> None:
    data = case["input"]
    actual = quantize_reference(torch.tensor(data["values"], dtype=torch.float32),
                                QuantSpec.from_dict(data["spec"]))
    for key in ("values", "scale", "zero_point", "global_scale"):
        _equal(getattr(actual, key), case["expected"][key])
    _equal(actual.dequantize(), case["expected"]["dequantized"])


def _mma(case: dict[str, Any]) -> None:
    data = case["input"]
    a = Operand(torch.tensor(data["a"], dtype=torch.float32), get_format(data["format"]))
    b = Operand(torch.tensor(data["b"], dtype=torch.float32), get_format(data["format"]))
    _equal(gemm_reference(a, b, MMASpec.from_dict(data["spec"])), case["expected"])


def _recipe_error(case: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"^" + re.escape(case["expected"]["path"]) + r":"):
        load_recipe(case["input"])


def _preset(case: dict[str, Any]) -> None:
    assert case["expected"]["nonempty_provenance"] is True
    assert PRESETS
    for name, spec in PRESETS.items():
        assert isinstance(spec.provenance, str) and spec.provenance.strip(), name


def _tiny_model(seed: int, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=32, vocab_size=32, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.0,
    )
    model = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i + 4 for i in range(28)})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]",
        unk_token="[UNK]", model_max_length=32,
    )
    return model, tokenizer


def _runner_env(case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tricast.eval import runner

    data, expected = case["input"], case["expected"]
    model, tokenizer = _tiny_model(data["seed"], "cpu", torch.float32)
    monkeypatch.setattr(runner, "_load_model", lambda *args: (model, tokenizer))
    recipe = load_recipe({
        "name": "golden-env", "backend": "reference",
        "defaults": {"mma": {"preset": "fp32_fma", "out_format": "fp32"}},
        "include": ["model.layers.0.self_attn.q_proj"],
    })
    runner.run_config({
        "model": str(tmp_path), "dtype": "fp32", "device": "cpu", "seed": data["seed"],
        "recipes": [recipe.to_dict()], "output_dir": str(tmp_path),
        "tasks": {"ppl": {"texts": [data["text"]], "seqlen": data["seqlen"]}},
    })
    record = json.loads((tmp_path / "golden-env.json").read_text(encoding="utf-8"))
    env = json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))
    assert set(expected["record_keys"]) <= record.keys()
    assert set(expected["env_keys"]) <= env.keys()
    assert set(expected["version_keys"]) <= env["versions"].keys()
    assert record["env"] == env
    assert record["status"] == "complete"
    assert re.fullmatch(r"[0-9a-f]{40}", env["git_sha"])
    assert env["model_id"] == str(tmp_path) and env["model_sha"] is None
    assert env["seed"] == data["seed"]
    canonical = json.dumps(record["recipe"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert record["recipe_hash"] == hashlib.sha256(canonical.encode()).hexdigest()
    metrics = record["metrics"]["ppl"]
    assert set(expected["metric_keys"]) <= metrics.keys()
    assert metrics["dataset_fingerprint"] == hashlib.sha256(data["text"].encode()).hexdigest()
    assert metrics["n_tokens"] == 6  # Two 4-token windows each predict three targets.


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, f"{path}[{index}]")
    else:
        yield path, value


def _parser(case: dict[str, Any]) -> None:
    parser = pytest.importorskip("tricast.agent.parser")
    result = parser.parse_request(case["input"]["text"], llm="offline")
    assert not result.errors
    assert result.request is not None
    request = result.request.to_dict()
    assert request["model"] == case["expected"]["model"]
    assumptions = request.get("assumptions", [])
    assert isinstance(assumptions, list) and all(isinstance(item, str) for item in assumptions)
    fields = set(case["expected"]["unspecified_fields"])
    for path, value in _leaves({key: value for key, value in request.items() if key != "assumptions"}):
        if path.rsplit(".", 1)[-1] in fields and value is not None:
            pattern = rf"(?<![\w.\]]){re.escape(path)}(?![\w.\[])"
            assert any(re.search(pattern, item) for item in assumptions), (
                f"unrequested {path}={value!r} has no exact-path assumption"
            )


def _nadpe(case: dict[str, Any]) -> None:
    assert case["expected"]["all_vectors_match"] is True
    if "file" not in case["input"]:
        pytest.skip("independent NADPE vectors are absent: tests/data/nadpe/*.pt")
    path = Path(case["input"]["file"])
    vector = torch.load(path, map_location="cpu", weights_only=True)
    assert {"a_codes", "w_codes", "scale_a", "scale_b", "cases"} <= vector.keys()
    operands = []
    for code_key, scale_key in (("a_codes", "scale_a"), ("w_codes", "scale_b")):
        codes = vector[code_key]
        assert codes.dtype == torch.uint8 and codes.ndim == 2
        values = codes.contiguous().view(torch.float8_e4m3fn).float()
        scale = torch.as_tensor(vector[scale_key], dtype=torch.float32).reshape(())
        operands.append(Operand(values, FP8_E4M3, scale=scale, scale_fmt=FP32, scale_kind="tensor"))
    assert vector["cases"], f"{path}: no oracle cases"
    for index, data in enumerate(vector["cases"]):
        algorithm = int(data["algorithm"])
        assert algorithm in (1, 2, 3), (path, index, algorithm)
        spec = MMASpec(
            algorithm="gdfs" if algorithm == 1 else "cofda", f_bits=int(data["f_bits"]),
            g_bits=int(data["g_bits"]), group_size=int(data["group_size"]),
            chunk_size=int(data["chunk_size"]), k_tile=32, f2_bits=23,
            c_mode="decoupled" if algorithm == 3 else "fused", out_format="bf16",
        )
        expected = data["out_bits"]
        assert expected.dtype == torch.int16
        actual = gemm_reference(*operands, spec).contiguous().view(torch.int16)
        assert torch.equal(actual, expected), f"{path.name} case {index}: {spec}"


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _fp4_scales(codes: torch.Tensor, scale_format: str) -> tuple[torch.Tensor, Any]:
    if scale_format == "ue4m3":  # NADPE masks the sign bit of UE4M3 scale codes
        return (codes & 0x7F).contiguous().view(torch.float8_e4m3fn).float(), UE4M3
    assert scale_format == "e8m0", scale_format
    return torch.ldexp(torch.ones(codes.shape), codes.to(torch.int64) - 127).float(), E8M0


def _nadpe_fp4(case: dict[str, Any]) -> None:
    assert case["expected"]["all_vectors_match"] is True
    if "file" not in case["input"]:
        pytest.skip("independent NADPE FP4 vectors are absent: tests/data/nadpe/fp4_*.pt")
    path = Path(case["input"]["file"])
    vector = torch.load(path, map_location="cpu", weights_only=True)
    block = int(vector["block_size"])
    alpha = float(vector["alpha"])
    operands = []
    for side in ("a", "w"):
        values = torch.tensor(_E2M1)[vector[f"{side}_codes"].to(torch.int64)]
        scale, scale_fmt = _fp4_scales(vector[f"{side}_scale_codes"], vector["scale_format"])
        side_alpha = torch.tensor(alpha) if side == "a" and vector["scale_format"] == "ue4m3" else None
        operands.append(Operand(values, FP4_E2M1, scale=scale, scale_fmt=scale_fmt, scale_kind="k",
                                k_domain=block, alpha=side_alpha))
    assert vector["cases"], f"{path}: no oracle cases"
    for index, data in enumerate(vector["cases"]):
        algorithm = int(data["algorithm"])
        assert algorithm in (1, 2), (path, index, algorithm)
        if algorithm == 1:
            spec = MMASpec("gdfs", f_bits=int(data["f_bits"]), g_bits=int(data["g_bits"]),
                           group_size=int(data["group_size"]), k_tile=64, out_format="bf16")
        else:
            spec = MMASpec("cofda", f_bits=int(data["f_bits"]), chunk_size=int(data["chunk_size"]),
                           out_format="bf16")
        actual = gemm_reference(*operands, spec).contiguous().view(torch.int16)
        assert torch.equal(actual, data["out_bits"]), f"{path.name} case {index}: {spec}"


def _e2e(case: dict[str, Any], tmp_path: Path) -> None:
    from tricast.eval.envinfo import capture_env
    from tricast.eval.ppl import perplexity
    from tricast.nn.patch import iter_emulinear, patch_model, unpatch_model

    data = case["input"]
    model, tokenizer = _tiny_model(data["seed"], "cuda", torch.bfloat16)
    recipe = load_recipe(data["recipe"])
    kwargs = {"texts": [data["text"]], "seqlen": data["seqlen"], "device": "cuda"}
    baseline = perplexity(model, tokenizer, **kwargs)
    expected_layers = {name for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear) and recipe.spec_for(name) is not None}
    report = patch_model(model, recipe, backend="triton")
    layers = dict(iter_emulinear(model))
    assert report.patched and layers
    assert set(layers) == expected_layers
    fired = set()
    handles = []
    for name, layer in layers.items():
        assert layer.backend == "triton"
        handles.append(layer.register_forward_hook(lambda module, args, result, name=name: fired.add(name)))
    try:
        emulated = perplexity(model, tokenizer, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        unpatch_model(model)
    assert fired == set(layers), "every patched layer must execute"
    relative = abs(emulated["ppl"] - baseline["ppl"]) / baseline["ppl"]
    env = capture_env(str(tmp_path), {
        "seed": data["seed"], "recipe_hash": recipe.sha256,
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "model_config": model.config.to_dict(),
    })
    artifact = {"env": env, "baseline": baseline, "emulated": emulated, "relative_ppl_difference": relative}
    (tmp_path / "e2e.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    assert relative <= case["expected"]["max_relative_ppl_difference"], artifact


def test_golden_manifest() -> None:
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids))
    assert {case["ac"] for case in CASES} == {f"AC{i}" for i in range(1, 7)}
    required = {"id", "ac", "kind", "input", "expected", "note"}
    for case in CASES:
        assert required <= case.keys() <= required | {"marks"}, case["id"]
        assert re.search(rf"\b{case['ac']}\b", case["note"]), case["id"]
        assert set(case.get("marks", [])) <= {"gpu", "slow"}


def test_golden_vectors_present() -> None:
    """The NADPE vectors are tracked in git, so a missing or altered file fails here instead of
    turning the vector cases into a skip."""
    manifest = json.loads((ROOT / "tests/data/nadpe/manifest.json").read_text(encoding="utf-8"))
    for group in (manifest, manifest["fp4"]):  # FP8 at the top level, FP4 under "fp4"
        assert sum(item["cases"] for item in group["files"]) == group["total_cases"]
        for item in group["files"]:
            path = ROOT / "tests/data/nadpe" / item["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["file"]
    for case in CASES:
        if case["kind"].startswith("nadpe"):
            assert sorted(ROOT.glob(case["input"]["glob"])), case["id"]


@pytest.mark.parametrize("case", _parameters())
def test_golden(case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handlers = {"cast": _cast, "quantize": _quantize, "mma": _mma, "recipe_error": _recipe_error,
                "preset": _preset, "parser": _parser, "nadpe": _nadpe, "nadpe_fp4": _nadpe_fp4}
    if case["kind"] == "runner_env":
        _runner_env(case, tmp_path, monkeypatch)
    elif case["kind"] == "e2e":
        _e2e(case, tmp_path)
    else:
        handlers[case["kind"]](case)
