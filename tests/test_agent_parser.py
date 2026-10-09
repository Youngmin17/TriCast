"""Offline extraction and SDK request/repair contracts without network access."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from tricast.agent.offline import parse_offline
from tricast.agent.parser import parse_request, render_prompt
from tricast.agent.request import REQUEST_SCHEMA, api_schema


def _at(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


@pytest.mark.parametrize(("text", "path", "expected"), [
    ("mxfp4 평가", "recipes.0.weight.scheme", "mxfp4"),
    ("Evaluate mxfp8_e4m3", "recipes.0.activation.scheme", "mxfp8_e4m3"),
    ("mxfp4 F=13", "recipes.0.mma.f_bits", 13),
    ("mxfp4 f_bits 17", "recipes.0.mma.f_bits", 17),
    ("mxfp4 CS=32", "recipes.0.mma.chunk_size", 32),
    ("mxfp4 chunk 16", "recipes.0.mma.chunk_size", 16),
    ("mxfp4 gdfs G=6", "recipes.0.mma.g_bits", 6),
    ("mxfp4 decoupled", "recipes.0.mma.c_mode", "decoupled"),
    ("fp8_tensor nvidia_hopper_fp8", "recipes.0.mma.preset", "nvidia_hopper_fp8"),
    ("mxfp4 Qwen/Qwen3-0.6B", "model", "Qwen/Qwen3-0.6B"),
    ("bf16 Example/Model-1.5B", "model", "Example/Model-1.5B"),
    ("mxfp4 hellaswag coqa", "tasks", ["hellaswag", "coqa"]),
    ("mxfp4 PPL", "tasks", ["wikitext2_ppl"]),
    ("가중치 mxfp4 활성화 bf16", "recipes.0.activation.scheme", "bf16"),
    ("weight int4_g128 activation bf16", "recipes.0.weight.scheme", "int4_g128"),
    ("mxfp4 random_hadamard gptq", "recipes.0.transform", "random_hadamard"),
    ("mxfp4 gptq", "recipes.0.weight_algo", "gptq"),
    ("hopper_fp8_w8a8 평가", "recipes.0.base", "hopper_fp8_w8a8"),
    ("mxfp4 max_windows 2 limit 0.5", "limits", {"max_windows": 2, "limit": 0.5}),
    ("fp8_e5m2 rtz", "recipes.0.weight.rounding", "rtz"),
    ("mxfp4 스윕 F 11에서 13까지", "sweep", {"axis": "f_bits", "values": [11, 12, 13]}),
    ("sweep mxfp4 f_bits from 11 to 13", "sweep", {"axis": "f_bits", "values": [11, 12, 13]}),
    ("sweep mxfp4 c_mode [fused, decoupled]", "sweep",
     {"axis": "c_mode", "values": ["fused", "decoupled"]}),
    ("mxfp4와 mxfp8_e4m3 비교", "intent", "compare"),
    ("explain Hopper F=13", "intent", "explain"),
    ("inspect fp8_e4m3", "intent", "inspect"),
])
def test_offline_explicit_values(text: str, path: str, expected: Any) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors
    assert result.request is not None
    assert _at(result.request.to_dict(), path) == expected


def test_offline_preserves_missing_fields_and_records_defaults() -> None:
    result = parse_request("mxfp4 평가", llm="offline")
    data = result.request.to_dict()
    assert data["recipes"][0]["mma"] is None
    assert data["recipes"][0]["weight"]["rounding"] is None
    assert data["recipes"][0]["weight"]["group_size"] is None
    assert data["limits"] == {"max_windows": None, "limit": None}
    assert any("Qwen/Qwen3-0.6B" in value for value in data["assumptions"])
    assert any("wikitext2_ppl" in value for value in data["assumptions"])
    assert result.request.to_dict() == parse_request("mxfp4 평가", llm="offline").request.to_dict()


@pytest.mark.parametrize("text", ["빠르게 평가해줘", "FP8 평가", "Hopper로 평가", "mxfp4 mxfp8_e4m3",
                                 "mxfp4 F=13 F=17", "sweep mxfp4", "mxfp4 G=6", "compare mxfp4"])
def test_offline_ambiguity_asks_instead_of_inventing(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors
    assert result.request is not None
    assert result.request.questions


def test_model_name_does_not_set_arithmetic() -> None:
    result = parse_request("mxfp4 Org/fp16", llm="offline")
    assert result.request.model == "Org/fp16"
    assert result.request.recipes[0]["weight"]["scheme"] == "mxfp4"
    assert not result.request.questions


def test_sweep_range_is_not_an_invented_fixed_value() -> None:
    result = parse_request("mxfp4 sweep F=11 to 13", llm="offline")
    assert result.request.recipes[0]["mma"] is None
    assert len(result.request.to_run_config()["recipes"]) == 3


@pytest.mark.parametrize("text", ["mxfp4 sweep F=1 to 2.5", "mxfp4 sweep F over 3,13,23.5",
                                 "mxfp4 sweep F=1 to 2e3", "mxfp4 sweep F over 3,13,unknown"])
def test_invalid_sweep_values_are_not_truncated_to_integers(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert result.errors or result.request.questions
    if result.request:
        assert result.request.sweep is None


def test_compare_keeps_alternatives_separate() -> None:
    result = parse_request("compare mxfp4 vs mxfp8_e4m3", llm="offline")
    assert [recipe["weight"]["scheme"] for recipe in result.request.recipes] == ["mxfp4", "mxfp8_e4m3"]
    assert not result.request.questions


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _response(data: Any) -> SimpleNamespace:
    content = [SimpleNamespace(type="text", text=json.dumps(data))]
    return SimpleNamespace(stop_reason="end_turn", content=content)


def test_anthropic_call_uses_exact_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRICAST_AGENT_MODEL", raising=False)
    client = FakeClient([_response(parse_offline("mxfp4"))])
    result = parse_request("mxfp4", llm="anthropic", client=client)
    assert not result.errors
    assert result.source == "anthropic"
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 16000
    assert kwargs["output_config"] == {
        "effort": "medium", "format": {"type": "json_schema", "schema": api_schema(REQUEST_SCHEMA)},
    }
    assert kwargs["extra_headers"] == {"anthropic-beta": "server-side-fallback-2026-07-01"}
    assert kwargs["extra_body"] == {"fallbacks": "default"}
    assert "thinking" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "mxfp4"}]
    assert kwargs["system"] == render_prompt()
    assert "{schemes}" not in kwargs["system"]
    assert "nvidia_hopper_fp8" in kwargs["system"]


def test_anthropic_model_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRICAST_AGENT_MODEL", "test-model")
    client = FakeClient([_response(parse_offline("mxfp4"))])
    parse_request("mxfp4", client=client)
    assert client.calls[0]["model"] == "test-model"


def test_refusal_does_not_read_content() -> None:
    class Refusal:
        stop_reason = "refusal"

        @property
        def content(self) -> Any:
            raise AssertionError("refusal content must not be read")

    client = FakeClient([Refusal()])
    result = parse_request("mxfp4", client=client)
    assert result.request is None
    assert "refused" in result.errors[0]
    assert len(client.calls) == 1


def test_repair_is_append_only_and_reports_paths() -> None:
    invalid = parse_offline("mxfp4")
    invalid["recipes"][0]["weight"]["rounding"] = "invented"
    client = FakeClient([_response(invalid), _response(parse_offline("mxfp4"))])
    result = parse_request("mxfp4", client=client)
    assert not result.errors
    first, second = [call["messages"] for call in client.calls]
    assert first == second[:1]
    assert second[1] == {"role": "assistant", "content": json.dumps(invalid)}
    assert second[2]["role"] == "user"
    assert "rounding" in second[2]["content"]
    assert len(second) == 3


def test_repair_includes_recipe_semantic_errors() -> None:
    invalid = parse_offline("mxfp4")
    invalid["recipes"][0]["weight"]["group_size"] = 0
    client = FakeClient([_response(invalid), _response(parse_offline("mxfp4"))])
    result = parse_request("mxfp4", client=client)
    assert not result.errors
    assert len(client.calls) == 2
    assert "group_size" in client.calls[1]["messages"][2]["content"]


def test_repair_is_bounded_to_two_retries() -> None:
    client = FakeClient([_response({}), _response({}), _response({})])
    result = parse_request("mxfp4", client=client)
    assert result.request is None and result.errors
    assert len(client.calls) == 3
    assert [len(call["messages"]) for call in client.calls] == [1, 3, 5]


def test_question_response_does_not_force_recipe_repair() -> None:
    client = FakeClient([_response(parse_offline("어떤 정밀도를 쓰지?"))])
    result = parse_request("어떤 정밀도를 쓰지?", client=client)
    assert result.request.questions
    assert not result.errors
    assert len(client.calls) == 1


@pytest.mark.parametrize("error_name", ["AuthenticationError", "APIConnectionError"])
def test_auto_fallback_only_for_sdk_auth_or_connection(
    monkeypatch: pytest.MonkeyPatch, error_name: str,
) -> None:
    class AuthenticationError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    sdk = SimpleNamespace(AuthenticationError=AuthenticationError, APIConnectionError=APIConnectionError)
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    error = getattr(sdk, error_name)("unavailable")
    result = parse_request("mxfp4", client=FakeClient([error]))
    assert result.request is not None
    assert not result.errors
    assert result.source == f"offline (anthropic fallback: {error_name})"
    strict = parse_request("mxfp4", llm="anthropic", client=FakeClient([error]))
    assert strict.request is None and strict.errors


def test_missing_sdk_falls_back_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = parse_request("mxfp4")
    assert result.request is not None
    assert result.source.startswith("offline (anthropic fallback:")


def test_unrelated_api_error_is_not_hidden_by_fallback() -> None:
    result = parse_request("mxfp4", client=FakeClient([RuntimeError("bad deployment")]))
    assert result.request is None
    assert result.source == "anthropic"
    assert result.errors == ["anthropic: RuntimeError"]


def test_preset_comparison_preserves_both_presets_and_asks_for_operands() -> None:
    result = parse_request("compare nvidia_ada_fp8 vs nvidia_hopper_fp8", llm="offline")
    assert result.request.questions
    assert [recipe["mma"]["preset"] for recipe in result.request.recipes] == [
        "nvidia_ada_fp8", "nvidia_hopper_fp8",
    ]


def test_preset_comparison_uses_explicit_operand_scheme() -> None:
    result = parse_request("compare nvidia_ada_fp8 vs nvidia_hopper_fp8 fp8_tensor", llm="offline")
    assert not result.errors and not result.request.questions
    assert [recipe.defaults.mma.f_bits for recipe in result.request.to_recipes()] == [13, 13]
    assert [recipe.defaults.mma.chunk_size for recipe in result.request.to_recipes()] == [16, 32]


def test_base_recipe_does_not_hide_extra_precision() -> None:
    result = parse_request("hopper_fp8_w8a8 bf16", llm="offline")
    assert not result.errors
    assert result.request.recipes[0]["weight"]["scheme"] == "bf16"
    recipe = result.request.to_recipes()[0]
    assert recipe.defaults.weight.format.name == "bf16"
    assert recipe.defaults.activation.format.name == "bf16"
    assert any("base의 weight" in value for value in result.request.assumptions)


def test_sweep_keeps_other_fixed_parameters() -> None:
    result = parse_request("mxfp4 CS=32 sweep F=11 to 13", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["mma"]["chunk_size"] == 32
    assert result.request.recipes[0]["mma"]["f_bits"] is None
    assert [recipe["defaults"]["mma"]["f_bits"] for recipe in result.request.to_run_config()["recipes"]] == [
        11, 12, 13,
    ]


def test_sentence_punctuation_is_not_part_of_model_id() -> None:
    result = parse_request("Evaluate perplexity for Qwen/Qwen3-0.6B.", llm="offline")
    assert result.request.model == "Qwen/Qwen3-0.6B"
    assert result.request.questions


@pytest.mark.parametrize("text", ["mxfp4 F=13.5 max_windows=1", "mxfp4 F=unknown", "mxfp4 CS=",
                                 "mxfp4 max_windows=1.5", "mxfp4 hellaswag limit unlimited"])
def test_malformed_explicit_numbers_cannot_silently_use_defaults(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert result.errors or result.request.questions


def test_negative_lm_eval_limit_is_rejected_not_made_unbounded() -> None:
    result = parse_request("mxfp4 hellaswag limit -1", llm="offline")
    assert result.errors
    assert any("limit" in error for error in result.errors)
    assert result.raw["limits"]["limit"] == -1


def test_comparison_keeps_shared_quantization_modifiers() -> None:
    result = parse_request("compare mxfp4 vs mxfp8_e4m3 rtz F=13", llm="offline")
    assert not result.errors and not result.request.questions
    assert [recipe.defaults.weight.rounding.value for recipe in result.request.to_recipes()] == ["rtz", "rtz"]
    assert [recipe.defaults.mma.f_bits for recipe in result.request.to_recipes()] == [13, 13]


@pytest.mark.parametrize("text", ["mxfp4 sweep G=5 to 6", "mxfp4 sweep group_size=16 to 17",
                                 "mxfp4 nvidia_hopper_fp8 G=6"])
def test_group_parameters_need_gdfs_not_silent_cofda_defaults(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors
    assert any("gdfs" in question for question in result.request.questions)


def test_gdfs_sweep_retains_explicit_algorithm() -> None:
    result = parse_request("mxfp4 gdfs sweep G=5 to 6", llm="offline")
    assert not result.errors and not result.request.questions
    recipes = result.request.to_run_config()["recipes"]
    assert [recipe["defaults"]["mma"]["g_bits"] for recipe in recipes] == [5, 6]
    assert [recipe["defaults"]["mma"]["algorithm"] for recipe in recipes] == ["gdfs", "gdfs"]


def test_provider_exception_does_not_copy_potential_credentials() -> None:
    secret = "sentinel-credential-do-not-report"
    result = parse_request("mxfp4", client=FakeClient([RuntimeError(f"bad headers Authorization: {secret}")]))
    assert result.request is None
    assert result.errors == ["anthropic: RuntimeError"]
    assert secret not in repr(result)


@pytest.mark.parametrize("text", ["가중치와 활성 모두 bf16으로 평가", "가중치와 활성화 모두 bf16 평가",
                                 "bf16 weights and bf16 activations", "weights and activations both bf16"])
def test_conjoined_or_postfix_operands_are_preserved(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["weight"]["scheme"] == "bf16"
    assert result.request.recipes[0]["activation"]["scheme"] == "bf16"


def test_korean_short_activation_role_does_not_merge_operands() -> None:
    result = parse_request("가중치 int4_g128, 활성 bf16으로 평가", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["weight"]["scheme"] == "int4_g128"
    assert result.request.recipes[0]["activation"]["scheme"] == "bf16"


@pytest.mark.parametrize("text", ["mxfp4 GDFS F=35으로 G=6으로 group_size=16으로 평가",
                                 "mxfp4 GDFS F=35, G=6, group_size=16으로 평가"])
def test_korean_numeric_particles_preserve_integer_values(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    mma = result.request.recipes[0]["mma"]
    assert (mma["f_bits"], mma["g_bits"], mma["group_size"]) == (35, 6, 16)


@pytest.mark.parametrize("text", ["mxfp4 CoFDA f_bits를 3, 13, 23으로 스윕하고 chunk_size=32로 평가",
                                 "Sweep mxfp4 CoFDA f_bits over 3, 13, 23 with chunk_size=32"])
def test_unbracketed_sweep_lists_preserve_order_and_other_parameters(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.sweep == {"axis": "f_bits", "values": [3, 13, 23]}
    assert result.request.recipes[0]["mma"]["chunk_size"] == 32


@pytest.mark.parametrize("text", ["가중치와 활성 모두 nvfp4로 평가. 모델은 나중에 정할게.",
                                 "Evaluate nvfp4 weights and nvfp4 activations. Ask me which model to use."])
def test_explicitly_deferred_model_is_not_replaced_with_default(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors
    assert result.request.model is None
    assert any("model" in question for question in result.request.questions)
    assert not any("Qwen/Qwen3-0.6B" in value for value in result.request.assumptions)


def test_korean_task_particles_do_not_hide_explicit_ppl() -> None:
    result = parse_request("mxfp4로 PPL과 hellaswag 평가", llm="offline")
    assert result.request.tasks == ["wikitext2_ppl", "hellaswag"]


def test_repair_replays_thinking_and_signature_without_modification() -> None:
    class Thinking:
        type = "thinking"

        def model_dump(self, *, exclude_none: bool) -> dict[str, str]:
            assert exclude_none
            return {"type": "thinking", "thinking": "Original reasoning", "signature": "original-signature"}

    invalid = SimpleNamespace(stop_reason="end_turn", content=[
        Thinking(), SimpleNamespace(type="text", text="not valid JSON"),
    ])
    client = FakeClient([invalid, _response(parse_offline("mxfp4"))])
    result = parse_request("mxfp4", client=client)
    assert not result.errors
    second = client.calls[1]["messages"]
    assert second[:1] == client.calls[0]["messages"]
    assert second[1] == {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Original reasoning", "signature": "original-signature"},
        {"type": "text", "text": "not valid JSON"},
    ]}
    assert second[2]["role"] == "user"


@pytest.mark.parametrize("format_name", [
    "int3", "uint3", "int3:full", "int5:frac=2", "e3m4:none:bias=3:nosub",
])
def test_custom_operand_formats_follow_get_format_grammar(format_name: str) -> None:
    result = parse_request(f"weight {format_name} activation bf16", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["weight"]["format"] == format_name
    assert result.request.to_recipes()[0].defaults.weight.format.name == format_name


@pytest.mark.parametrize("format_name", ["int99", "uint0", "e9m2", "nonsense"])
def test_invalid_explicit_operand_does_not_become_unquantized(format_name: str) -> None:
    result = parse_request(f"weight {format_name} activation bf16", llm="offline")
    assert not result.errors
    assert result.request.questions
    assert any("weight" in question or "format" in question for question in result.request.questions)
    with pytest.raises(ValueError, match="questions"):
        result.request.to_run_config()


def test_scale_rounding_never_changes_element_rounding() -> None:
    result = parse_request("nvfp4 scale.rounding=rtz", llm="offline")
    assert not result.errors and not result.request.questions
    choice = result.request.recipes[0]["weight"]
    assert choice["rounding"] is None and choice["scale_rounding"] == "rtz"
    quant = result.request.to_recipes()[0].defaults.weight
    assert quant.scale.rounding.value == "rtz"
    assert quant.rounding.value == "rne"


def test_observer_keyword_is_not_a_scale_search_request() -> None:
    result = parse_request("fp8_tensor observer=percentile", llm="offline")
    assert not result.errors and not result.request.questions
    choice = result.request.recipes[0]["activation"]
    assert choice["observer"] == "percentile"
    assert choice["scale_method"] is None
    assert result.request.to_recipes()[0].defaults.activation.scale.method == "absmax"


def test_scoped_quant_fields_preserve_independent_values() -> None:
    result = parse_request("nvfp4 scale.format=e3m4 scale.method=mse scale.rounding=rtz "
                           "rounding=rna group_size=32", llm="offline")
    assert not result.errors and not result.request.questions
    choice = result.request.recipes[0]["weight"]
    assert choice["format"] is None
    assert choice["scale_format"] == "e3m4"
    assert choice["scale_method"] == "mse"
    assert choice["scale_rounding"] == "rtz"
    assert choice["rounding"] == "rna"
    assert choice["group_size"] == 32


@pytest.mark.parametrize("text, expected", [
    ("Evaluate kivi2 mode=cache residual=32", {"preset": "kivi2", "mode": "cache", "residual": 32}),
    ("kivi4 KV 캐시 모드 fakequant 잔여 64 평가", {"preset": "kivi4", "mode": "fakequant", "residual": 64}),
    ("bf16 kv=kv_fp8 kv.mode=fakequant kv.residual=0",
     {"preset": "kv_fp8", "mode": "fakequant", "residual": 0}),
])
def test_kv_vocabulary_is_separate_from_operands(text: str, expected: dict[str, Any]) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["kv"] == expected
    recipe = result.request.to_recipes()[0]
    assert recipe.kv.mode == expected["mode"]
    assert recipe.kv.residual == expected["residual"]


def test_kivi_explicit_weight_scheme_is_not_reinterpreted_as_kv() -> None:
    result = parse_request("weight kivi2 activation bf16", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["weight"]["scheme"] == "kivi2"
    assert result.request.recipes[0]["kv"] is None


@pytest.mark.parametrize("text", [
    'bf16 layers "0-3" modules [q_proj, v_proj] skip calibration dataset=wikitext2 '
    'samples=4 seqlen=32 seed=42 sequential=true',
    'bf16 레이어 "0-3" 모듈 [q_proj, v_proj] 제외 교정 데이터셋 wikitext2 '
    '샘플 4 시퀀스_길이 32 시드 42 순차',
])
def test_layer_and_calibration_vocabulary_in_both_languages(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    choice = result.request.recipes[0]
    assert choice["layers"] == "0-3"
    assert choice["modules"] == ["q_proj", "v_proj"]
    assert choice["skip"] is True
    assert choice["calibration"] == {
        "dataset": "wikitext2", "samples": 4, "seqlen": 32, "seed": 42, "sequential": True,
    }


@pytest.mark.parametrize("text", [
    "report mxfp4 layer MSE SQNR cosine logits KL", "mxfp4 레이어별 MSE/SQNR/cos logits KL 리포트",
])
def test_report_intent_does_not_treat_metric_names_as_model_or_scale(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.intent == "report"
    assert result.request.model == "Qwen/Qwen3-0.6B"
    assert result.request.tasks == []
    assert result.request.recipes[0]["weight"]["scale_method"] is None


@pytest.mark.parametrize("text", ["bf16 kv=unknown", "bf16 kivi2 mode=unknown", "bf16 sequential=maybe"])
def test_invalid_new_explicit_options_require_clarification(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors
    assert result.request.questions


def test_calibration_false_and_dataset_id_do_not_change_model() -> None:
    result = parse_request("bf16 calibration.dataset=wikitext2 sequential false", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.model == "Qwen/Qwen3-0.6B"
    assert result.request.recipes[0]["calibration"]["dataset"] == "wikitext2"
    raw = parse_offline("bf16 calibration.dataset=Org/Data")
    assert raw["model"] == "Qwen/Qwen3-0.6B"
    assert raw["recipes"][0]["calibration"]["dataset"] == "Org/Data"
    assert result.request.recipes[0]["calibration"]["sequential"] is False


def test_sdk_schema_is_projected_but_local_constraints_remain() -> None:
    from tricast.agent.request import api_schema

    client = FakeClient([_response(parse_offline("mxfp4"))])
    result = parse_request("mxfp4", llm="anthropic", client=client)
    assert not result.errors
    sent = client.calls[0]["output_config"]["format"]["schema"]
    assert sent == api_schema()
    forbidden = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                 "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems"}

    def check(value: Any) -> None:
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            if "object" in value.get("type", []):
                assert value["additionalProperties"] is False
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(sent)
    assert "minimum" in json.dumps(REQUEST_SCHEMA)


@pytest.mark.parametrize("format_name", ["int99", "int3bad", "nonsense"])
def test_unknown_postfix_precision_blocks_without_swapping_roles(format_name: str) -> None:
    result = parse_request(f"{format_name} weights and bf16 activations", llm="offline")
    assert not result.errors
    assert result.request.questions
    assert result.request.recipes[0]["weight"] is None
    assert result.request.recipes[0]["activation"]["scheme"] == "bf16"
    with pytest.raises(ValueError, match="questions"):
        result.request.to_run_config()


def test_postfix_trailing_quant_field_requires_operand_scope() -> None:
    result = parse_request("nvfp4 weight and bf16 activation scale.rounding=rtz", llm="offline")
    assert not result.errors
    assert any("scale_rounding" in question and "rtz" in question for question in result.request.questions)
    with pytest.raises(ValueError, match="questions"):
        result.request.to_run_config()


def test_postfix_scoped_field_before_role_is_preserved() -> None:
    result = parse_request("nvfp4 scale.rounding=rtz weight and bf16 activation", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["weight"]["scale_rounding"] == "rtz"
    assert result.request.recipes[0]["activation"]["scale_rounding"] is None


def test_module_selector_list_accepts_spaces_after_commas() -> None:
    result = parse_request("bf16 modules q_proj, k_proj layers 0-3, 27", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["modules"] == ["q_proj", "k_proj"]
    assert result.request.recipes[0]["layers"] == "0-3, 27"


@pytest.mark.parametrize("alias", ["channel", "token"])
def test_scoped_granularity_aliases_use_canonical_row(alias: str) -> None:
    result = parse_request(f"fp8_tensor granularity={alias}", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["weight"]["granularity"] == "row"
    assert result.request.to_recipes()[0].defaults.weight.granularity == "row"


@pytest.mark.parametrize("text", [
    "evaluate mxfp4 on Llama-3.1-8B", "mxfp4 model=Qwen3-0.6B", "mxfp4 GPT2 모델 평가",
])
def test_orgless_model_is_not_replaced_with_an_unspecified_default(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors
    assert result.request.model is None
    assert any("model" in question for question in result.request.questions)
    assert not any("model 미지정" in assumption for assumption in result.request.assumptions)
    with pytest.raises(ValueError, match="questions"):
        result.request.to_run_config()


@pytest.mark.parametrize("options, seqlen, seed", [
    ("seqlen 512", 512, 42), ("seed 7", 2048, 7),
    ("seqlen=512 seed=7", 512, 7), ("evaluation.seqlen=512 evaluation.seed=7", 512, 7),
])
def test_evaluation_options_reach_the_plan_instead_of_calibration(
    options: str, seqlen: int, seed: int,
) -> None:
    result = parse_request(f"mxfp4 PPL {options}", llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.recipes[0]["calibration"] is None
    plan = result.request.to_run_config()
    assert plan["tasks"]["ppl"]["seqlen"] == seqlen
    assert plan["seed"] == seed


def test_explicit_calibration_and_evaluation_options_remain_independent() -> None:
    result = parse_request("mxfp4 PPL calibration.seqlen=128 calibration.seed=3 seqlen=512 seed=7",
                           llm="offline")
    assert not result.errors and not result.request.questions
    calibration = result.request.recipes[0]["calibration"]
    assert calibration["seqlen"] == 128 and calibration["seed"] == 3
    plan = result.request.to_run_config()
    assert plan["tasks"]["ppl"]["seqlen"] == 512 and plan["seed"] == 7


@pytest.mark.parametrize("text", [
    "Evaluate mxfp4 on hellaswag and report the accuracy",
    "mxfp4로 hellaswag 평가하고 결과를 설명해줘",
])
def test_named_evaluation_tasks_survive_report_or_explain_wording(text: str) -> None:
    result = parse_request(text, llm="offline")
    assert not result.errors and not result.request.questions
    assert result.request.intent == "evaluate"
    assert result.request.recipes
    assert result.request.tasks == ["hellaswag"]
    assert result.request.to_run_config()["tasks"]["lm_eval"]["tasks"] == ["hellaswag"]


def test_layer_report_and_named_evaluation_task_require_clarification() -> None:
    result = parse_request("mxfp4 hellaswag 평가하고 layer MSE report", llm="offline")
    assert not result.errors
    assert result.request.tasks == ["hellaswag"]
    assert any("intent" in question for question in result.request.questions)
    with pytest.raises(ValueError, match="questions"):
        result.request.to_run_config()


@pytest.mark.parametrize("options", ["seqlen unknown", "seed=7.5", "seqlen="])
def test_malformed_evaluation_options_cannot_disappear_into_defaults(options: str) -> None:
    result = parse_request(f"mxfp4 PPL {options}", llm="offline")
    assert result.errors or result.request.questions


def test_mixed_calibration_and_evaluation_scope_requires_clarification() -> None:
    result = parse_request("mxfp4 PPL calibration seqlen=512 seed=7", llm="offline")
    assert not result.errors
    assert any("calibration" in question for question in result.request.questions)
    with pytest.raises(ValueError, match="questions"):
        result.request.to_run_config()
