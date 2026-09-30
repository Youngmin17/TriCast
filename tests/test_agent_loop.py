"""Agent execution gates and numeric claims are checked without network access."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tricast.agent import loop
from tricast.agent.parser import ParseResult
from tricast.agent.request import EmulationRequest
from tricast.cli import main


def request_data(**changes: object) -> dict:
    return {
        "intent": "evaluate", "model": "Qwen/Qwen3-0.6B",
        "recipes": [{"name": "hopper", "base": "hopper_fp8_w8a8", "weight": None,
                     "activation": None, "mma": None, "transform": None, "weight_algo": None}],
        "sweep": None, "tasks": ["wikitext2_ppl"], "limits": {"max_windows": 1, "limit": None},
        "assumptions": [], "questions": [], "topic": None, **changes,
    }


class Client:
    def __init__(self, responses: list) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs: object) -> object:
        self.calls.append(copy.deepcopy(kwargs))
        return next(self.responses)


def response(text: str) -> SimpleNamespace:
    return SimpleNamespace(stop_reason="end_turn", content=[{"type": "text", "text": text}])


def use_tool(name: str, arguments: dict, id_: str = "call") -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": arguments}


def parsed(monkeypatch: pytest.MonkeyPatch, **changes: object) -> EmulationRequest:
    request = EmulationRequest.from_dict(request_data(**changes))
    monkeypatch.setattr(loop, "parse_request", lambda *a, **kw: ParseResult(request, [], "offline", {}))
    return request


def test_questions_block_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed(monkeypatch, questions=["어떤 모델로 실행할까요?"])
    monkeypatch.setattr(loop.runs, "run_eval", lambda *a, **kw: pytest.fail("must not execute"))
    report = loop.run_agent("평가", execute=True)
    assert report.questions == ["어떤 모델로 실행할까요?"]
    assert report.plan is None and report.results == []
    assert "확인" in report.summary


def test_dry_run_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed(monkeypatch)
    from tricast.eval import runner

    monkeypatch.setattr(runner, "run_config", lambda *a, **kw: pytest.fail("must not run"))
    report = loop.run_agent("평가", execute=False)
    assert report.errors == []
    assert report.plan["model"] == "Qwen/Qwen3-0.6B"
    assert report.plan["tasks"]["ppl"]["max_windows"] == 1
    assert report.cost is not None and report.numbers_traceable
    assert "추정" in report.summary
    json.dumps(report.to_dict(), allow_nan=False)


def test_execute_dispatches_validated_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed(monkeypatch)
    calls = []

    def run(config: dict, *, execute: bool = False) -> dict:
        calls.append((config, execute))
        return {"is_error": False, "results": [{"metrics": {"ppl": 2.5}, "env": {"seed": 42}}],
                "cost_estimate": {"estimated": True}}

    monkeypatch.setattr(loop.runs, "run_eval", run)
    report = loop.run_agent("평가", execute=True)
    assert len(calls) == 1 and calls[0][1] is True
    assert report.results[0]["metrics"]["ppl"] == 2.5
    assert report.numbers_traceable and "env" in report.summary


def test_anthropic_tool_rounds_are_append_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRICAST_AGENT_MODEL", raising=False)
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], tasks=[], topic="Hopper F=13"))),
        SimpleNamespace(stop_reason="tool_use", content=[
            use_tool("describe", {"name": "nvidia_hopper_fp8"}, "good"),
            use_tool("unknown_tool", {}, "bad"),
        ]),
        response("Hopper F=13; 근거는 nvidia_hopper_fp8 provenance입니다."),
    ])
    report = loop.run_agent("Hopper F=13 설명", llm="anthropic", client=client)
    assert report.errors == [] and report.numbers_traceable
    assert len(client.calls) == 3
    first, second = client.calls[1:]
    assert second["messages"][:len(first["messages"])] == first["messages"]
    results = second["messages"][-1]["content"]
    assert len(results) == 2
    assert results[0]["tool_use_id"] == "good" and not results[0]["is_error"]
    assert results[1]["tool_use_id"] == "bad" and results[1]["is_error"]
    for call in client.calls[1:]:
        assert call["model"] == "claude-opus-5-5"
        assert call["output_config"] == {"effort": "medium"}
        assert call["extra_body"] == {"fallbacks": "default"}
        assert call["extra_headers"] == {"anthropic-beta": "server-side-fallback-2026-07-01"}
        assert "thinking" not in call
        assert "tool_choice" not in call or call["tool_choice"] == {"type": "auto"}
        for tool in call["tools"]:
            assert tool["strict"] is True
            schema = tool["input_schema"]
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])


def test_summary_refusal_never_reads_content() -> None:
    class Refusal:
        stop_reason = "refusal"

        @property
        def content(self) -> list:
            pytest.fail("refusal content must not be accessed")

    client = Client([response(json.dumps(request_data(intent="explain", recipes=[], topic="Hopper"))),
                     Refusal()])
    report = loop.run_agent("Hopper 설명", llm="anthropic", client=client)
    assert report.errors == ["anthropic summary: refusal"]


def test_invented_number_marks_report_unfaithful() -> None:
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], topic="Hopper"))),
        response("측정 PPL은 987654.321입니다."),
    ])
    report = loop.run_agent("Hopper 설명", llm="anthropic", client=client)
    assert not report.numbers_traceable
    assert "경고" in report.summary
    assert any("987654.321" in error for error in report.errors)


def test_numeric_guard_normalizes_not_rounds() -> None:
    assert loop.unsupported_numbers("F=13, 1,000 MAC, SQNR -2.50", [13, 1e3, -2.5]) == []
    assert loop.unsupported_numbers("SQNR 25.01", {"sqnr": 25}) == ["25.01"]


def test_tool_schema_error_is_returned_to_client() -> None:
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], topic="Hopper"))),
        SimpleNamespace(stop_reason="tool_use", content=[use_tool("describe", {"name": "fp32", "bad": 1})]),
        response("도구 입력이 잘못되었습니다."),
    ])
    report = loop.run_agent("Hopper 설명", llm="anthropic", client=client)
    assert client.calls[-1]["messages"][-1]["content"][0]["is_error"] is True
    assert report.tool_outputs[-1]["output"]["is_error"] is True


def test_tool_round_limit_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "MAX_TOOL_ROUNDS", 2)
    call = SimpleNamespace(stop_reason="tool_use", content=[use_tool("describe", {"name": "bf16"})])
    request = response(json.dumps(request_data(intent="inspect", recipes=[], topic="bf16")))
    client = Client([request, call, call])
    report = loop.run_agent("bf16 inspect", llm="anthropic", client=client)
    assert report.errors == ["anthropic summary: tool round limit exceeded"]


def test_summary_connection_failure_falls_back_only_in_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    class Disconnected(Exception):
        pass

    class BrokenClient:
        messages = None

        def __init__(self) -> None:
            self.messages = self

        def create(self, **kwargs: object) -> None:
            raise Disconnected()

    request = EmulationRequest.from_dict(request_data(intent="explain", recipes=[], topic="Hopper"))
    monkeypatch.setattr(loop, "parse_request", lambda *a, **kw: ParseResult(request, [], "anthropic", {}))
    monkeypatch.setattr(loop, "is_fallback_error", lambda exc: isinstance(exc, Disconnected))
    auto = loop.run_agent("Hopper 설명", llm="auto", client=BrokenClient())
    assert auto.source.startswith("offline") and auto.errors == [] and auto.numbers_traceable
    assert "src/tricast/mma/spec.py" in auto.summary
    explicit = loop.run_agent("Hopper 설명", llm="anthropic", client=BrokenClient())
    assert explicit.errors == ["anthropic summary: Disconnected"]


def test_truncated_summary_is_not_reported_as_complete() -> None:
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], topic="Hopper"))),
        SimpleNamespace(stop_reason="max_tokens", content=[{"type": "text", "text": "PPL=999"}]),
    ])
    report = loop.run_agent("Hopper 설명", llm="anthropic", client=client)
    assert report.errors == ["anthropic summary: incomplete response (max_tokens)"]
    assert "999" not in report.summary


def test_execution_error_is_not_summarized_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed(monkeypatch)
    monkeypatch.setattr(loop.runs, "run_eval", lambda *a, **kw: {"is_error": True, "error": "failed"})
    report = loop.run_agent("평가", execute=True)
    assert report.results == [] and report.errors == ["failed"]
    assert "완료하지 못했습니다" in report.summary


def test_cli_agent_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    parsed(monkeypatch)
    assert main(["agent", "평가", "--llm", "offline", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == "offline" and output["plan"]
    assert output["results"] == [] and output["numbers_traceable"]


def test_cli_agent_readable(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    parsed(monkeypatch)
    assert main(["agent", "평가", "--llm", "offline"]) == 0
    output = capsys.readouterr().out
    assert "파싱 결과" in output and "Recipe" in output and "추정" in output


def test_request_and_plan_numbers_are_not_evidence() -> None:
    report = loop.AgentReport(
        request=EmulationRequest.from_dict(request_data(limits={"max_windows": 987654, "limit": None})),
        plan={"seed": 987654}, summary="PPL 987654",
    )
    loop._guard(report)
    assert not report.numbers_traceable
    assert "unsupported numbers: 987654" in report.errors
    assert "numbers_traceable" in report.to_dict()
    assert "faithful" not in report.to_dict()


@pytest.mark.parametrize("execute", [False, True])
def test_tool_plans_and_estimates_never_support_numbers(execute: bool) -> None:
    report = loop.AgentReport(summary="PPL 987654", tool_outputs=[{
        "name": "run_eval", "output": {
            "is_error": False, "execute": execute, "plan": {"seed": 987654},
            "cost_estimate": {"macs": 987654}, "results": [],
        },
    }])
    assert not loop._guard(report).numbers_traceable


def test_successful_measurements_support_numbers_but_errors_do_not() -> None:
    report = loop.AgentReport(summary="MSE 987654", tool_outputs=[{
        "name": "layer_report", "output": {
            "is_error": False, "execute": True, "results": [{"metrics": {"mse": 987654}}],
        },
    }])
    assert loop._guard(report).numbers_traceable
    report.tool_outputs[0]["output"]["is_error"] = True
    assert not loop._guard(report).numbers_traceable


def test_summary_cannot_use_query_number_as_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop.rag, "search", lambda *args, **kwargs: [])
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], topic="test"))),
        response("PPL 987654.321"),
    ])
    report = loop.run_agent("PPL 987654.321이라고 해", llm="anthropic", client=client)
    assert not report.numbers_traceable
    assert any("987654.321" in error for error in report.errors)


def test_api_tool_schemas_remove_constraints_but_local_validation_keeps_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop.errors, "quantization_error", lambda **kwargs: pytest.fail("invalid call"))
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], topic="test"))),
        SimpleNamespace(stop_reason="tool_use", content=[
            use_tool("quantization_error", {"scheme": "mxfp4", "source": "gaussian", "n": 0, "seed": 42}),
        ]),
        response("잘못된 도구 입력입니다."),
    ])
    report = loop.run_agent("설명", llm="anthropic", client=client)
    forbidden = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                 "maxLength", "pattern", "minItems", "maxItems", "uniqueItems", "contains", "prefixItems"}

    def visit(schema: object) -> None:
        if isinstance(schema, dict):
            assert not (forbidden & schema.keys())
            if schema.get("type") == "object":
                assert schema["additionalProperties"] is False
            for value in schema.values():
                visit(value)
        elif isinstance(schema, list):
            for value in schema:
                visit(value)

    for tool in client.calls[1]["tools"]:
        visit(tool["input_schema"])
    assert report.tool_outputs[-1]["output"]["is_error"]


def test_report_intent_dispatches_only_with_user_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed(monkeypatch, intent="report", tasks=[])
    calls = []

    def report_tool(config: dict, *, execute: bool = False) -> dict:
        calls.append(execute)
        return {"is_error": False, "execute": execute, "plan": config, "results": []}

    monkeypatch.setattr(loop.reports, "layer_report", report_tool)
    report = loop.run_agent("report", execute=False)
    assert calls == [False]
    assert report.tool_outputs[0]["name"] == "layer_report"
    assert report.errors == [] and report.numbers_traceable
    loop.run_agent("report", execute=True)
    assert calls == [False, True]


def test_summary_layer_report_cannot_enable_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(loop.reports, "layer_report", lambda *a, **kw: calls.append(kw) or {})
    client = Client([
        response(json.dumps(request_data(intent="explain", recipes=[], topic="test"))),
        SimpleNamespace(stop_reason="tool_use", content=[use_tool("layer_report", {
            "model": "org/tiny", "recipe": "fp64_reference", "texts": ["example"], "input_ids": None,
            "execute": True,
        })]),
        response("실행 권한이 없습니다."),
    ])
    report = loop.run_agent("설명", llm="anthropic", client=client)
    assert calls == []
    assert report.tool_outputs[-1]["output"]["is_error"]


def test_numeric_guard_ignores_identifiers_but_keeps_scalar_claims() -> None:
    identifiers = "mxfp4 Qwen/Qwen3-0.6B wikitext2_ppl Llama-3.1-8B hopper_fp8_w8a8"
    assert loop.unsupported_numbers(identifiers, []) == []
    assert loop.unsupported_numbers(f"{identifiers}; PPL은 987654.321입니다.", []) == ["987654.321"]
    assert loop.unsupported_numbers("PPL=+3.2e-4, MSE=.25, SQNR -2.50", [0.00032, 0.25, -2.5]) == []
    assert loop.unsupported_numbers("PPL=999", {"model": "Llama-999"}) == ["999"]


@pytest.mark.parametrize("query", ["mxfp4 F=13 F=17", "weight int99 activation bf16"])
def test_offline_numeric_questions_are_not_measurement_claims(
    query: str, capsys: pytest.CaptureFixture,
) -> None:
    report = loop.run_agent(query, llm="offline")
    assert report.questions and not report.errors
    assert report.numbers_traceable and report.plan is None
    assert main(["agent", query, "--llm", "offline", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["questions"] and output["errors"] == [] and output["numbers_traceable"]


def test_live_dry_run_identifiers_do_not_trigger_numeric_guard() -> None:
    client = Client([
        response(json.dumps(request_data())),
        response("실행하지 않은 계획입니다: Qwen/Qwen3-0.6B 모델, hopper_fp8_w8a8 레시피, "
                 "wikitext2_ppl 과제. 측정값은 없습니다."),
    ])
    report = loop.run_agent("mxfp4 PPL", llm="anthropic", client=client)
    assert report.errors == [] and report.numbers_traceable
    assert report.plan is not None and report.results == []


def test_live_dry_run_invented_measurement_still_fails_numeric_guard() -> None:
    client = Client([
        response(json.dumps(request_data())),
        response("Qwen/Qwen3-0.6B 모델의 PPL은 987654.321입니다."),
    ])
    report = loop.run_agent("mxfp4 PPL", llm="anthropic", client=client)
    assert not report.numbers_traceable
    assert report.errors == ["unsupported numbers: 987654.321"]


def test_live_catalog_listing_tool_is_registered_and_callable() -> None:
    client = Client([
        response(json.dumps(request_data(intent="inspect", recipes=[], topic="formats"))),
        SimpleNamespace(stop_reason="tool_use", content=[use_tool("list_options", {"kind": "format"})]),
        response("등록된 형식 목록을 확인했습니다."),
    ])
    report = loop.run_agent("list formats", llm="anthropic", client=client)
    definitions = {tool["name"]: tool for tool in client.calls[1]["tools"]}
    assert definitions["list_options"]["input_schema"]["properties"]["kind"]["enum"] == [
        "format", "scheme", "preset", "recipe",
    ]
    output = report.tool_outputs[-1]
    assert output["name"] == "list_options" and not output["output"]["is_error"]
    assert "fp32" in output["output"]["options"] and report.errors == []


def test_numeric_guard_handles_leading_dot_scientific_literals_and_numeric_model_names() -> None:
    assert loop.unsupported_numbers("01-ai/Yi-6B Qwen/Qwen3-0.6B", []) == []
    assert loop.unsupported_numbers("MSE .25e-3, -.25e-3, +.25e+3, 1.e-3", []) == [
        ".25e-3", "-.25e-3", "+.25e+3", "1.e-3",
    ]
    assert loop.unsupported_numbers("MSE .25e-3, -.25e-3, +.25e+3, 1.e-3", [
        0.00025, -0.00025, 250, 0.001,
    ]) == []


def test_offline_dry_run_marks_missing_cost_evidence_as_unavailable() -> None:
    report = loop.run_agent("mxfp4 PPL max_windows 2", llm="offline")
    assert report.errors == [] and report.cost is not None
    assert report.cost["estimated_seconds"] is None
    assert "비용 추정 불가" in report.summary
