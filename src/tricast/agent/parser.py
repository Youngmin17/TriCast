"""Structured request parsing with bounded repairs and an offline fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from ..formats import REGISTRY
from ..mma.spec import PRESETS, Algorithm
from ..quant.spec import KV_PRESETS, SCHEMES, TransformKind, WeightAlgo
from .offline import TASKS, parse_offline, recipe_names
from .request import EmulationRequest, RequestValidationError, api_schema


@dataclass
class ParseResult:
    request: EmulationRequest | None
    errors: list[str] = field(default_factory=list)
    source: str = "offline"
    raw: Any = None


def create_client() -> Any:
    """Let the SDK resolve credentials without reading or copying their values."""
    import anthropic

    return anthropic.Anthropic()


def request_options() -> dict[str, Any]:
    """Common message options for parsing and the manual tool loop."""
    return {
        "model": os.environ.get("TRICAST_AGENT_MODEL", "claude-opus-5-5"),
        "max_tokens": 16000,
        "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
        "extra_body": {"fallbacks": "default"},
    }


def render_prompt() -> str:
    template = Path(__file__).resolve().parents[1] / "prompts" / "parse_query.md"
    return template.read_text(encoding="utf-8").format(
        formats=", ".join(sorted(REGISTRY)), schemes=", ".join(sorted(SCHEMES)),
        presets=", ".join(sorted(PRESETS)), recipes=", ".join(recipe_names()),
        tasks=", ".join(TASKS), algorithms=", ".join(get_args(Algorithm)),
        transforms=", ".join(get_args(TransformKind)), weight_algorithms=", ".join(get_args(WeightAlgo)),
        kv_presets=", ".join(KV_PRESETS),
    )


def _value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _assistant_content(response: Any, raw: str | None) -> Any:
    blocks = _value(response, "content", [])
    if len(blocks) == 1 and _value(blocks[0], "type") == "text":
        return raw
    return [
        block.model_dump(exclude_none=True) if hasattr(block, "model_dump")
        else dict(block) if isinstance(block, dict) else vars(block).copy()
        for block in blocks
    ]


def _validate(data: Any) -> EmulationRequest:
    request = EmulationRequest.from_dict(data)
    if not request.questions:
        request.to_recipes()
    return request


def _offline(text: str, source: str = "offline") -> ParseResult:
    data = parse_offline(text)
    try:
        return ParseResult(_validate(data), source=source, raw=data)
    except (RequestValidationError, ValueError, TypeError) as exc:
        return ParseResult(None, getattr(exc, "errors", [str(exc)]), source, data)


def is_fallback_error(exc: Exception) -> bool:
    if isinstance(exc, ImportError):
        return True
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(exc, (anthropic.AuthenticationError, anthropic.APIConnectionError))


def parse_request(
    text: str, *, llm: Literal["auto", "anthropic", "offline"] = "auto", client: Any = None,
) -> ParseResult:
    """Parse and validate; at most two append-only repairs follow a bad response."""
    if llm not in ("auto", "anthropic", "offline"):
        raise ValueError("llm must be auto, anthropic, or offline")
    if not isinstance(text, str) or not text.strip():
        return ParseResult(None, ["text: a nonempty request is required"], "offline" if llm == "offline"
                           else "anthropic")
    if llm == "offline":
        return _offline(text)
    messages = [{"role": "user", "content": text}]
    raw = None
    try:
        client = client if client is not None else create_client()
        for attempt in range(3):
            response = client.messages.create(
                **request_options(), system=render_prompt(), messages=list(messages),
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": api_schema()}},
            )
            if _value(response, "stop_reason") == "refusal":
                return ParseResult(None, ["anthropic: request refused"], "anthropic")
            raw = next((_value(block, "text") for block in _value(response, "content", [])
                        if _value(block, "type") == "text"), None)
            try:
                if raw is None:
                    raise ValueError("response: no text block")
                return ParseResult(_validate(json.loads(raw)), source="anthropic", raw=raw)
            except (RequestValidationError, ValueError, TypeError) as exc:
                errors = getattr(exc, "errors", [str(exc)])
                if attempt == 2:
                    return ParseResult(None, errors, "anthropic", raw)
                messages.extend([
                    {"role": "assistant", "content": _assistant_content(response, raw)},
                    {"role": "user", "content": "Correct JSON without inventing values. Validation errors:\n"
                     + "\n".join(errors)},
                ])
    except Exception as exc:
        if llm == "auto" and is_fallback_error(exc):
            return _offline(text, f"offline (anthropic fallback: {type(exc).__name__})")
        return ParseResult(None, [f"anthropic: {type(exc).__name__}"], "anthropic", raw)
    raise AssertionError("unreachable repair state")
