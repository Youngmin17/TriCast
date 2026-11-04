"""One live round of the tune agent for the Auto-tune page (``POST /api/autotune/propose``, served only by
``python -m app.server ... --agent <loop.json>``).

The session's evidence analysis (``app.tune_loop.analyze``) goes to the Claude Code proposer
(``app.tune_loop.claude_code``: no tools, the proposal schema, a spend cap), and every proposal is put
through ``app.tune_agent.propose``'s checks (builds, budgets, tabled methods, cited ledger records, supported
numbers) in a scratch copy of the session, so the page shows which proposals would be accepted or refused and
why. Nothing is measured, and neither the session nor the ledger is written.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_PROPOSALS = 8
_BUSY = threading.Lock()


class LiveError(ValueError):
    """A refused request or a failed round; ``status`` is the HTTP status the server answers with."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("session", "table", "budget_bits"):
        if key not in config:
            raise LiveError(500, "agent_config", f"{path}: loop.json 에 {key} 가 없습니다")
    return config


def parse_request(config: dict[str, Any], root: Path, text: str,
                  runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """One tune request sentence through ``app.tune_request`` against the session's table."""
    from app import export_autotune_demo as ed
    from app import tune_request as tr

    table = ed.table_bundle(root / config["table"])
    models = [table["meta"]["model_id"]]
    try:
        raw = tr.ask(text, models, table["methods"], config.get("llm"), runner)
        result = tr.check(raw["reply"], table, models)
    except tr.RequestError as exc:
        raise LiveError(422, "request_refused", str(exc)) from exc
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise LiveError(502, "agent_failed", f"요청 해석 호출이 실패했습니다: {exc}") from exc
    call = {key: value for key, value in raw["call"].items() if key != "prompt"}
    return {"mode": "live", "text": text, "request": result, "llm": call}


def propose_round(config: dict[str, Any], root: Path, count: int,
                  runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Ask the proposer for at most ``count`` proposals on the session's current evidence and check each."""
    from app import tune_agent as ta
    from app import tune_loop as tl

    if not 1 <= count <= MAX_PROPOSALS:
        raise LiveError(400, "invalid_request",
                        f"count 는 1 이상 {MAX_PROPOSALS} 이하여야 합니다 (지금 {count}).")
    if not _BUSY.acquire(blocking=False):
        raise LiveError(409, "agent_busy", "다른 제안 요청이 진행 중입니다. 끝난 뒤 다시 요청하세요.")
    try:
        session = root / config["session"]
        rounds = len(list((session / "llm").glob("round_*.json")))
        with tempfile.TemporaryDirectory(prefix="autotune_live_") as scratch:
            copy = Path(scratch) / session.name
            shutil.copytree(session, copy)
            scratch_config = {**config, "session": str(copy)}
            try:
                analysis = tl.analyze(scratch_config, root)
                proposer = tl.claude_code(scratch_config, root, runner)
                reply = tl.check_reply(proposer(analysis, rounds + 1, count))
            except (tl.LoopError, ta.TuneError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                raise LiveError(502, "agent_failed", f"제안 엔진 호출이 실패했습니다: {exc}") from exc
            proposals = []
            for item in reply["proposals"]:
                row = {key: item.get(key)
                       for key in ("name", "base", "changes", "options", "rationale", "cites")}
                try:
                    accepted = tl.materialize(scratch_config, root, item, "claude-code-live")
                except (ta.TuneError, KeyError, IndexError, ValueError) as exc:
                    row.update(status="refused", reason=str(exc))
                else:
                    row.update(status="accepted", reason=None, avg_bits=accepted["avg_bits"],
                               predicted=accepted["predicted"], grid=accepted["spec"]["grid"],
                               kl=None, ppl=None)
                proposals.append(row)
            call = json.loads((copy / "llm" / f"round_{rounds + 1}.json").read_text(encoding="utf-8"))
        out = json.loads(call["stdout"]) if call.get("stdout") else {}
        llm = {"models": sorted(out.get("modelUsage") or {}), "cost_usd": out.get("total_cost_usd"),
               "duration_s": None if out.get("duration_ms") is None else out["duration_ms"] / 1000,
               "returncode": call.get("returncode"), "started_utc": call.get("started_utc")}
        return {"mode": "live", "round": rounds + 1, "analysis": reply["analysis"], "stop": reply["stop"],
                "proposals": proposals, "llm": llm,
                "note": "검사만 했고 측정하지 않았습니다. 채택된 제안의 KL 은 GPU 측정이 필요합니다."}
    finally:
        _BUSY.release()
