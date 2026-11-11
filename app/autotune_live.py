"""One live round of the tune agent for the Auto-tune page (``POST /api/autotune/propose``, served only by
``python -m app.server ... --agent <loop.json>``).

The session's evidence analysis (``app.tune_loop.analyze``) goes to the Claude Code proposer
(``app.tune_loop.claude_code``: no tools, the proposal schema, a spend cap), and every proposal is put
through ``app.tune_agent.propose``'s checks (builds, budgets, tabled methods, cited ledger records, supported
numbers) in a scratch copy of the session, so the page shows which proposals would be accepted or refused and
why. Nothing is measured, and neither the session nor the ledger is written.

For a venue without a network or the claude CLI, ``record_answer`` keeps each live answer (server flag
``--record DIR``) and ``replay_answer`` serves it again (``--replay DIR``), marked ``mode: "replay"``.
"""

from __future__ import annotations

import datetime
import hashlib
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
        # the next round after every recorded one: LLM rounds (llm/round_<n>.json) and measured rounds
        # (rounds/<n>/, which include a replan round that made no LLM call)
        numbers = [int(path.stem.split("_")[1]) for path in (session / "llm").glob("round_*.json")]
        numbers += [int(path.name) for path in (session / "rounds").glob("*") if path.name.isdigit()]
        number = max(numbers, default=0) + 1
        with tempfile.TemporaryDirectory(prefix="autotune_live_") as scratch:
            copy = Path(scratch) / session.name
            shutil.copytree(session, copy)
            scratch_config = {**config, "session": str(copy)}
            try:
                analysis = tl.analyze(scratch_config, root)
                proposer = tl.claude_code(scratch_config, root, runner)
                reply = tl.check_reply(proposer(analysis, number, count))
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
            call = json.loads((copy / "llm" / f"round_{number}.json").read_text(encoding="utf-8"))
        out = json.loads(call["stdout"]) if call.get("stdout") else {}
        llm = {"models": sorted(out.get("modelUsage") or {}), "cost_usd": out.get("total_cost_usd"),
               "duration_s": None if out.get("duration_ms") is None else out["duration_ms"] / 1000,
               "returncode": call.get("returncode"), "started_utc": call.get("started_utc")}
        return {"mode": "live", "round": number, "analysis": reply["analysis"], "stop": reply["stop"],
                "proposals": proposals, "llm": llm,
                "note": "검사만 했고 측정하지 않았습니다. 채택된 제안의 KL 은 GPU 측정이 필요합니다."}
    finally:
        _BUSY.release()


# ---------- recorded answers for an offline demo (app.server --record DIR | --replay DIR) ----------
REPLAY_SCHEMA = "tricast.autotune.replay/1"


def _replay_key(kind: str, value: str | int) -> str:
    """A parse request is keyed by its sentence (white space runs made one space), a round by its count."""
    return " ".join(str(value).split()) if kind == "parse" else f"count={value}"


def _replay_path(folder: Path, kind: str, key: str) -> Path:
    digest = hashlib.sha256((kind + "\n" + key).encode()).hexdigest()[:16]
    return folder / f"{kind}_{digest}.json"


def record_answer(folder: Path, kind: str, value: str | int, answer: dict[str, Any]) -> Path:
    """Keep one live answer (``kind`` "parse" or "propose") so ``replay_answer`` serves it with no claude CLI
    and no network; a later recording of the same request replaces the earlier one."""
    folder.mkdir(parents=True, exist_ok=True)
    key = _replay_key(kind, value)
    entry = {"schema": REPLAY_SCHEMA, "kind": kind, "key": key,
             "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "answer": answer}
    path = _replay_path(folder, kind, key)
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def recorded_keys(folder: Path, kind: str) -> list[str]:
    paths = folder.glob(f"{kind}_*.json")
    return sorted(json.loads(path.read_text(encoding="utf-8"))["key"] for path in paths)


def replay_answer(folder: Path, kind: str, value: str | int) -> dict[str, Any]:
    """The recorded answer to this request, marked ``mode: "replay"`` with the time it was recorded (the page
    badges it as a recording, not a live call). LiveError 404 ``not_recorded`` lists what is recorded."""
    key = _replay_key(kind, value)
    path = _replay_path(folder, kind, key)
    if not path.is_file():
        what = "문장" if kind == "parse" else "제안 수"
        listing = " / ".join(f"“{item}”" for item in recorded_keys(folder, kind)) or "없음"
        raise LiveError(404, "not_recorded",
                        f"오프라인 재생 모드라 미리 녹화한 {what}에만 답합니다. 녹화된 {what}: {listing}")
    entry = json.loads(path.read_text(encoding="utf-8"))
    if entry.get("schema") != REPLAY_SCHEMA or entry.get("kind") != kind or entry.get("key") != key:
        raise LiveError(500, "replay_corrupt", f"{path.name}: 녹화 파일이 이 요청의 것이 아닙니다")
    return {**entry["answer"], "mode": "replay", "recorded_utc": entry["recorded_utc"]}
