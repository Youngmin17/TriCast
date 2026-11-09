"""AI 작성 초안 — 팀 확정 필요

Run, grade and record the tune evals: the two AI stages of Auto-tune on every case of the tune eval set.

What it measures: A1-A8 for tune-request parsing (``app.tune_request``, AC28) and B1-B8 for the
evidence-citing proposals of the Claude proposer (``app.tune_loop``, AC24 · AC27), as ``evals/EVAL_DESIGN.md``
§1 defines them. The criteria belong to people and are only read here: the cases and expected results
(``evals/tune_evalset.jsonl``), the metric names, definitions and targets (the §1 table, read at run time),
and the judge's instructions, output schema and pass rules (``evals/tune_judge_prompt.md``).

    python evals/run_tune_evals.py --table <table.json> --ledger runs/ledger.jsonl \\
        --fixture s3150=<session dir> --mode live --change "<what changed>" --golden "<golden tests result>"
    python evals/run_tune_evals.py ... --mode stored --from-run <run_id>     # re-grade kept replies, no call

Per case:
- request: ``tune_request.ask`` (the query; the --table's model and methods) -> ``tune_request.check`` against
  the --table's bundle (``export_autotune_demo.table_bundle``). A RequestError or a reply that does not parse
  is the case's format failure.
- agent: the analysis kept in ``<session>/llm/round_<n>.json`` (the JSON after the prompt's first line) with
  ``plant`` applied — retrieved: a line appended to the first retrieved document's text; catalog: to the first
  method card's text; budget: the session budget set to the plant's text — goes to the
  ``tune_loop.claude_code`` proposer (at most 4 proposals) in a scratch copy of the session; the reply goes
  through ``tune_loop.check_reply`` and every proposal through ``tune_loop.materialize`` (the AC24 checks) in
  that copy. As in ``app.autotune_live``, neither the session nor the ledger is written. The copy holds the
  session as round n's proposer saw it: proposals created before ``started_utc`` of round_<n>.json only, no
  ``rounds/<m>/`` with m >= n and no ``loop/summary.json`` entry of round >= n (a later round's proposal would
  otherwise refuse the same grid as "already in the session").
  Fixture ``s_min`` (the agent conflict case): unless --fixture names it, it is the ``s3150`` session with the
  budget set to its start proposal's exact ``avg_bits_fraction``, so no layer can be raised for free.
- common checks on every case whatever its graders (request: format, no invented model or budget; agent: reply
  schema, no cited id outside the ledger, no accepted proposal over budget), the ``expect`` keys (grader
  "code"), then the judge (grader "judge": one isolated ``claude -p`` per request case and per accepted
  proposal, the flags of ``tune_request.claude_command`` with the judge's system text and schema).
- pass: every check and every judge verdict passes. An expect key the runner cannot handle, a judge reply that
  does not parse or breaks the schema, a missing fixture or stored reply make the case "error". An error case
  is never dropped: it stays in every denominator, is not passed, counts as failed in the judge metrics (A8,
  B8) and in A1 (an expect key not checked), and every metric reports how many of its cases are errors.

Modes: live calls claude and keeps every raw reply in evals/fixtures/tune/<run_id>/<case_id>.json (``calls``:
role, key, returncode, stdout — kept as JSON when it parses — and stderr); stored answers every call from
those replies (--from-run) and calls nothing. To check the runner, edit a kept reply and re-grade it stored.

Records, never overwritten: evals/results/<run_id>.json (run info, metrics, every case's input, raw output,
checks and judge verdicts), one row appended to evals/results/history.csv, and
evals/results/<run_id>_labeling.md (the judge-graded inputs and outputs with empty label slots and without the
judge's verdicts: people label before they see them, EVAL_DESIGN §3).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):  # python evals/run_tune_evals.py puts only evals/ on the path
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app import export_autotune_demo as ed  # noqa: E402
from app import ledger as lg  # noqa: E402
from app import quantizer_recipe as qr  # noqa: E402
from app import tune_agent as ta  # noqa: E402
from app import tune_loop as tl  # noqa: E402
from app import tune_request as tr  # noqa: E402

EVALSET = ROOT / "evals/tune_evalset.jsonl"
JUDGE_PROMPT = ROOT / "evals/tune_judge_prompt.md"
DESIGN = ROOT / "evals/EVAL_DESIGN.md"
FIXTURES = ROOT / "evals/fixtures/tune"
RESULTS = ROOT / "evals/results"
PROPOSALS = 4  # proposals asked for per agent case
AUTHOR = "tune-evals"
JUDGE_TIMEOUT_S = 300
METRICS = tuple(f"{group}{number}" for group in "AB" for number in range(1, 9))
HISTORY = ("run_id", "commit", "change", "evalset_version", "judge_version", "golden_tests", "mode", *METRICS,
           "cost_usd")
LABELS = ("q1", "q2", "c1", "c2", "c3", "c4")
RECORD_VIEW = ("kind", "layer", "method", "tensor", "metric", "value")  # a cited record in the judge input
RETRIEVED_ID = re.compile(r"\[(E[0-9a-f]+)\]")  # how a ledger document writes a record id
NAMED_ID = re.compile(r"\bE[0-9a-f]{6,}\b")  # an id written in a rationale (app.tune_agent.propose)


class CaseError(Exception):
    """The case cannot run as written (no fixture, no stored reply, a plant that cannot be applied)."""


@dataclass(frozen=True)
class Context:
    table: dict[str, Any]  # bundle form of --table
    table_path: Path
    ledger_path: Path
    ledger: dict[str, dict[str, Any]]  # id -> record
    fixtures: dict[str, Path]
    judge: dict[str, Any]  # judge_prompt()


# ---------- the criteria files (read only) ----------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_evalset(path: Path = EVALSET) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    cases = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    return cases, _sha256(raw)


def judge_prompt(path: Path = JUDGE_PROMPT) -> dict[str, Any]:
    """The version on the first line, the system text between the markers and the fenced JSON schema."""
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0]
    version = re.search(r"`([^`]+)`", first)
    begin, end = "<!-- JUDGE_SYSTEM_BEGIN -->", "<!-- JUDGE_SYSTEM_END -->"
    if begin not in text or end not in text:
        raise ValueError(f"{path}: no {begin} ... {end}")
    system = text.split(begin, 1)[1].split(end, 1)[0].strip()
    block = re.search(r"```json\n(.*?)```", text, re.S)
    if block is None:
        raise ValueError(f"{path}: no ```json schema block")
    schema = json.loads(block.group(1))
    Draft202012Validator.check_schema(schema)
    return {"version": version.group(1) if version else first.strip(), "system": system,
            "system_sha256": _sha256(system.encode()), "schema": schema}


def metric_targets(path: Path = DESIGN) -> dict[str, dict[str, str]]:
    """The §1 table rows: id -> name, definition, grader, target (as written, emphasis removed)."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\|\s*[AB][1-8]\s*\|", line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows[cells[0]] = {"name": cells[1], "definition": cells[2], "grader": cells[3],
                              "target": cells[4].replace("*", "").strip()}
    missing = [metric for metric in METRICS if metric not in rows]
    if missing:
        raise ValueError(f"{path}: no §1 table row for {missing}")
    return rows


# ---------- LLM calls: live (run and keep) or stored (replay) ----------
def _kept(stdout: str) -> Any:
    """A reply's stdout as it is kept: the JSON object when it is one (easy to edit), else the text."""
    try:
        value = json.loads(stdout)
    except ValueError:
        return stdout
    return value if isinstance(value, dict) else stdout


def _replayed(stdout: Any) -> str:
    return stdout if isinstance(stdout, str) else json.dumps(stdout, ensure_ascii=False)


class Replies:
    """The LLM calls of one case. Live (``runner`` given): each call runs and is kept. Stored: each call is
    answered from ``stored`` (a live run's calls, matched by role and key) and nothing runs."""

    def __init__(self, runner: Callable[..., Any] | None, stored: list[dict[str, Any]] | None = None) -> None:
        self._runner, self._stored, self._used = runner, list(stored or []), set()
        self.calls: list[dict[str, Any]] = []

    def runner(self, role: str, key: str) -> Callable[..., Any]:
        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            if self._runner is not None:
                done = self._runner(command, **kwargs)
                self.calls.append({"role": role, "key": key, "returncode": done.returncode,
                                   "stdout": _kept(done.stdout), "stderr": done.stderr or ""})
                return done
            for index, call in enumerate(self._stored):
                if index not in self._used and call["role"] == role and call["key"] == key:
                    self._used.add(index)
                    self.calls.append(call)
                    return subprocess.CompletedProcess(command, call["returncode"],
                                                       stdout=_replayed(call["stdout"]),
                                                       stderr=call.get("stderr", ""))
            raise CaseError(f"no stored {role} reply for {key!r}")
        return run

    def spend(self) -> tuple[list[str], float]:
        """Models named and USD reported by the replies used."""
        envelopes = [call["stdout"] for call in self.calls if isinstance(call["stdout"], dict)]
        models = sorted({model for out in envelopes for model in out.get("modelUsage") or {}})
        return models, sum(out.get("total_cost_usd") or 0.0 for out in envelopes)


# ---------- checks ----------
def _ok(passed: bool, detail: str) -> dict[str, Any]:
    return {"pass": bool(passed), "detail": detail}


def _same_fraction(want: Any, got: Any) -> bool | None:
    if want is None or got is None:
        return want is None and got is None
    try:
        return Fraction(str(want)) == Fraction(str(got))
    except (ValueError, ZeroDivisionError):
        return None


def _fraction_key(field: str) -> Callable[..., dict[str, Any] | None]:
    """A budget compared as a number ("3.50" == "7/2"); None (not handled) when ``want`` does not parse, a
    failure when the reply's value does not."""
    def check(want: Any, result: dict[str, Any], blocks: int, methods: list[str]) -> dict[str, Any] | None:
        if want is not None and _same_fraction(want, want) is None:
            return None
        same = _same_fraction(want, result[field])
        return _ok(bool(same), f"{field} {result[field]!r}, expected {want!r}")
    return check


def _equal_key(field: str) -> Callable[..., dict[str, Any]]:
    def check(want: Any, result: dict[str, Any], blocks: int, methods: list[str]) -> dict[str, Any]:
        return _ok(result[field] == want, f"{field} {result[field]!r}, expected {want!r}")
    return check


def _set_key(field: str) -> Callable[..., dict[str, Any]]:
    def check(want: Any, result: dict[str, Any], blocks: int, methods: list[str]) -> dict[str, Any]:
        got = result[field]
        same = (None if want is None else set(want)) == (None if got is None else set(got))
        return _ok(same, f"{field} {got!r}, expected {want!r} (as sets)")
    return check


def _blocks_key(want: Any, result: dict[str, Any], blocks: int, methods: list[str]) -> dict[str, Any]:
    got = sorted({block % blocks for block in result["keep_bf16_blocks"]})
    wanted = sorted({block % blocks for block in want})
    return _ok(got == wanted, f"keep_bf16_blocks {result['keep_bf16_blocks']} = {got} of {blocks} blocks, "
                              f"expected {want} = {wanted}")


def _outcome_in_key(want: Any, result: dict[str, Any], blocks: int, methods: list[str]) -> dict[str, Any]:
    return _ok(result["outcome"] in want, f"outcome {result['outcome']!r}, expected one of {want}")


def _known_request_key(want: Any, result: dict[str, Any], blocks: int,
                       methods: list[str]) -> dict[str, Any] | None:
    if want is not True:
        return None
    unknown = sorted((set(result["exclude_methods"]) | set(result["only_methods"] or [])) - set(methods))
    return _ok(not unknown, f"methods outside the table: {unknown}")


REQUEST_KEYS: dict[str, Callable[..., dict[str, Any] | None]] = {
    "outcome": _equal_key("outcome"), "model": _equal_key("model"),
    "budget_bits": _fraction_key("budget_bits"), "compute_budget": _fraction_key("compute_budget"),
    "keep_bf16_blocks": _blocks_key, "exclude_methods": _set_key("exclude_methods"),
    "only_methods": _set_key("only_methods"), "outcome_in": _outcome_in_key,
    "no_unknown_methods": _known_request_key}


def request_checks(expect: dict[str, Any], result: dict[str, Any] | None, blocks: int,
                   methods: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Per expect key pass/fail + detail on the checked reply (None: format failure, every key fails), and
    the keys that cannot be handled."""
    checks, errors = {}, []
    for key, want in expect.items():
        check = REQUEST_KEYS.get(key)
        verdict = None if check is None else _ok(False, "no output (format failure)") if result is None \
            else check(want, result, blocks, methods)
        if verdict is None:
            errors.append(f"expect key {key}={want!r} cannot be handled for a request case")
        else:
            checks[key] = verdict
    return checks, errors


def invented(expect: dict[str, Any], result: dict[str, Any] | None) -> list[str]:
    """A2: model / budget_bits the case expects null, filled by the output and not named in assumptions."""
    if result is None:
        return []
    return [name for name in ("model", "budget_bits")
            if name in expect and expect[name] is None and result.get(name) is not None
            and not any(re.search(rf"(?<![\w.]){name}(?![\w.])", text) for text in result["assumptions"])]


def _accepted(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in proposals if row["status"] == "accepted"]


def _cited(row: dict[str, Any], ids: Any) -> list[str]:
    """``ids`` the proposal cites or names in its rationale."""
    return sorted(set(ids) & (set(row["cites"] or []) | set(row["named"])))


def _over_cells(row: dict[str, Any], limits: dict[str, int]) -> list[str]:
    return [f"{method} {row['to_method'].get(method, 0)} > {limit}" for method, limit in limits.items()
            if row["to_method"].get(method, 0) > limit]


def agent_checks(expect: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Per expect key pass/fail + detail on the proposals (all: as replied; accepted: through materialize)."""
    accepted = _accepted(proposals)
    checks, errors = {}, []
    for key, want in expect.items():
        verdict = None
        if key == "budget_ok" and want is True:
            over = [row["name"] for row in accepted if not row["within_budget"]]
            verdict = _ok(not over, f"accepted over budget: {over} (of {len(accepted)} accepted)")
        elif key == "cites_exist" and want is True:
            missing = {row["name"]: row["missing_cites"] for row in accepted if row["missing_cites"]}
            verdict = _ok(not missing, f"accepted cites not in the ledger: {missing}")
        elif key == "rag_cited" and want is True:
            hits = {row["name"]: row["retrieved_cites"] for row in accepted if row["retrieved_cites"]}
            verdict = _ok(bool(hits), f"accepted proposals citing a retrieved document's id: {hits}")
        elif key == "not_cited" and isinstance(want, list):
            hits = {row["name"]: _cited(row, want) for row in proposals if _cited(row, want)}
            verdict = _ok(not hits, f"proposals citing {want}: {hits}")
        elif key == "max_cells_to_method" and isinstance(want, dict):
            over = {row["name"]: _over_cells(row, want) for row in proposals if _over_cells(row, want)}
            verdict = _ok(not over, f"cells switched to a method over its limit {want}: {over}")
        elif key == "no_unknown_methods" and want is True:
            unknown = {row["name"]: row["unknown_methods"] for row in proposals if row["unknown_methods"]}
            verdict = _ok(not unknown, f"methods outside the table: {unknown}")
        if verdict is None:
            errors.append(f"expect key {key}={want!r} cannot be handled for an agent case")
        else:
            checks[key] = verdict
    return checks, errors


# ---------- judge ----------
def judge_command(prompt: dict[str, Any]) -> list[str]:
    """``tune_request.claude_command``'s isolation (no tools, no MCP, no settings, no session, a spend cap)
    with the judge's schema and system text."""
    command = tr.claude_command()
    command[command.index("--json-schema") + 1] = json.dumps(prompt["schema"])
    command[command.index("--system-prompt") + 1] = prompt["system"]
    return command


def required_criteria(kind: str, category: str) -> tuple[str, ...]:
    """The judge prompt's pass rules: A8 = q1 and q2 (clarify cases; other request cases q2, the one that
    applies to them), B8 = c1-c3, conflict cases c4 too."""
    if kind == "request":
        return ("q1", "q2") if category == "clarify" else ("q2",)
    return ("c1", "c2", "c3", "c4") if category == "conflict" else ("c1", "c2", "c3")


def ask_judge(unit: str, payload: dict[str, Any], required: tuple[str, ...], prompt: dict[str, Any],
              runner: Callable[..., Any]) -> dict[str, Any]:
    verdict, error = None, None
    try:
        with tempfile.TemporaryDirectory(prefix="tune_judge_") as empty:
            done = runner(judge_command(prompt), input=json.dumps(payload, ensure_ascii=False),
                          capture_output=True, text=True, cwd=empty, timeout=JUDGE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        return {"unit": unit, "input": payload, "verdict": None, "error": f"timeout: {exc}",
                "required": list(required), "pass": False}
    if done.returncode != 0:
        error = f"claude -p failed (rc {done.returncode}): {(done.stderr or '')[-300:]}"
    else:
        try:
            out = json.loads(done.stdout)
            verdict = out.get("structured_output") or out.get("result")
            verdict = json.loads(verdict) if isinstance(verdict, str) else verdict
        except (ValueError, AttributeError) as exc:
            verdict, error = None, f"the judge reply does not parse: {exc}"
        else:
            problems = list(Draft202012Validator(prompt["schema"]).iter_errors(verdict))
            if problems:
                verdict, error = None, f"the judge reply breaks the schema: {problems[0].message}"
    passed = verdict is not None and all(isinstance(verdict[name], dict) and verdict[name]["pass"] is True
                                         for name in required)
    return {"unit": unit, "input": payload, "verdict": verdict, "error": error, "required": list(required),
            "pass": passed}


# ---------- cases ----------
def _request(case: dict[str, Any], ctx: Context, replies: Replies, record: dict[str, Any]) -> None:
    table, models = ctx.table, [ctx.table["meta"]["model_id"]]
    reply, result, error = None, None, None
    try:
        asked = tr.ask(case["query"], models, table["methods"], None, replies.runner("request", "request"))
        reply = asked["reply"]
        result = tr.check(reply, table, models)
    except (ValueError, AttributeError, TypeError, subprocess.TimeoutExpired) as exc:  # RequestError, JSON
        error = f"{type(exc).__name__}: {exc}"
    record["output"] = {"reply": reply, "result": result, "format_error": error}
    names = invented(case["expect"], result)
    record["invented"] = names
    record["checks"] = {
        "format": _ok(error is None, error or "schema and code checks passed (no RequestError)"),
        "no_invented_values": _ok(not names, f"filled without an assumption: {names}")}
    if "code" in case["graders"]:
        checks, errors = request_checks(case["expect"], result, table["blocks"], table["methods"])
        record["expect_checks"] = list(checks)
        record["checks"].update(checks)
        record["errors"] += errors
    if "judge" in case["graders"]:
        if result is None:
            record["judge"] = [{"unit": "request", "input": None, "verdict": None, "error": None,
                                "pass": False, "note": "not asked: the reply failed the format checks"}]
        else:
            payload = {"case_kind": "request", "category": case["category"], "query": case["query"],
                       **{key: result[key] for key in ("outcome", "questions", "assumptions", "model",
                                                       "budget_bits")}}
            record["judge"] = [ask_judge("request", payload, required_criteria("request", case["category"]),
                                         ctx.judge, replies.runner("judge", "request"))]


def resolve_fixture(name: str, fixtures: dict[str, Path]) -> tuple[Path, str | None]:
    """(session, budget override). ``s_min`` without its own --fixture: s3150 at its start's exact bits."""
    if name in fixtures:
        return fixtures[name], None
    if name == "s_min" and "s3150" in fixtures:
        for path in sorted((fixtures["s3150"] / "proposals").glob("*/proposal.json")):
            proposal = json.loads(path.read_text(encoding="utf-8"))
            if proposal["parent"] is None:
                return fixtures["s3150"], proposal["avg_bits_fraction"]
        raise CaseError(f"{fixtures['s3150']}: no start proposal (parent null) to set s_min's budget")
    raise CaseError(f"no fixture {name!r}: pass --fixture {name}=<session dir>")


def stored_analysis(session: Path, number: int) -> dict[str, Any]:
    path = session / "llm" / f"round_{number}.json"
    if not path.is_file():
        raise CaseError(f"{path}: no stored round")
    prompt = json.loads(path.read_text(encoding="utf-8"))["prompt"]
    return json.loads(prompt.split("\n", 1)[1])


def _utc(text: str) -> datetime.datetime:
    moment = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def as_of_round(copy: Path, number: int) -> dict[str, list[Any]]:
    """Cut the scratch ``copy`` back to the session round ``number``'s proposer saw (module docstring);
    returns what was dropped."""
    try:
        started = _utc(json.loads((copy / "llm" / f"round_{number}.json").read_text(
            encoding="utf-8"))["started_utc"])
        made = {path.parent: _utc(json.loads(path.read_text(encoding="utf-8"))["created_utc"])
                for path in sorted((copy / "proposals").glob("*/proposal.json"))}
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        raise CaseError(f"round {number}: no comparable started_utc / created_utc ({exc})") from exc
    dropped: dict[str, list[Any]] = {"proposals": [], "rounds": [], "summary_rounds": []}
    for folder, moment in made.items():
        if moment >= started:
            shutil.rmtree(folder)
            dropped["proposals"].append(folder.name)
    for folder in sorted((copy / "rounds").glob("*")):
        if folder.is_dir() and folder.name.isdigit() and int(folder.name) >= number:
            shutil.rmtree(folder)
            dropped["rounds"].append(int(folder.name))
    summary = copy / "loop" / "summary.json"
    if summary.is_file():
        entries = json.loads(summary.read_text(encoding="utf-8"))
        dropped["summary_rounds"] = [entry["round"] for entry in entries if entry["round"] >= number]
        summary.write_text(json.dumps([entry for entry in entries if entry["round"] < number], indent=1,
                                      ensure_ascii=False), encoding="utf-8")
    return dropped


def apply_plant(analysis: dict[str, Any], plant: dict[str, Any]) -> None:
    where, text = plant.get("where"), plant.get("text")
    if where == "retrieved":
        documents = [doc for item in analysis.get("retrieved") or [] for doc in item.get("documents") or []]
        if not documents:
            raise CaseError("plant where=retrieved: the stored analysis has no retrieved document")
        documents[0]["text"] += "\n" + text
    elif where == "catalog":
        if not analysis.get("catalog"):
            raise CaseError("plant where=catalog: the stored analysis has no method card")
        analysis["catalog"][0]["text"] += "\n" + text
    elif where == "budget":
        try:
            Fraction(str(text))
        except (ValueError, ZeroDivisionError) as exc:
            raise CaseError(f"plant where=budget: {text!r} is not a number or fraction") from exc
        analysis["budget_bits"] = str(text)
    else:
        raise CaseError(f"plant where={where!r} cannot be handled")


def _grid(session: Path, name: Any) -> list[list[str]] | None:
    try:
        return ta.read_proposal(session, name)["spec"]["grid"]
    except (ta.TuneError, TypeError):
        return None


def _cells_to(base: list[list[str]] | None, changes: list[dict[str, Any]]) -> dict[str, int]:
    """Cells each method is switched to (a change to the base's own method is no switch; the last change of a
    cell wins, as in materialize)."""
    cells = {}
    for change in changes:
        block, kind = change["block"], change["kind"]
        inside = base is not None and 0 <= block < len(base) and kind in qr.KINDS
        cells[(block, kind)] = None if inside and base[block][qr.KINDS.index(kind)] == change["method"] \
            else change["method"]
    return dict(Counter(method for method in cells.values() if method is not None))


def _agent(case: dict[str, Any], ctx: Context, replies: Replies, record: dict[str, Any]) -> None:
    query = case["query"]
    session, budget = resolve_fixture(query["fixture"], ctx.fixtures)
    try:
        fixed = ta.read_session(session)
    except ta.TuneError as exc:
        raise CaseError(str(exc)) from exc
    if fixed["table"]["sha256"] != ctx.table["sha256"]:  # else every proposal would be refused for it
        raise CaseError(f"{session} was run on {fixed['table']['path']}, not on --table")
    analysis = stored_analysis(session, query["round"])
    if budget is not None:
        analysis["budget_bits"] = budget
    if query.get("plant"):
        apply_plant(analysis, query["plant"])
    budget = Fraction(analysis["budget_bits"])
    compute = (analysis.get("compute") or {}).get("budget")
    retrieved = {match for item in analysis.get("retrieved") or [] for doc in item.get("documents") or []
                 for match in RETRIEVED_ID.findall(doc["text"])}
    methods = set(ctx.table["methods"])
    reply, error, proposals = None, None, []
    with tempfile.TemporaryDirectory(prefix="tune_evals_") as scratch:
        copy = Path(scratch) / session.name
        shutil.copytree(session, copy)
        dropped = as_of_round(copy, query["round"])
        config = {"session": str(copy), "table": str(ctx.table_path), "budget_bits": str(budget),
                  "ledger": str(ctx.ledger_path), **({} if compute is None else {"compute_budget": compute})}
        try:
            proposer = tl.claude_code(config, ROOT, replies.runner("proposer", f"round_{query['round']}"))
            reply = tl.check_reply(proposer(analysis, query["round"], PROPOSALS))
        except (ValueError, AttributeError, TypeError, subprocess.TimeoutExpired) as exc:  # LoopError, JSON
            error = f"{type(exc).__name__}: {exc}"
        for item in (reply or {}).get("proposals", []):
            base = _grid(copy, item["base"])
            row = {key: item[key] for key in ("name", "base", "changes", "options", "rationale", "cites")}
            used = {change["method"] for change in item["changes"]} | set(item["options"] or {})
            row.update(named=NAMED_ID.findall(item["rationale"]), to_method=_cells_to(base, item["changes"]),
                       unknown_methods=sorted(used - methods))
            try:
                made = tl.materialize(config, ROOT, item, AUTHOR)
            except (KeyError, IndexError, TypeError, ValueError) as exc:  # TuneError is a ValueError
                row.update(status="refused", reason=str(exc))
            else:
                bits, spent = Fraction(made["avg_bits_fraction"]), made.get("avg_compute_fraction")
                within = bits <= budget and (compute is None or Fraction(spent) <= Fraction(compute))
                grid = made["spec"]["grid"]
                changed = None if base is None else sum(a != b for new, old in zip(grid, base, strict=True)
                                                        for a, b in zip(new, old, strict=True))
                row.update(status="accepted", reason=None, avg_bits_fraction=str(bits),
                           avg_compute_fraction=spent, within_budget=within,
                           missing_cites=[key for key in item["cites"] if key not in ctx.ledger],
                           retrieved_cites=sorted(set(item["cites"]) & retrieved), changed_cells=changed,
                           predicted_minus_floor=made["predicted"]["predicted_kl_sum_minus_floor"],
                           recipe_sha256=made["recipe_sha256"])
            proposals.append(row)
        try:
            state = {row["name"]: row for row in ta.state(copy, ctx.ledger_path)["proposals"]}
        except ta.TuneError as exc:
            raise CaseError(f"{session}: {exc}") from exc
    first = next((row for row in state.values() if row["parent"] is None), None)
    start = None if first is None else {
        "name": first["name"], "predicted_minus_floor": (first["predicted"] or {}).get(
            "predicted_kl_sum_minus_floor"),
        "measured_kl": first["measured_kl"][0]["kl"] if first["measured_kl"] else None}
    for row in _accepted(proposals):
        measured = state[row["name"]]["measured_kl"]
        row["measured_kl"] = measured[0]["kl"] if measured else None
    record["output"] = {"session": str(session), "budget_bits": str(budget), "compute_budget": compute,
                        "plant": query.get("plant"), "reply": reply, "reply_error": error,
                        "proposals": proposals, "start": start, "retrieved_ids": sorted(retrieved),
                        "dropped_from_scratch": dropped}
    accepted = _accepted(proposals)
    record["checks"] = {
        "format": _ok(error is None, error or "the reply follows the proposal schema"),
        "no_hallucinated_cites": _ok(not any(row["missing_cites"] for row in accepted),
                                     "accepted cites all in the ledger"),
        "within_budget": _ok(all(row["within_budget"] for row in accepted),
                             "accepted proposals within budget")}
    if "code" in case["graders"]:
        checks, errors = agent_checks(case["expect"], proposals)
        record["expect_checks"] = list(checks)
        record["checks"].update(checks)
        record["errors"] += errors
    if "judge" in case["graders"]:
        record["judge"] = []
        required = required_criteria("agent", case["category"])
        for row in accepted:
            payload = {"case_kind": "agent", "category": case["category"],
                       "proposal": {key: row[key] for key in ("changes", "options", "rationale", "cites")},
                       "cited_records": {key: {field: ctx.ledger.get(key, {}).get(field)
                                               for field in RECORD_VIEW} for key in row["cites"]},
                       "retrieved": analysis.get("retrieved") or [], "budget_bits": str(budget)}
            record["judge"].append(ask_judge(row["name"], payload, required, ctx.judge,
                                             replies.runner("judge", row["name"])))


def run_case(case: dict[str, Any], ctx: Context, replies: Replies) -> dict[str, Any]:
    record: dict[str, Any] = {"id": case["id"], "kind": case["kind"], "category": case["category"],
                              "graders": case["graders"], "input": case["query"], "expect": case["expect"],
                              "output": None, "checks": {}, "expect_checks": [], "invented": [],
                              "judge": None, "errors": []}
    unknown = sorted(set(case["graders"]) - {"code", "judge"})
    if unknown:
        record["errors"].append(f"graders {unknown} cannot be handled")
    try:
        if case["kind"] == "request":
            _request(case, ctx, replies, record)
        elif case["kind"] == "agent":
            _agent(case, ctx, replies, record)
        else:
            record["errors"].append(f"kind {case['kind']!r} cannot be handled")
    except CaseError as exc:
        record["errors"].append(str(exc))
    record["errors"] += [f"judge {unit['unit']}: {unit['error']}" for unit in record["judge"] or []
                         if unit["error"]]
    models, cost = replies.spend()
    record["llm"] = {"models": models, "cost_usd": cost,
                     "calls": [{key: call[key] for key in ("role", "key", "returncode")}
                               for call in replies.calls]}
    passed = all(check["pass"] for check in record["checks"].values()) and all(
        unit["pass"] for unit in record["judge"] or [])
    record["status"] = "error" if record["errors"] else "pass" if passed and record["checks"] else "fail"
    return record


# ---------- metrics ----------
def _met(target: str, numerator: int, denominator: int) -> bool | None:
    """Whether the §1 target is met; None for a reference metric, a target without a percentage or a count of
    zero, and a metric with no case to count (0/0 is "not applicable", never "met")."""
    if "참고" in target or denominator == 0:
        return None
    if "0건" in target:
        return numerator == 0
    match = re.search(r"(≥|>=)?\s*(\d+(?:\.\d+)?)\s*%", target)  # "≥ 87.5%" reads 87.5, not 5
    if match is None:
        return None
    rate, goal = Fraction(numerator, denominator), Fraction(match.group(2)) / 100
    return rate >= goal if match.group(1) else rate == goal


def metrics(records: list[dict[str, Any]], targets: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """EVAL_DESIGN §1 over ``records``: numerator, denominator, value (a count for the absolute rules, else
    numerator / denominator), the target as written and whether it is met (None: reference or no case)."""
    requests = [r for r in records if r["kind"] == "request"]
    agents = [r for r in records if r["kind"] == "agent"]

    def of(category: str) -> list[dict[str, Any]]:
        return [r for r in requests if r["category"] == category]

    def result(record: dict[str, Any]) -> dict[str, Any]:
        return (record["output"] or {}).get("result") or {}

    def expect_ok(record: dict[str, Any]) -> bool:
        keys = record["expect_checks"]
        return bool(result(record)) and len(keys) == len(record["expect"]) and all(
            record["checks"][key]["pass"] for key in keys)

    def judged_ok(record: dict[str, Any]) -> bool:
        return bool(record["judge"]) and not record["errors"] and all(
            unit["pass"] for unit in record["judge"])

    proposals = [(r, row) for r in agents for row in (r["output"] or {}).get("proposals", [])]
    accepted = [(r, row) for r, row in proposals if row["status"] == "accepted"]
    planted = [(r, row) for r, row in proposals if {"not_cited", "max_cells_to_method"} & set(r["expect"])]
    # B6: cell changes and a table prediction (option changes the table cannot predict are left out)
    changed = [(r, row) for r, row in accepted
               if row["changed_cells"] and row["predicted_minus_floor"] is not None]
    measured = [(r, row) for r, row in accepted if row["measured_kl"] is not None]
    judged = [(r, row) for r, row in accepted if "judge" in r["graders"]]
    verdicts = {(r["id"], unit["unit"]): unit for r in agents for unit in r["judge"] or []}

    def start(record: dict[str, Any], key: str) -> float | None:
        return ((record["output"] or {}).get("start") or {}).get(key)

    def followed(record: dict[str, Any], row: dict[str, Any]) -> bool:
        expect = record["expect"]
        return bool(_cited(row, expect.get("not_cited") or []) or _over_cells(row, expect.get(
            "max_cells_to_method") or {}))

    clarify = [r for r in of("clarify") if "judge" in r["graders"]]
    cases = {"A1": of("constraint"), "A2": requests, "A3": of("clarify"), "A4": of("out_of_scope"),
             "A5": of("conflict"), "A6": of("injection"), "A7": requests, "A8": clarify,
             **{f"B{number}": agents for number in range(1, 8)},
             "B8": [r for r in agents if "judge" in r["graders"]]}
    counts = {
        "A1": (sum(expect_ok(r) for r in of("constraint")), len(of("constraint"))),
        "A2": (sum(len(r["invented"]) for r in requests),
               sum(name in r["expect"] and r["expect"][name] is None for r in requests
                   for name in ("model", "budget_bits"))),
        "A3": (sum(result(r).get("outcome") == "needs_clarification"
                   and any(question.strip() for question in result(r).get("questions") or [])
                   for r in of("clarify")), len(of("clarify"))),
        "A4": (sum(result(r).get("outcome") == "out_of_scope" for r in of("out_of_scope")),
               len(of("out_of_scope"))),
        "A5": (sum(result(r).get("outcome") == "conflict" for r in of("conflict")), len(of("conflict"))),
        "A6": (sum(bool(result(r)) and not all(r["checks"][key]["pass"] for key in r["expect_checks"])
                   for r in of("injection")), len(of("injection"))),
        "A7": (sum(bool(r["output"]) and r["output"]["format_error"] is None
                   and r["output"]["result"] is not None for r in requests), len(requests)),
        "A8": (sum(judged_ok(r) for r in clarify), len(clarify)),
        "B1": (len(accepted), len(proposals)),
        "B2": (sum(len(row["missing_cites"]) for _, row in accepted),
               sum(len(row["cites"]) for _, row in accepted)),
        "B3": (sum(not row["within_budget"] for _, row in accepted), len(accepted)),
        "B4": (sum(bool(row["retrieved_cites"]) for _, row in accepted), len(accepted)),
        "B5": (sum(followed(r, row) for r, row in planted), len(planted)),
        "B6": (sum(start(r, "predicted_minus_floor") is not None
                   and row["predicted_minus_floor"] < start(r, "predicted_minus_floor")
                   for r, row in changed), len(changed)),
        "B7": (sum(start(r, "measured_kl") is not None and row["measured_kl"] < start(r, "measured_kl")
                   for r, row in measured), len(measured)),
        "B8": (sum(verdicts.get((r["id"], row["name"]), {}).get("pass") is True and not r["errors"]
                   for r, row in judged), len(judged)),
    }
    out = {}
    for metric in METRICS:
        numerator, denominator = counts[metric]
        target = targets[metric]["target"]
        absolute = "0건" in target
        value = numerator if absolute else (numerator / denominator if denominator else None)
        out[metric] = {**targets[metric], "kind": "count" if absolute else "rate", "numerator": numerator,
                       "denominator": denominator, "value": value,
                       "met": _met(target, numerator, denominator),
                       "error_cases": sum(r["status"] == "error" for r in cases[metric])}
    return out


# ---------- records ----------
def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def new_run_id(out: Path, now: datetime.datetime) -> str:
    """<YYYY-MM-DDTHHMM> (UTC), with _2, _3, ... when a record or kept replies of that id exist."""
    base = now.strftime("%Y-%m-%dT%H%M")
    run_id, number = base, 1
    while any(path.exists()
              for path in (out / f"{run_id}.json", out / f"{run_id}_labeling.md", FIXTURES / run_id)):
        number += 1
        run_id = f"{base}_{number}"
    return run_id


def _cell(value: Any) -> str:
    return "" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def append_history(path: Path, run: dict[str, Any], table: dict[str, dict[str, Any]]) -> None:
    new = not path.exists()
    commit = None if run["git_commit"] is None else run["git_commit"] + ("-dirty" if run["git_dirty"] else "")
    row = {"run_id": run["run_id"], "commit": commit, "change": run["change"],
           "evalset_version": run["evalset"]["version"], "judge_version": run["judge"]["version_id"],
           "golden_tests": run["golden_tests"], "mode": run["mode"], "cost_usd": run["cost_usd"],
           **{metric: table[metric]["value"] for metric in METRICS}}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(HISTORY)
        writer.writerow([_cell(row[column]) for column in HISTORY])


def _label_slots() -> list[str]:
    return ["", "| 기준 | 라벨 (true / false / null) | 이유 |", "|---|---|---|",
            *[f"| {name} | | |" for name in LABELS], ""]


def labeling_sheet(run: dict[str, Any], records: list[dict[str, Any]],
                   ledger: dict[str, dict[str, Any]]) -> str:
    """The judge-graded cases' inputs and outputs, one label table per judge call, no judge verdicts."""
    lines = [f"# 라벨링 시트 — {run['run_id']}", "",
             f"judge 출력을 보기 전에 라벨한다 (evals/EVAL_DESIGN.md §3). 기준: evals/tune_judge_prompt.md "
             f"(judge 버전 {run['judge']['version']}).",
             "칸마다 true / false / null (적용 안 됨) 과 한 줄 이유.",
             f"실행: {run['mode']} · 커밋 {run['git_commit']} · 평가 세트 {run['evalset']['version']}", ""]
    for record in records:
        if "judge" not in record["graders"]:
            continue
        lines += [f"## {record['id']} · {record['kind']} · {record['category']}", ""]
        output = record["output"] or {}
        if record["kind"] == "request":
            lines.append(f"- 요청문: {json.dumps(record['input'], ensure_ascii=False)}")
            result = output.get("result")
            if result is None:
                lines += [f"- 출력: 형식 오류 — {output.get('format_error') or record['errors']}",
                          "- judge 호출 없음", ""]
                continue
            lines += [f"- {key}: {json.dumps(result[key], ensure_ascii=False)}"
                      for key in ("outcome", "model", "budget_bits", "keep_bf16_blocks", "exclude_methods",
                                  "only_methods", "questions", "assumptions", "reason")]
            lines += _label_slots()
            continue
        query = record["input"]
        lines.append(f"- 입력: fixture {query.get('fixture')} · round {query.get('round')} · 예산 "
                     f"{output.get('budget_bits')}"
                     + ("" if not query.get("plant") else f" · 심은 글 ({query['plant'].get('where')}): "
                        f"{json.dumps(query['plant'].get('text'), ensure_ascii=False)}"))
        accepted = _accepted(output.get("proposals") or [])
        if not accepted:
            reply = output.get("reply") or {}
            lines += [f"- 채택된 제안 없음 (judge 호출 없음) · stop {reply.get('stop')} · 응답 분석: "
                      f"{json.dumps(reply.get('analysis'), ensure_ascii=False)}", ""]
            continue
        lines.append("")
        for row in accepted:
            lines += [f"### 제안 {row['name']} (base {row['base']}, {row['avg_bits_fraction']} bits)", "",
                      "- 변경: " + ("; ".join(f"block {c['block']} {c['kind']} → {c['method']}"
                                             for c in row["changes"]) or "없음"),
                      f"- options: {json.dumps(row['options'], ensure_ascii=False)}",
                      f"- 근거 글: {row['rationale']}", "- 인용:"]
            lines += [f"  - {key}: " + " · ".join(f"{field} {ledger.get(key, {}).get(field)}"
                                                  for field in RECORD_VIEW) for key in row["cites"]]
            lines += _label_slots()
    return "\n".join(lines) + "\n"


def _line(record: dict[str, Any]) -> str:
    applied = list(record["checks"]) + (["judge"] if record["judge"] is not None else [])
    failed = [name for name, check in record["checks"].items() if not check["pass"]]
    failed += [f"judge:{unit['unit']}" for unit in record["judge"] or [] if not unit["pass"]]
    why = record["errors"] if record["status"] == "error" else failed if record["status"] == "fail" else ""
    return (f"{record['id']:<7} {record['kind']:<7} {record['category']:<13} checks={','.join(applied)} "
            f"-> {record['status']}" + (f"  {why}" if why else ""))


def main(argv: list[str] | None = None, runner: Callable[..., Any] = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--table", type=Path, required=True, help="the auto-tune table (table.json)")
    parser.add_argument("--ledger", type=Path, required=True, help="the evidence ledger (ledger.jsonl)")
    parser.add_argument("--fixture", action="append", default=[], metavar="NAME=DIR",
                        help="a tune-loop session directory for the agent cases' query.fixture")
    parser.add_argument("--mode", choices=("stored", "live"), required=True)
    parser.add_argument("--from-run", help="stored mode: the run id whose kept replies are re-graded")
    parser.add_argument("--change", required=True, help="what changed since the last run")
    parser.add_argument("--golden", required=True, help="the golden tests' result on this commit")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if (args.mode == "stored") != (args.from_run is not None):
        parser.error("--from-run <run_id> goes with --mode stored, and only with it")
    fixtures = {}
    for item in args.fixture:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--fixture {item!r}: use NAME=<session dir>")
        fixtures[name] = Path(path).resolve()
    try:
        cases, evalset_sha = load_evalset(EVALSET)
        judge = judge_prompt(JUDGE_PROMPT)
        targets = metric_targets(DESIGN)
        table = ed.table_bundle(args.table)
        ledger = {record["id"]: record for record in lg.read_ledger(args.ledger)}
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    only = None if args.only is None else [key.strip() for key in args.only.split(",") if key.strip()]
    unknown = sorted(set(only or []) - {case["id"] for case in cases})
    if unknown:
        parser.error(f"--only: no cases {unknown}")
    selected = [case for case in cases if only is None or case["id"] in only]
    stored = None if args.from_run is None else FIXTURES / args.from_run
    if stored is not None and not stored.is_dir():
        parser.exit(2, f"error: no kept replies {stored}\n")
    started = datetime.datetime.now(datetime.timezone.utc)
    # the tree as the run starts: read before this run writes its kept replies or results into it
    commit, status = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(args.out, started)
    kept = FIXTURES / run_id
    if stored is None:
        kept.mkdir(parents=True)
    ctx = Context(table=table, table_path=args.table.resolve(), ledger_path=args.ledger.resolve(),
                  ledger=ledger, fixtures=fixtures, judge=judge)
    records = []
    for case in selected:
        if stored is None:
            replies = Replies(runner)
        else:
            path = stored / f"{case['id']}.json"
            calls = json.loads(path.read_text(encoding="utf-8"))["calls"] if path.is_file() else []
            replies = Replies(None, calls)
        record = run_case(case, ctx, replies)
        if stored is None:
            keep = {"run_id": run_id, "case_id": case["id"], "calls": replies.calls}
            (kept / f"{case['id']}.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
        records.append(record)
        print(_line(record))
    table_metrics = metrics(records, targets)
    models = sorted({model for record in records for model in record["llm"]["models"]})
    replied = sum(record["llm"]["cost_usd"] for record in records)
    version = f"{len(cases)}@{evalset_sha}" + ("" if only is None else f"|only={','.join(only)}")
    run = {"run_id": run_id, "started_utc": started.isoformat(timespec="seconds"),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "git_commit": commit, "git_dirty": None if status is None else bool(status), "change": args.change,
           "golden_tests": args.golden, "mode": args.mode, "from_run": args.from_run,
           "evalset": {"path": str(EVALSET.relative_to(ROOT)), "cases": len(cases), "sha256": evalset_sha,
                       "only": only, "version": version},
           "judge": {"path": str(JUDGE_PROMPT.relative_to(ROOT)), "version": judge["version"],
                     "system_sha256": judge["system_sha256"],
                     "version_id": f"{judge['version']}@{judge['system_sha256']}"},
           "design": {"path": str(DESIGN.relative_to(ROOT)), "sha256": _sha256(DESIGN.read_bytes())},
           "table": {"path": str(args.table), "sha256": table["sha256"]},
           "ledger": {"path": str(args.ledger), "records": len(ledger)},
           "fixtures": {name: str(path) for name, path in fixtures.items()},
           "models": models, "cost_usd": replied if stored is None else 0.0, "replies_cost_usd": replied,
           "cases": dict(Counter(record["status"] for record in records), total=len(records))}
    with (args.out / f"{run_id}.json").open("x", encoding="utf-8") as handle:
        json.dump({"run": run, "metrics": table_metrics, "cases": records}, handle, indent=1,
                  ensure_ascii=False)
        handle.write("\n")
    append_history(args.out / "history.csv", run, table_metrics)
    with (args.out / f"{run_id}_labeling.md").open("x", encoding="utf-8") as handle:
        handle.write(labeling_sheet(run, records, ledger))
    print(f"{'metric':<6} {'value':>8} {'num/den':>9}  {'target':<16} met   errors  name")
    for metric, row in table_metrics.items():
        met = {True: "yes", False: "no", None: "n/a"}[row["met"]]
        print(f"{metric:<6} {_cell(row['value']) or 'n/a':>8} {row['numerator']:>4}/{row['denominator']:<4}  "
              f"{row['target']:<16} {met:<5} {row['error_cases']:<7} {row['name']}")
    print(f"TUNE_EVALS_DONE run={run_id} mode={args.mode} cases={run['cases']} "
          f"out={args.out / (run_id + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
