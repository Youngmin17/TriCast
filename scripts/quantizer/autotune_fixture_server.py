"""Fixture server for scripts/quantizer/autotune_ui_check.mjs (QA of app/web/autotune.html, not the app).

It serves app/web under three path prefixes and fakes only the Auto-tune bundle and the two live endpoints:
- /fixture/  demo/autotune/bundle.json = the repo bundle plus FAKE `requests` (one stored reply per outcome)
             and FAKE layer profiles and report errors; POST api/autotune/parse answers a canned live
             "tune" request (a text containing "422" gets 422 request_refused, "slow" waits 30 s first);
             POST api/autotune/propose answers a canned live round; POST api/autotune/recipe answers a
             grid with the real app.autotune_recipe.grid_recipe on the served bundle, a proposal with a FAKE
             file;
- /down/     the same bundle; parse and propose answer 503 agent_unavailable, recipe answers grids (as the
             server without --agent does) and 503 for a proposal;
- /missing/  the bundle answers 404.
Every other path under a prefix is the file from app/web. The fake bundle is written to a temporary directory
this process creates and removes; no repository file is written. The fake values are derived from the served
table (its methods and blocks) so they stay valid for any table; every number in them is made up.
Prints AUTOTUNE_FIXTURE_READY <port> once it listens (port 0 = any free port).

usage: python scripts/quantizer/autotune_fixture_server.py <repo> [port]
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import math
import signal
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

DELAY_S = 1.5
PREFIXES = ("fixture", "down", "missing")
ROUTES = ("api/autotune/parse", "api/autotune/propose", "api/autotune/recipe")
BUNDLE = "demo/autotune/bundle.json"
DOWN = "실시간 AI 기능은 --agent 로 띄운 서버에서만 됩니다 (fixture 503)."
REFUSED = "exclude_methods: ['no_such_method'] are not methods of the table (fixture 422)"
LLM = {"models": ["fixture-model"], "cost_usd": 0.02, "duration_s": 7.0,
       "started_utc": "2026-11-04T08:00:00+00:00"}


def bits(table: dict[str, Any], method: str, in_features: int) -> Fraction:
    element, scale, per = table["weight_bits"][method]
    return element + Fraction(scale, per or in_features)


def cheapest(table: dict[str, Any], keep: list[int], exclude: list[str], only: list[str] | None) -> Fraction:
    """The smallest average bits under these constraints (the rule of app.tune_request.constrained)."""
    blocks = table["blocks"]
    kept = {block % blocks for block in keep}
    allowed = [m for m in table["methods"] if m not in exclude and (not only or m in only or m == "bf16")]
    total, params = Fraction(0), 0
    for layer in table["layers"]:
        options = [m for j, m in enumerate(table["methods"]) if layer["value"][j] is not None and m in allowed
                   and (layer["block"] not in kept or m == "bf16")]
        total += min(bits(table, m, layer["in_features"]) for m in options) * layer["params"]
        params += layer["params"]
    return total / params


def above(value: Fraction) -> Fraction:
    """``value`` + 1/2 bit, rounded up to 1/128 bit."""
    return Fraction(math.ceil((value + Fraction(1, 2)) * 128), 128)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)


def request(outcome: str, **fields: Any) -> dict[str, Any]:
    base = {"outcome": outcome, "model": None, "budget_bits": None, "compute_budget": None,
            "keep_bf16_blocks": [], "exclude_methods": [], "only_methods": None, "questions": [],
            "reason": None, "assumptions": [], "checks": []}
    return {**base, **fields}


def stored(text: str, reply: dict[str, Any], minute: int) -> dict[str, Any]:
    return {"text": text, "request": reply, "llm": LLM, "recorded_utc": f"2026-11-04T08:0{minute}:00+00:00"}


def profile(block: int) -> dict[str, Any]:
    """A FAKE layer profile: a bell-shaped weight histogram that widens with the block number."""
    sigma = 0.02 + 0.001 * block
    edges = [-4 * sigma + i * (8 * sigma / 24) for i in range(25)]
    raw = [math.exp(-(((edges[i] + edges[i + 1]) / 2) / sigma) ** 2 / 2) for i in range(24)]
    total = sum(raw) * 1.0004
    histogram = {"edges": edges, "fraction": [r / total for r in raw], "below": 0.0002, "above": 0.0002}
    return {
        "weight": {"absmax": 6 * sigma, "std": sigma, "kurtosis": 3.2 + 0.1 * block, "outlier_3sigma": 0.0031,
                   "outlier_6sigma": 0.00002, "row_absmax_ratio": 3.1, "col_absmax_ratio": 4.2,
                   "zero_fraction": 0.0, "mean": 0.0, "histogram": histogram},
        "activation": {"absmax": 41.0, "rms": 1.3, "kurtosis": 27.0, "channel_absmax_ratio": 18.4,
                       "outlier_channels": 0.0078125,
                       "channel_absmax_quantiles": {"p50": 2.1, "p90": 4.0, "p99": 11.0, "max": 41.0}},
        "sparsity": {"magnitude_50": 0.142, "wanda_50": 0.0871, "magnitude_2of4": 0.231, "wanda_2of4": 0.164},
    }


def fake_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The bundle with FAKE requests and profiles, and the canned live parse request."""
    table = bundle["table"]
    model = table["meta"].get("model_id")
    quant = [m for m in table["methods"] if m != "bf16"]
    budget = fraction_text(max(Fraction(6), above(cheapest(table, [0, -1], [quant[-1]], None))))
    floor = float(cheapest(table, [], [], [quant[0]]))
    replies = [
        stored(f"{model} 를 {budget}비트 안에서, 첫 블록과 마지막 블록은 BF16, {quant[-1]} 는 빼고 (fixture)",
               request("tune", model=model, budget_bits=budget, keep_bf16_blocks=[0, -1],
                       exclude_methods=[quant[-1]],
                       assumptions=["fixture: KV 캐시는 양자화하지 않는다고 봤습니다"]),
               0),
        stored("첫 블록만 BF16 으로 두고 최대한 작게 (fixture)",
               request("needs_clarification", keep_bf16_blocks=[0],
                       questions=["어떤 모델인가요?", "크기 예산(가중치당 평균 비트)을 알려 주세요."],
                       checks=["tune without ['model', 'budget_bits']: needs_clarification (code)"]),
               1),
        stored("H100 에서 지연 시간이 가장 짧은 배정을 찾아줘 (fixture)",
               request("out_of_scope", reason="fixture: TriCast 는 지연 시간을 예측하지 않습니다."), 2),
        stored(f"{model} 를 1비트 안에서, 모든 층을 {quant[0]} 로 (fixture)",
               request("conflict", model=model, budget_bits="1", only_methods=[quant[0]],
                       reason=f"budget 1 bits: the cheapest assignment these constraints allow is "
                              f"{floor:.4f} bits (code)",
                       checks=["budget below the constrained cheapest assignment: conflict (code)"]),
               3),
    ]
    live_budget = fraction_text(above(cheapest(table, [-1], [], quant[:2])))
    live = request("tune", model=model, budget_bits=live_budget, keep_bf16_blocks=[-1],
                   only_methods=quant[:2])
    out = json.loads(json.dumps(bundle))
    out["requests"] = replies
    for layer in out["table"]["layers"]:
        layer["profile"] = profile(layer["block"])
        layer["report"] = [[1.25e-5, 27.5, 0.99731] for _ in table["methods"]]
    return out, live


def handler_class(fake: Path, live: dict[str, Any], quant: list[str]) -> type:
    llm = {**LLM, "cost_usd": 0.01, "duration_s": DELAY_S, "started_utc": "2026-11-04T09:00:00+00:00"}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def split(self) -> tuple[str | None, str]:
            path = self.path.split("?", 1)[0].lstrip("/")
            prefix, _, rest = path.partition("/")
            return (prefix, rest) if prefix in PREFIXES else (None, rest)

        def send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server's method name)
            prefix, rest = self.split()
            if prefix is None:
                self.send_error(404, "unknown prefix")
                return
            if rest == BUNDLE:
                if prefix == "missing":
                    self.send_error(404, "no bundle (fixture)")
                    return
                data = fake.read_bytes()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.path = "/" + rest
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            prefix, rest = self.split()
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if prefix is None or rest not in ROUTES:
                missing = {"code": "not_found", "message": f"없는 API 경로입니다: /{rest}"}
                self.send_json(404, {"error": missing})
                return
            if rest == "api/autotune/recipe":
                self.recipe(prefix, body)
                return
            text = str(body.get("text", ""))
            time.sleep(30 if "slow" in text else DELAY_S)
            if prefix != "fixture":
                self.send_json(503, {"error": {"code": "agent_unavailable", "message": DOWN}})
            elif rest == "api/autotune/parse":
                if "422" in text:
                    self.send_json(422, {"error": {"code": "request_refused", "message": REFUSED}})
                else:
                    self.send_json(200, {"mode": "live", "text": text, "request": live, "llm": llm})
            else:
                proposal = {"name": "fixture_live_change", "base": body.get("base"),
                            "changes": [{"block": 5, "kind": "q_proj", "method": quant[0]}], "options": None,
                            "rationale": "fixture: canned live proposal", "cites": [], "status": "accepted",
                            "reason": None, "avg_bits": None, "predicted": None, "kl": None, "ppl": None}
                self.send_json(200, {"mode": "live", "round": 99, "analysis": "fixture: canned analysis",
                                     "stop": False, "proposals": [proposal], "llm": llm,
                                     "note": "fixture: 검사만 했고 측정하지 않았습니다."})

        def recipe(self, prefix: str, body: dict[str, Any]) -> None:
            from app import autotune_recipe as ar  # stdlib only; the check runs this with PYTHONPATH=<repo>

            if body.get("proposal") is not None:
                if prefix != "fixture":
                    self.send_json(503, {"error": {"code": "agent_unavailable", "message": DOWN}})
                    return
                text = f"# FAKE recipe of {body['proposal']} (fixture)\n"
                sha = hashlib.sha256(text.encode()).hexdigest()
                measured = {"id": body["proposal"], "same_file": True, "recorded_sha256": sha, "kl": 0.123,
                            "ppl": 12.5, "gpu": "NVIDIA FIXTURE", "git_sha": "f1x7ure"}
                name = body["proposal"]
                self.send_json(200, {"kind": "proposal", "name": name, "file": f"{name}.yaml", "yaml": text,
                                     "sha256": sha, "default": None, "measured": measured,
                                     "command": "fixture"})
                return
            try:
                served = json.loads(fake.read_text(encoding="utf-8"))
                self.send_json(200, ar.grid_recipe(served, body.get("grid"), body.get("budget_bits")))
            except ar.RecipeError as exc:
                self.send_json(exc.status, {"error": {"code": exc.code, "message": exc.message}})

    return Handler


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    web = repo / "app" / "web"
    bundle = json.loads((web / BUNDLE).read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with tempfile.TemporaryDirectory(prefix="autotune_fixture_") as scratch:
        fake = Path(scratch) / "bundle.json"
        out, live = fake_bundle(bundle)
        fake.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        quant = [m for m in bundle["table"]["methods"] if m != "bf16"]
        handler = functools.partial(handler_class(fake, live, quant), directory=str(web))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        server.daemon_threads = True
        print(f"AUTOTUNE_FIXTURE_READY {server.server_address[1]}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
