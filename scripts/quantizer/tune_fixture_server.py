"""Fixture server for scripts/quantizer/tune_app_check.mjs (QA of app/web/tune.html, not the app).

The real app (``app.server.create_app``) in live mode on a FAKE GPU: the device report says "Fake GPU", the
catalog is the app tests' fake one, and the quantizer job runner (``POST /api/quantizer/jobs``) is replaced by
one that reports its stages for about two seconds and returns MADE-UP numbers (PPL 26.6, accuracies
0.50 + 0.01 i). A job whose tasks include hellaswag fails, so the page's error path can be seen. The tune
session (--agent) is the packed s3150 one and the Claude answers are the recorded ones (--replay): no GPU,
model or claude is needed. The demo bundle is the repository's (app/web/demo). Runs go to a temporary
directory this process removes. Prints TUNE_FIXTURE_READY <port> once it listens (port 0 = any free port).

usage: python scripts/quantizer/tune_fixture_server.py [port]
"""

from __future__ import annotations

import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import uvicorn  # noqa: E402
from app import server  # noqa: E402
from app.tests.test_server import fake_catalog  # noqa: E402

LOOP = ROOT / "evals" / "fixtures" / "tune_inputs" / "s3150" / "loop.json"
REPLAY = ROOT / "app" / "web" / "demo" / "autotune" / "replay"


def fake_runner(args, progress) -> dict:
    """Stands in for run_quantizer_live: stages over ~2 s, then made-up metrics (hellaswag: a failure)."""
    for stage, at in (("모델 불러오기", 0.05), ("보정", 0.2), ("WikiText-2 PPL", 0.45), ("lm-eval", 0.75)):
        progress(stage, at)
        time.sleep(0.5)
    if "hellaswag" in args.tasks:
        raise RuntimeError("fixture: hellaswag 를 고른 작업은 실패 경로 확인용으로 실패합니다")
    lm = [task for task in args.tasks if task not in ("ppl", "report")]
    tasks = [{"task": task, "label": task, "metric": "acc_norm", "value": 0.5 + 0.01 * i, "stderr": 0.02,
              "n": args.limit} for i, task in enumerate(lm)]
    return {"metrics": {"run": {"ppl": 26.6, "n_windows": 146, "n_tokens": 298862}, "report": None,
                        "lmeval": {"tasks": tasks, "limit": args.limit} if tasks else None},
            "env": {"git_sha": "fixture", "gpu": "Fake GPU"},
            "evidence": {"out_dir": "fixture", "files": [], "layer_map_command": None}}


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.modules["app.catalog"] = fake_catalog()
    server._device_info = lambda device: {"kind": "cuda", "name": "Fake GPU", "backend": "triton"}
    with tempfile.TemporaryDirectory() as tmp:
        app = server.create_app("live", "cuda", server.DEFAULT_DEMO_DIR, runs_dir=Path(tmp) / "runs",
                                quantizer_runner=fake_runner, agent=LOOP, replay=REPLAY)
        sock = socket.socket()
        sock.bind(("127.0.0.1", port))
        print(f"TUNE_FIXTURE_READY {sock.getsockname()[1]}", flush=True)
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
