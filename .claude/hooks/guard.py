#!/usr/bin/env python3
"""Claude Code hooks for TriCast, wired in .claude/settings.json. TRICAST_HOOKS=off disables both
(export it before starting claude; hooks read the Claude Code process environment).

pre:  editing tests, golden data, a shared spec or the gate itself asks a human first (AGENTS.md §4).
post: an edited Python file must pass `ruff check` (AGENTS.md §6).
Only Edit/Write tool calls are covered; changes made through Bash are not.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unicodedata
from fnmatch import fnmatch
from pathlib import Path

PROTECTED = (
    "tests/*",
    "app/tests/*",
    "app/web/demo/webgpu/golden.json",
    "src/tricast/formats.py",
    "src/tricast/rounding.py",
    "src/tricast/quant/spec.py",
    "src/tricast/mma/spec.py",
    "src/tricast/reference/*",
    # the gate: narrowing it would narrow the definition of done (AGENTS.md §4.2)
    "Makefile",
    "pyproject.toml",
    ".github/workflows/*",
    ".claude/*",
)


ROOT = unicodedata.normalize("NFC", str(Path(__file__).resolve().parents[2]))


def fold(path: str) -> str:
    """APFS ignores case; elsewhere paths compare exactly."""
    return path.casefold() if sys.platform == "darwin" else path


def repo_path(tool_input: dict) -> str | None:
    """Repo-relative path, or None outside the repo. macOS keeps this repo's Hangul directory in NFD
    while tool inputs arrive in NFC, so both sides are normalized before comparing."""
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not raw:
        return None
    full = unicodedata.normalize("NFC", os.path.realpath(raw))
    if not fold(full).startswith(fold(ROOT) + os.sep):
        return None
    return full[len(ROOT) + 1:].replace(os.sep, "/")


def main(stage: str) -> int:
    if os.environ.get("TRICAST_HOOKS", "on").lower() in ("off", "0", "false"):
        return 0
    path = repo_path(json.load(sys.stdin).get("tool_input", {}))
    if path is None:
        return 0
    if stage == "pre":
        if any(fnmatch(fold(path), fold(pattern)) for pattern in PROTECTED):
            reason = f"{path}: 테스트·골든 데이터·공유 명세·게이트 변경은 사람이 승인한다 (AGENTS.md §4)."
            decision = {"hookEventName": "PreToolUse", "permissionDecision": "ask"}
            decision["permissionDecisionReason"] = reason
            print(json.dumps({"hookSpecificOutput": decision}, ensure_ascii=False))
        return 0
    if not path.endswith(".py") or not Path(ROOT, path).exists():
        return 0
    if shutil.which("ruff") is None:
        print("ruff not found on PATH; lint skipped", file=sys.stderr)
        return 1
    lint = subprocess.run(["ruff", "check", "--quiet", path], cwd=ROOT, capture_output=True, text=True)
    if lint.returncode:
        print(lint.stdout + lint.stderr, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
