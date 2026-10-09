"""CPU-only lm-eval task capture validation and ordered dataset/cache audit.

Loads the complete cached validation split, not a model. An optional historical
run link is an AFTER-RUN cache audit, never in-process capture or quality proof.
No original run evidence is edited and no checkpoint or dataset is downloaded.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from lm_eval import evaluator
from lm_eval.api.group import Group
from lm_eval.tasks import TaskManager

from tricast.eval.envinfo import capture_env
from tricast.eval.lmeval import _dataset_fingerprints, _DatasetTaskManager


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(value: dict, dotted: str) -> Any:
    for key in dotted.split("."):
        value = value[key]
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_checked(task_list: Any) -> tuple[dict, _DatasetTaskManager]:
    manager = _DatasetTaskManager()
    # Observe the actual base return, not a reconstructed stand-in TaskDict.
    original = TaskManager.load
    returned: list[dict] = []

    def observed(*args: Any, **kwargs: Any) -> dict:
        value = original(*args, **kwargs)
        returned.append(value)
        return value

    with patch.object(TaskManager, "load", observed):
        loaded = manager.load(task_list)
    require(len(returned) == 1 and loaded is returned[0], "manager changed the base TaskDict return")
    require(set(manager.loaded) == set(loaded["tasks"]), "captured leaf task names differ")
    require(all(manager.loaded[name] is task for name, task in loaded["tasks"].items()),
            "manager changed returned task identities")
    return loaded, manager


def audit(args: argparse.Namespace, env: dict) -> dict:
    preflight = json.loads(args.preflight.read_text())
    expected_count = field(preflight, args.preflight_count_key)
    expected_fingerprint = field(preflight, args.preflight_fingerprint_key)
    require(type(expected_count) is int and expected_count == args.expected_count,
            "preflight count differs from explicit full-split count")
    require(isinstance(expected_fingerprint, str) and bool(expected_fingerprint),
            "preflight has no expected fingerprint")
    if args.expected_fingerprint is not None:
        require(args.expected_fingerprint == expected_fingerprint,
                "explicit fingerprint differs from preflight")

    # Reproduce the old capture bug without changing the installed source.
    with patch.object(_DatasetTaskManager, "load", TaskManager.load):
        legacy = _DatasetTaskManager()
        legacy_result = legacy.load([args.task])
        require(args.task in legacy_result["tasks"] and not legacy.loaded,
                "legacy named-task capture negative control did not reproduce")

    loaded, manager = load_checked([args.task])
    require(set(loaded["tasks"]) == {args.task}, "named task resolved to unexpected leaves")
    task = loaded["tasks"][args.task]
    fingerprints = _dataset_fingerprints(manager.loaded)
    require(fingerprints[args.task][args.split] == expected_fingerprint,
            "captured validation fingerprint differs from preflight")
    checks = [{"name": "legacy_named_capture_missing", "passed": True},
              {"name": "named_task_capture_and_return_identity", "passed": True}]
    # Top-level task dicts are complete inline configs, not registered aliases.
    # asdict preserves all TaskConfig fields and callable preprocessing functions.
    inline_config = asdict(task.config)
    require(inline_config["task"] == args.task and bool(inline_config["dataset_path"]),
            "named task has no complete inline configuration")
    subgroup, group = Group("tricast_metadata_leaf"), Group("tricast_metadata_probe")
    subgroup.add(task)
    group.add(subgroup)
    for label, spec in (
        ("inline_task", [inline_config]),
        ("nested_group_leaves", [group]),
        ("prebuilt_task", [task]),
    ):
        result, captured = load_checked(spec)
        require(set(result["tasks"]) == {args.task}, f"{label}: unexpected leaf tasks")
        require(_dataset_fingerprints(captured.loaded) == fingerprints,
                f"{label}: split fingerprints changed")
        if label == "prebuilt_task":
            require(result["tasks"][args.task] is task, "prebuilt task identity changed")
        if label == "nested_group_leaves":
            require(result["groups"] == {group.name: group, subgroup.name: subgroup}
                    and result["group_map"] == {group.name: [subgroup.name], subgroup.name: [args.task]},
                    "nested group structure changed")
        checks.append({"name": label, "passed": True})

    dataset = task.dataset[args.split]
    require(len(dataset) == args.expected_count, "complete validation split count mismatch")
    require(dataset._fingerprint == expected_fingerprint, "actual dataset fingerprint mismatch")
    content = hashlib.sha256()
    count = 0
    for row in dataset:
        line = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8") + b"\n"
        content.update(line)
        count += 1
    require(count == args.expected_count, "canonical audit omitted validation rows")
    cache_files = []
    for item in dataset.cache_files:
        path = Path(item["filename"]).resolve()
        require(path.is_file() and path.suffix == ".arrow", f"missing Arrow cache file: {path}")
        before = path.stat()
        digest = file_hash(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Arrow cache changed while hashing: {path}")
        cache_files.append({"path": str(path), "bytes": after.st_size,
                            "mtime_ns": after.st_mtime_ns, "sha256": digest})
    require(bool(cache_files), "validation has no auditable on-disk Arrow cache")

    result = {"status": "passed", "scope": __doc__, "checks": checks,
              "dataset": {"task": args.task, "split": args.split,
                          "count": count, "fingerprints": fingerprints[args.task],
                          "canonical_rows_sha256": content.hexdigest(),
                          "canonical_encoding": "ordered UTF-8 JSONL; sorted keys; compact; no NaN",
                          "arrow_cache_files": cache_files},
              "historical_capture_modified": False, "full_quality_claim": False,
              "gpu_or_kernel_proof": False}
    result["task_config"] = {key: task.get_config(key) for key in (
        "task", "dataset_path", "dataset_name", "output_type", "training_split",
        "validation_split", "test_split", "num_fewshot")}
    # Keep executable task/source identity distinct from the data-content hash.
    sources = {}
    for name, function in (("TaskManager.load", TaskManager.load),
                           ("simple_evaluate", evaluator.simple_evaluate),
                           ("capture_manager", _DatasetTaskManager.load)):
        path = Path(inspect.getsourcefile(function)).resolve()
        sources[name] = {"path": str(path), "sha256": file_hash(path)}
    for key in ("doc_to_text", "doc_to_target", "doc_to_choice", "process_docs"):
        function = task.get_config(key)
        if callable(function):
            path = Path(inspect.getsourcefile(function)).resolve()
            sources[key] = {"path": str(path), "sha256": file_hash(path)}
    yaml_path = Path(manager.task_index[args.task].yaml_path).resolve()
    sources["task_yaml"] = {"path": str(yaml_path), "sha256": file_hash(yaml_path)}
    env["implementation_sources"] = sources
    env["preflight_sha256"] = file_hash(args.preflight)
    if args.run_env is not None:
        run_env = json.loads(args.run_env.read_text())
        run_result = json.loads(args.run_result.read_text())
        require(run_result.get("status") not in (None, "running"), "linked run is not completed")
        started = datetime.fromisoformat(run_env["utc"])
        require(started.tzinfo is not None, "linked run has no timezone-aware start time")
        for item in cache_files:
            item["mtime_before_linked_run_start"] = item["mtime_ns"] <= int(started.timestamp() * 1e9)
        result["historical_run_link"] = {
            "audit_timing": "after-run; cache state observed now, NOT captured during model forwards",
            "env_path": str(args.run_env.resolve()), "env_sha256": file_hash(args.run_env),
            "result_path": str(args.run_result.resolve()), "result_sha256": file_hash(args.run_result),
            "run_source_sha256": run_env.get("src_sha256"),
            "run_harness_sha256": run_env.get("harness_sha256"),
            "run_status": run_result.get("status"),
            "raw_dataset_fingerprints": {
                name: record.get("metrics", {}).get("lm_eval", {}).get("dataset_fingerprints")
                for name, record in run_result.get("runs", {}).items()},
            "limitation": ("preflight/fingerprints/cache mtimes support linkage; "
                           "no historical in-memory content attestation"),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="winogrande")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--expected-count", type=int, default=1267)
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-count-key", default="winogrande_validation")
    parser.add_argument("--preflight-fingerprint-key", default="winogrande_fingerprints.validation")
    parser.add_argument("--run-env", type=Path, help="read-only link to an existing historical run")
    parser.add_argument("--run-result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        parser.error("CPU-only audit requires CUDA_VISIBLE_DEVICES='' or -1")
    if (args.run_env is None) != (args.run_result is None):
        parser.error("--run-env and --run-result must be supplied together")
    if args.expected_count < 1:
        parser.error("expected-count must be the positive full-split count")
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    env = capture_env(extra={"command": [sys.executable, *sys.argv], "scope": __doc__,
        "audit_utc": datetime.now(timezone.utc).isoformat(), "harness_sha256": file_hash(Path(__file__)),
        "offline": True, "full_quality_claim": False, "gpu_or_kernel_proof": False})
    result = {"status": "running", "scope": __doc__}
    with (args.out / "stdout.log").open("w") as stdout, (args.out / "stderr.log").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                result = audit(args, env)
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            write_json(args.out / "env.json", env)
            write_json(args.out / "result.json", result)
            print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
