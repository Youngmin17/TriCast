"""Live Quantizer job (``POST /api/quantizer/jobs``): one recipe on Qwen3-0.6B with tricast's own functions.

The records are written the way the CLI writes them, so the exporters read them unchanged:

- ``<out_dir>/report/<name>/report.json`` (+ ``report.md``, ``env.json``): ``tricast report``'s record, by
  default on WikiText-2 train, 8 windows × 512 tokens (the recorded demo reports);
- ``<out_dir>/eval/<name>.json`` (+ ``env.json``): a ``tricast run`` record with WikiText-2 test PPL
  (2048-token windows) and/or lm-eval, recipe, calibration and environment.

The job result is ``{"metrics": {"run", "report", "lmeval"}, "env", "evidence"}``, where ``metrics`` uses the
shapes app/export_quantizer_demo.py writes for recorded presets. torch and tricast are imported only when a
job runs; this module's top level is stdlib, so the server can import it in any mode.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-0.6B"
TASKS = ("ppl", "report", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag")
LM_EVAL_TASKS = TASKS[2:]

Progress = Callable[..., None]
# (model_id, dtype, device, revision) -> (model, tokenizer); tests inject a tiny model here.
ModelFactory = Callable[[str, str, str, str | None], tuple[Any, Any]]


@dataclass(frozen=True)
class QuantizerArgs:
    """One live job. The window settings default to the recorded demo runs'; tests shrink them."""

    name: str
    recipe: dict
    tasks: tuple[str, ...]
    limit: int | None
    device: str
    out_dir: Path
    model_id: str = MODEL_ID
    ppl_seqlen: int = 2048
    report_samples: int = 8
    report_seqlen: int = 512
    lm_batch_size: int = 16
    text: str | None = None  # one text in place of WikiText-2 for PPL and report (tests)
    calibration: dict | None = None  # tricast's --calib-* / --sequential settings, over the recipe's own


def _load(model_id: str, dtype: str, device: str, revision: str | None) -> tuple[Any, Any]:
    from tricast.eval.runner import _load_model  # what tricast run / tricast report load with

    return _load_model(model_id, dtype, device, revision=revision)


def run_quantizer(args: QuantizerArgs, progress: Progress, *,
                  model_factory: ModelFactory | None = None) -> dict:
    import random

    import numpy as np
    import torch

    from app import export_quantizer_demo as views
    from tricast import calibrate, load_recipe, patch_model
    from tricast.analysis import layer_report
    from tricast.eval import capture_env, perplexity
    from tricast.eval.ppl import _dataset_texts
    from tricast.eval.runner import _write_json
    from tricast.nn.patch import unpatch_model

    def seed() -> None:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    dtype = "bfloat16" if args.device == "cuda" else "float32"
    lm_tasks = [task for task in LM_EVAL_TASKS if task in args.tasks]
    seed()
    progress("모델 불러오기", 0.02)
    env = capture_env(args.model_id, {"seed": 42, "dtype": dtype, "device": args.device})
    revision = env.get("model_sha") if env.get("model_kind") == "hf" else None
    model, tokenizer = (model_factory or _load)(args.model_id, dtype, args.device, revision)
    recipe = load_recipe(args.recipe)
    if args.calibration:  # as tricast run / report apply --calib-* (tricast.cli._calibration_overrides)
        recipe = load_recipe({**recipe.to_dict(),
                              "calibration": {**(recipe.calibration or {}), **args.calibration}})
    env["recipe_hash"] = recipe.sha256
    if args.text is not None:
        texts, fingerprint = [args.text], hashlib.sha256(args.text.encode("utf-8")).hexdigest()
    results: dict[str, Any] = {"run": None, "report": None, "lmeval": None}
    files: list[str] = []

    if "report" in args.tasks:
        progress("레이어별 리포트", 0.08)
        seed()
        if args.text is None:
            texts, fingerprint = _dataset_texts("wikitext2", "train")
        source = {"dataset": "wikitext2", "split": "train"} if args.text is None else {"text": "inline"}
        ids = torch.as_tensor(tokenizer("\n\n".join(texts), return_tensors="pt")["input_ids"])
        report = layer_report(model, recipe, input_ids=ids, tokenizer=tokenizer, samples=args.report_samples,
                              seqlen=args.report_seqlen, device=args.device)
        report_env = {**env, "dataset_fingerprint": report["dataset_fingerprint"]}
        folder = args.out_dir / "report" / args.name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "report.json"
        report.update(status="complete", recipe=recipe.to_dict(), env=report_env,
                      source={**source, "fingerprint": fingerprint}, result_path=str(path))
        _write_json(folder / "env.json", report_env)
        _write_json(path, report)
        (folder / "report.md").write_text(report["markdown"], encoding="utf-8")
        files.append(str(path))
        saved = json.loads(path.read_text(encoding="utf-8"))
        results["report"] = views.report_view(path, saved, args.model_id)

    if "ppl" in args.tasks or lm_tasks:
        seed()
        folder = args.out_dir / "eval"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{args.name}.json"
        run_env = {**env, "calibration_config": recipe.calibration_options, "dataset_fingerprints": {}}
        record: dict[str, Any] = {"status": "failed", "recipe_hash": recipe.sha256,
                                  "recipe": recipe.to_dict(), "metrics": {}, "env": run_env,
                                  "calibration": None, "calibration_config": recipe.calibration_options,
                                  "patch_report": {}, "result_path": str(path)}
        started = time.perf_counter()
        try:
            progress("패치", 0.4)
            record["patch_report"] = asdict(patch_model(model, recipe))
            if recipe.needs_calibration:
                progress("보정", 0.45)
                record["calibration"] = calibrate(model, recipe, tokenizer=tokenizer, device=args.device)
                fingerprints = run_env["dataset_fingerprints"]
                fingerprints["calibration"] = record["calibration"].get("dataset_fingerprint")
            model.eval()
            if "ppl" in args.tasks:
                progress("WikiText-2 PPL", 0.5)
                record["metrics"]["ppl"] = perplexity(model, tokenizer, device=args.device,
                                                      seqlen=args.ppl_seqlen,
                                                      texts=texts if args.text is not None else None)
                run_env["dataset_fingerprints"]["ppl"] = record["metrics"]["ppl"]["dataset_fingerprint"]
            if lm_tasks:
                from tricast.eval.lmeval import evaluate

                progress("lm-eval", 0.65)
                record["metrics"]["lm_eval"] = evaluate(model, tokenizer, lm_tasks, limit=args.limit,
                                                        batch_size=args.lm_batch_size)
                lm_fingerprints = record["metrics"]["lm_eval"].get("dataset_fingerprints")
                run_env["dataset_fingerprints"]["lm_eval"] = lm_fingerprints
            record["status"] = "complete"
        finally:
            unpatch_model(model)
            record["wall_time_s"] = time.perf_counter() - started
            _write_json(folder / "env.json", run_env)
            _write_json(path, record)
        files.append(str(path))
        saved = json.loads(path.read_text(encoding="utf-8"))
        if "ppl" in args.tasks:
            results["run"] = views.ppl_view(path, saved, args.model_id)
        if lm_tasks:
            results["lmeval"] = views.lmeval_view(path, saved, lm_tasks, args.model_id)

    progress("정리", 0.98)
    report_dir = args.out_dir / "report"
    evidence = {"out_dir": str(args.out_dir), "files": files,
                "layer_map_command": (f"python -m app.export_layers {report_dir} --out app/web/demo/layers"
                                      if results["report"] else None)}
    return {"metrics": results, "env": env, "evidence": evidence}
