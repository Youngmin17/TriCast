"""Command-line inspection, casting, recipe validation, and model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .formats import REGISTRY, get_format
from .mma.spec import PRESETS
from .quant.spec import SCHEMES
from .rounding import Rounding


def _table(headers: list[str], rows: list[list]) -> None:
    cells = [[str(cell) for cell in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for index, row in enumerate(cells):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))


def _calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calib-dataset", choices=("wikitext2", "c4", "pile"))
    parser.add_argument("--calib-samples", type=int)
    parser.add_argument("--calib-seqlen", type=int)
    parser.add_argument("--calib-seed", type=int)
    parser.add_argument("--sequential", action=argparse.BooleanOptionalAction, default=None)


def _calibration_overrides(args: argparse.Namespace) -> dict:
    values = {key: getattr(args, f"calib_{key}") for key in ("dataset", "samples", "seqlen", "seed")}
    values["sequential"] = args.sequential
    return {key: value for key, value in values.items() if value is not None}


def _output_directory(value: str | None) -> Path:
    return Path(value or f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}").resolve()


def _print_run(result: dict) -> int:
    status = result["status"]
    print(f"{result['recipe']['name']}: {status} ({result['wall_time_s']:.3f}s)")
    print(f"Result: {result['result_path']}")
    if status != "complete":
        print(f"Error: {result.get('error', 'evaluation did not complete')}")
        return 1
    metrics = result["metrics"]
    if "ppl" in metrics:
        ppl = metrics["ppl"]
        _table(["PPL", "NLL", "Tokens", "Windows"],
               [[ppl[key] for key in ("ppl", "nll", "n_tokens", "n_windows")]])
        print(f"Dataset fingerprint: {ppl['dataset_fingerprint']}")
    if "lm_eval" in metrics:
        from .eval.runner import _json_default

        evaluation = metrics["lm_eval"]
        _table(["Task", "Metrics"], [[task, json.dumps(scores, default=_json_default)]
                                     for task, scores in evaluation["results"].items()])
        fingerprints = evaluation.get("dataset_fingerprints", result["env"].get("dataset_fingerprints"))
        print(f"Dataset fingerprints: {json.dumps(fingerprints, default=_json_default)}")
    return 0


def _report(args: argparse.Namespace) -> int:
    import random

    import numpy as np
    import torch

    from .analysis import layer_report
    from .eval.envinfo import capture_env
    from .eval.ppl import _dataset_texts
    from .eval.runner import _load_model, _write_json
    from .recipe import load_recipe

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    env = capture_env(args.model, {"seed": 42, "dtype": args.dtype, "device": args.device})
    if env.get("model_kind") == "hf" and env.get("model_sha"):
        model, tokenizer = _load_model(args.model, args.dtype, args.device, revision=env["model_sha"])
    else:
        model, tokenizer = _load_model(args.model, args.dtype, args.device)
    actual_sha = getattr(model.config, "_commit_hash", None)
    if actual_sha and env.get("model_kind") != "local":
        env["model_sha"] = actual_sha
    recipe = load_recipe(args.recipe)
    overrides = _calibration_overrides(args)
    if overrides:
        data = recipe.to_dict()
        data["calibration"] = {**(recipe.calibration or {}), **overrides}
        recipe = load_recipe(data)
    if args.text:
        texts = [Path(args.text).read_text(encoding="utf-8")]
        source_fingerprint = hashlib.sha256(texts[0].encode("utf-8")).hexdigest()
        source = {"text": str(Path(args.text).resolve())}
    else:
        texts, source_fingerprint = _dataset_texts(args.dataset or "wikitext2", "test")
        source = {"dataset": args.dataset or "wikitext2", "split": "test"}
    ids = torch.as_tensor(tokenizer("\n\n".join(texts), return_tensors="pt")["input_ids"])
    result = layer_report(model, recipe, input_ids=ids, tokenizer=tokenizer,
                          samples=args.samples, seqlen=args.seqlen, device=args.device)
    env.update(recipe_hash=recipe.sha256, dataset_fingerprint=result["dataset_fingerprint"])
    output_dir = _output_directory(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    result.update(status="complete", recipe=recipe.to_dict(), env=env,
                  source={**source, "fingerprint": source_fingerprint}, result_path=str(path))
    _write_json(output_dir / "env.json", env)
    _write_json(path, result)
    markdown_path.write_text(result["markdown"], encoding="utf-8")
    print(result["markdown"])
    print(f"Result: {path}")
    print(f"Markdown: {markdown_path}")
    print(f"Dataset fingerprint: {result['dataset_fingerprint']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a command and return a shell-compatible exit status."""
    parser = argparse.ArgumentParser(prog="tricast")
    commands = parser.add_subparsers(dest="command", required=True)
    formats = commands.add_parser("formats", help="list number formats")
    formats.add_argument("--name")
    commands.add_parser("schemes", help="list quantization schemes")
    commands.add_parser("presets", help="list MMA presets")
    cast = commands.add_parser("cast", help="round values onto a format grid")
    cast.add_argument("--format", required=True)
    cast.add_argument("--rounding", default="rne", choices=[mode.value for mode in Rounding])
    cast.add_argument("--no-saturate", action="store_true")
    cast.add_argument("values", metavar="VALUES", nargs="+", type=float)
    recipe = commands.add_parser("recipe-check", help="validate a recipe path or name")
    recipe.add_argument("recipe")
    ppl = commands.add_parser("ppl", help="evaluate token perplexity")
    evaluate = commands.add_parser(
        "eval", help="evaluate lm-eval tasks",
        epilog="For the lm-eval CLI use: python -m tricast.eval.lmeval --model tricast --model_args ...",
    )
    report = commands.add_parser("report", help="compare layer and model quantization errors")
    for command in (ppl, evaluate, report):
        command.add_argument("--model", required=True)
        command.add_argument("--recipe", required=True)
        command.add_argument("--dtype", default="auto")
        command.add_argument("--device", help="execution device (default: cuda if available, otherwise cpu)")
        command.add_argument("--out", help="result directory (default: runs/<UTC>/)")
        _calibration_arguments(command)
    ppl.add_argument("--dataset", default="wikitext2", choices=("wikitext2", "c4", "pile"))
    ppl.add_argument("--seqlen", type=int, default=2048)
    ppl.add_argument("--max-windows", type=int)
    evaluate.add_argument("--tasks", required=True)
    evaluate.add_argument("--limit", type=float)
    evaluate.add_argument("--num-fewshot", type=int)
    evaluate.add_argument("--batch-size", type=int, default=8)
    report_source = report.add_mutually_exclusive_group()
    report_source.add_argument("--dataset", choices=("wikitext2",))
    report_source.add_argument("--text", metavar="FILE")
    report.add_argument("--samples", type=int, default=8)
    report.add_argument("--seqlen", type=int, default=128)
    run = commands.add_parser("run", help="run a YAML recipe/sweep configuration")
    run.add_argument("config")
    _calibration_arguments(run)
    agent = commands.add_parser("agent", help="parse a natural-language emulation request")
    agent.add_argument("query")
    agent.add_argument("--execute", action="store_true")
    agent.add_argument("--llm", choices=("auto", "anthropic", "offline"), default="auto")
    agent.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command in ("ppl", "eval", "report") and args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.command == "agent":
        from .agent.loop import run_agent

        report = run_agent(args.query, execute=args.execute, llm=args.llm)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            # which parser produced the plan; an automatic fallback to the offline parser is shown
            print(f"파서: {report.source}")
            print("파싱 결과:")
            print(json.dumps(report.request.to_dict() if report.request else None,
                             ensure_ascii=False, indent=2))
            for title, items in (("가정", report.assumptions), ("질문", report.questions),
                                 ("오류", report.errors)):
                if items:
                    print(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
            if report.plan:
                _table(["Recipe", "Tasks"], [[recipe["name"], ", ".join(report.plan["tasks"])]
                                             for recipe in report.plan["recipes"]])
                if report.cost and report.cost.get("estimated_seconds") is not None:
                    print("비용 (추정): " + json.dumps(report.cost, ensure_ascii=False))
                else:
                    print("비용: 추정 불가 (필요한 추정 근거가 없습니다.)")
            if report.results:
                _table(["Recipe", "Metrics", "Environment"], [
                    [result["recipe"]["name"], json.dumps(result["metrics"], ensure_ascii=False),
                     json.dumps(result.get("env"), ensure_ascii=False)] for result in report.results
                ])
            print(report.summary)
        return 1 if report.errors else 0
    elif args.command == "formats":
        formats = [get_format(args.name)] if args.name else list(REGISTRY.values())
        _table(["Name", "Kind", "Max", "Min normal"],
               [[fmt.name, fmt.kind, fmt.max_normal, fmt.min_normal] for fmt in formats])
    elif args.command == "schemes":
        _table(["Scheme", "Format", "Granularity", "Scale"],
               [[name, spec.format.name, spec.granularity, spec.scale.method if spec.scale else "none"]
                for name, spec in SCHEMES.items()])
    elif args.command == "presets":
        _table(["Preset", "Algorithm", "F", "Chunk", "G"],
               [[name, spec.algorithm, spec.f_bits, spec.chunk_size, spec.g_bits]
                for name, spec in PRESETS.items()])
    elif args.command == "cast":
        import torch

        from .reference.cast import round_to_format

        result = round_to_format(torch.tensor(args.values, dtype=torch.float64), args.format,
                                 args.rounding, saturate=not args.no_saturate)
        _table(["Input", "Output"], [[x, y] for x, y in zip(args.values, result.tolist(), strict=True)])
    elif args.command == "recipe-check":
        from .recipe import load_recipe

        recipe = load_recipe(args.recipe)
        _table(["Recipe", "SHA256", "Calibration"], [[recipe.name, recipe.sha256, recipe.needs_calibration]])
    elif args.command == "run":
        from .eval.runner import run_config

        results = run_config(args.config, calibration=_calibration_overrides(args))
        return max((_print_run(result) for result in results), default=0)
    elif args.command == "report":
        return _report(args)
    else:
        from .eval.runner import run_config

        if args.command == "ppl":
            tasks = {"ppl": {"dataset": args.dataset, "seqlen": args.seqlen,
                             "max_windows": args.max_windows}}
        else:
            tasks = {"lm_eval": {"tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
                                 "limit": args.limit, "num_fewshot": args.num_fewshot,
                                 "batch_size": args.batch_size}}
        results = run_config({"model": args.model, "dtype": args.dtype, "device": args.device,
                              "recipes": [args.recipe], "tasks": tasks,
                              "output_dir": str(_output_directory(args.out)),
                              "calibration": _calibration_overrides(args)})
        return max((_print_run(result) for result in results), default=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
