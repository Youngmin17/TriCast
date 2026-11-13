"""The tuning app's benchmarks beyond WikiText-2 (scripts/quantizer/tune_bench.py): the recipe files it
measures are the page's own, per-document lm-eval results averaged back to lm-eval's score, and the export
comparing each Auto-tune size with the best one-method setup within it on every benchmark (paired difference,
2-SE verdict). CPU only, no torch, no lm-eval, no GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from app.server import DEFAULT_DEMO_DIR

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tune_bench", ROOT / "scripts" / "quantizer" / "tune_bench.py")
assert SPEC is not None and SPEC.loader is not None
tb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tb)
ACC = [spec for spec in tb.BENCHMARKS if spec["kind"] == "acc"]


@pytest.mark.parametrize("bundle", ["bundle.json", "qwen3-8b/bundle.json"])
def test_recipes_are_the_files_the_bundle_measured(bundle: str, tmp_path: Path) -> None:
    """Every setup's file comes from the page's builder and is byte for byte the measured one (a recorded
    benchmark joins the page's PPL through this SHA-256)."""
    data = json.loads((DEFAULT_DEMO_DIR / "autotune" / bundle).read_text(encoding="utf-8"))
    tb.recipes(DEFAULT_DEMO_DIR / "autotune" / bundle, tmp_path)
    rows = data["baselines"] + data["solutions"]
    assert sorted(p.stem for p in tmp_path.glob("*.yaml")) == sorted(row["id"] for row in rows)
    for row in rows:
        assert (
            hashlib.sha256((tmp_path / f"{row['id']}.yaml").read_bytes()).hexdigest()
            == row["recipe_file_sha256"]
        )


def test_recipes_refuses_a_file_the_bundle_did_not_measure(tmp_path: Path) -> None:
    data = json.loads(
        (DEFAULT_DEMO_DIR / "autotune" / "qwen3-8b" / "bundle.json").read_text(encoding="utf-8")
    )
    data["baselines"][0]["recipe_file_sha256"] = "0" * 64
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="the bundle measured 000000000000"):
        tb.recipes(path, tmp_path / "out")


def test_per_document_results_follow_lm_eval_groups_and_filters() -> None:
    """A group's documents are its leaf subtasks in name order, each by doc_id; a filter keeps its own rows; a
    non-0/1 result is refused."""
    result = {
        "group_subtasks": {
            "mmlu": ["mmlu_stem", "mmlu_other"],
            "mmlu_stem": ["mmlu_b", "mmlu_a"],
            "mmlu_other": ["mmlu_c"],
        }
    }
    assert tb._leaves(result, "mmlu") == ["mmlu_a", "mmlu_b", "mmlu_c"]
    assert tb._leaves(result, "piqa") == ["piqa"]
    samples = {
        "mmlu_a": [{"doc_id": 1, "acc": 0.0}, {"doc_id": 0, "acc": 1.0}],
        "mmlu_b": [{"doc_id": 0, "acc": 1.0}],
        "mmlu_c": [{"doc_id": 0, "acc": 0.0}],
        "gsm": [{"doc_id": 0, "filter": "strict", "em": 1.0}, {"doc_id": 0, "filter": "flex", "em": 0.0}],
    }
    assert tb._docs(samples, ["mmlu_a", "mmlu_b", "mmlu_c"], "acc", "none") == "1010"
    assert tb._docs(samples, ["gsm"], "em", "strict") == "1"
    assert tb._docs(samples, ["gsm"], "em", "flex") == "0"
    with pytest.raises(ValueError, match="not 0 or 1"):
        tb._docs({"t": [{"doc_id": 0, "acc": 0.5}]}, ["t"], "acc", "none")


def test_paired_difference_and_its_standard_error() -> None:
    a, b = "1101100101", "1001000111"
    d = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    mean = sum(d) / len(d)
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (len(d) - 1))
    diff, se = tb.paired(a, b)
    assert diff == pytest.approx(mean) and se == pytest.approx(sd / math.sqrt(len(d)))
    assert tb.paired("111", "111") == (0.0, 0.0)
    with pytest.raises(ValueError, match="same documents"):
        tb.paired("10", "101")


def test_a_tasks_standard_error_is_lm_evals_mean_stderr_of_its_documents() -> None:
    docs = "1101100101" * 10
    mean = docs.count("1") / len(docs)
    sd = math.sqrt(sum((int(x) - mean) ** 2 for x in docs) / (len(docs) - 1))
    assert tb.mean_stderr(docs) == pytest.approx(sd / math.sqrt(len(docs)))
    assert tb.mean_stderr("1") == 0.0


def test_verdict_is_two_standard_errors_and_only_for_accuracy() -> None:
    acc, ppl = {"kind": "acc"}, {"kind": "ppl"}
    assert tb.verdict(acc, 0.021, 0.01) == "better"
    assert tb.verdict(acc, 0.019, 0.01) == "same"
    assert tb.verdict(acc, -0.021, 0.01) == "worse"
    assert tb.verdict(ppl, -1.0, None) is None


def _docs(n: int, ones: int) -> str:
    return "1" * ones + "0" * (n - ones)


def _bench(
    path: Path,
    name: str,
    sha: str | None,
    scores: dict[str, float],
    *,
    loaded: str = "rev",
    limit: int | None = None,
    n: int = 100,
    docs: dict[str, str] | None = None,
) -> None:
    """A record with ``scores`` per benchmark; accuracy documents are ``ones`` leading 1s out of ``n`` unless
    ``docs`` gives them."""
    results = {}
    for spec in tb.BENCHMARKS:
        value = scores[spec["id"]]
        if spec["kind"] == "ppl":
            results[spec["id"]] = {
                "value": value,
                "n_tokens": 1000,
                "n_windows": 2,
                "dataset_fingerprint": "f",
            }
        else:
            text = (docs or {}).get(spec["id"]) or _docs(n, round(value * n))
            results[spec["id"]] = {
                "metric": spec["metric"],
                "filter": "none",
                "value": text.count("1") / len(text),
                "stderr": 0.05,
                "n": len(text),
                "num_fewshot": spec["num_fewshot"],
                "versions": {spec["id"]: 1},
                "fingerprints": {spec["id"]: "x"},
                "docs": text,
            }
    path.write_text(
        json.dumps(
            {
                "schema": tb.SCHEMA,
                "model": "Qwen/Qwen3-8B",
                "revision": "rev",
                "name": name,
                "limit": limit,
                "file": None if sha is None else f"{name}.yaml",
                "file_sha256": sha,
                "recipe_hash": "h",
                "patched_layers": 0 if sha is None else 252,
                "lm_eval_version": "0.4.13",
                "lm_eval": {
                    "0": {"batch_size": {"requested": 32, "used": 32, "reason": None}, "limit": limit},
                    "5": {"batch_size": {"requested": 32, "used": 32, "reason": None}, "limit": limit},
                },
                "results": results,
                "seconds": {"total": 1.0},
                "env": {
                    "gpu_names": ["NVIDIA H200 NVL"],
                    "git_sha": "abc",
                    "git_dirty": False,
                    "model_sha": loaded,
                },
            }
        )
    )


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    """A bundle with two one-method setups (3 and 4 bits) and one Auto-tune size (budget 4 bits), and records:
    the 4-bit uniform wins every accuracy task but ARC-Easy (the 3-bit one does), Auto-tune beats both."""
    bundle = {
        "table": {"meta": {"model_id": "Qwen/Qwen3-8B"}},
        "baselines": [
            {"id": "u3", "method": "m3", "bits": 3.0, "recipe_file_sha256": "a" * 64},
            {"id": "u4", "method": "m4", "bits": 4.0, "recipe_file_sha256": "b" * 64},
            {"id": "u8", "method": "m8", "bits": 8.0, "recipe_file_sha256": "d" * 64},
        ],
        "solutions": [{"id": "auto_4", "bits": 3.99, "budget_bits": "4", "recipe_file_sha256": "c" * 64}],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    bench = tmp_path / "bench"
    bench.mkdir()
    base = {spec["id"]: (10.0 if spec["kind"] == "ppl" else 0.5) for spec in tb.BENCHMARKS}
    _bench(bench / "native.json", "native", None, {**base, "wikitext2": 9.0, "c4": 12.0})
    _bench(bench / "u3.json", "u3", "a" * 64, {**base, "wikitext2": 11.0, "c4": 14.0, "arc_easy": 0.6})
    _bench(
        bench / "u4.json",
        "u4",
        "b" * 64,
        {
            **base,
            "wikitext2": 10.0,
            "c4": 13.0,
            "arc_easy": 0.4,
            **{s["id"]: 0.55 for s in ACC if s["id"] != "arc_easy"},
        },
    )
    _bench(
        bench / "u8.json",
        "u8",
        "d" * 64,
        {**base, "wikitext2": 9.1, "c4": 12.1, **{s["id"]: 0.9 for s in ACC}},
    )
    _bench(
        bench / "auto_4.json",
        "auto_4",
        "c" * 64,
        {**base, "wikitext2": 9.5, "c4": 12.5, **{s["id"]: 0.7 for s in ACC}},
    )
    return path, bench


def test_export_compares_each_size_with_the_best_one_method_setup_per_benchmark(tmp_path: Path) -> None:
    bundle, bench = _setup(tmp_path)
    out = tmp_path / "benchmarks.json"
    tb.export(bench, bundle, out)
    data = json.loads(out.read_text())
    assert [b["id"] for b in data["benchmarks"]] == [spec["id"] for spec in tb.BENCHMARKS]
    assert set(data["setups"]) == {"a" * 64, "b" * 64, "c" * 64, "d" * 64}
    assert data["native"]["values"]["wikitext2"] == 9.0
    compare = data["compare"]["c" * 64]
    assert compare["uniforms"] == ["u3", "u4"]  # u8 (8 bits) is over the 4-bit budget
    rows = {row["bench"]: row for row in compare["rows"]}
    assert (
        rows["arc_easy"]["best"] == "u3" and rows["arc_easy"]["theirs"] == 0.6
    )  # per benchmark, the best one
    assert rows["piqa"]["best"] == "u4" and rows["piqa"]["theirs"] == 0.55
    assert (
        rows["wikitext2"]["best"] == "u4" and rows["wikitext2"]["verdict"] is None
    )  # PPL: no per-doc results
    assert rows["wikitext2"]["diff"] == pytest.approx(-0.5)
    diff, se = tb.paired(_docs(100, 70), _docs(100, 55))
    assert rows["piqa"]["diff"] == pytest.approx(diff) and rows["piqa"]["se"] == pytest.approx(se)
    assert rows["piqa"]["verdict"] == "better"
    macro = compare["macro"]  # one setup for the average: u4 averages higher than u3
    assert macro["best"] == "u4" and macro["tasks"] == len(ACC)
    assert macro["ours"] == pytest.approx(0.7) and macro["verdict"] == "better"


def test_export_without_a_clear_gap_says_same(tmp_path: Path) -> None:
    bundle, bench = _setup(tmp_path)
    # Auto-tune and u4 differ on 2 of 100 PIQA documents in opposite directions: no difference
    docs = {"piqa": "10" + _docs(98, 50)}
    _bench(
        bench / "auto_4.json",
        "auto_4",
        "c" * 64,
        {s["id"]: (9.5 if s["kind"] == "ppl" else 0.55) for s in tb.BENCHMARKS},
        docs=docs,
    )
    _bench(
        bench / "u4.json",
        "u4",
        "b" * 64,
        {s["id"]: (10.0 if s["kind"] == "ppl" else 0.55) for s in tb.BENCHMARKS},
        docs={"piqa": "01" + _docs(98, 50)},
    )
    tb.export(bench, bundle, tmp_path / "out.json")
    rows = {
        row["bench"]: row
        for row in json.loads((tmp_path / "out.json").read_text())["compare"]["c" * 64]["rows"]
    }
    assert rows["piqa"]["diff"] == 0.0 and rows["piqa"]["verdict"] == "same"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"limit": 50}, "a --limit record"),
        ({"loaded": "other"}, "not of Qwen/Qwen3-8B@rev"),
        ({"n": 90}, "differs from the native record"),
    ],
)
def test_export_refuses_records_that_do_not_compare(tmp_path: Path, change: dict, message: str) -> None:
    bundle, bench = _setup(tmp_path)
    scores = {s["id"]: (10.0 if s["kind"] == "ppl" else 0.5) for s in tb.BENCHMARKS}
    _bench(bench / "u4.json", "u4", "b" * 64, scores, **change)
    with pytest.raises(SystemExit, match=message):
        tb.export(bench, bundle, tmp_path / "out.json")


def test_export_refuses_a_record_of_no_page_setup_and_a_missing_native(tmp_path: Path) -> None:
    bundle, bench = _setup(tmp_path)
    _bench(bench / "stray.json", "stray", "e" * 64, {s["id"]: 0.5 for s in tb.BENCHMARKS})
    with pytest.raises(SystemExit, match="is no setup of"):
        tb.export(bench, bundle, tmp_path / "out.json")
    (bench / "stray.json").unlink()
    (bench / "native.json").unlink()
    with pytest.raises(SystemExit, match="no full native.json"):
        tb.export(bench, bundle, tmp_path / "out.json")


def test_shipped_benchmark_files_belong_to_their_bundles() -> None:
    """Every benchmarks file of the model list: its setups are setups of the model's bundle (recipe-file
    SHA-256), each compared size is an Auto-tune size of the bundle against one-method setups within its
    budget, and the numbers on the page are the setups' own."""
    folder = DEFAULT_DEMO_DIR / "autotune"
    entries = json.loads((folder / "models.json").read_text(encoding="utf-8"))["models"]
    for entry in (e for e in entries if e.get("benchmarks")):
        data = json.loads((folder / entry["benchmarks"]).read_text(encoding="utf-8"))
        bundle = json.loads((folder / entry["bundle"]).read_text(encoding="utf-8"))
        assert (
            data["schema"] == tb.SCHEMA
            and data["model"] == entry["id"] == bundle["table"]["meta"]["model_id"]
        )
        assert [b["id"] for b in data["benchmarks"]] == [spec["id"] for spec in tb.BENCHMARKS]
        rows = {row["recipe_file_sha256"]: row for row in bundle["baselines"] + bundle["solutions"]}
        assert set(data["setups"]) <= set(rows)
        for sha, setup in data["setups"].items():
            assert (
                setup["id"] == rows[sha]["id"]
                and setup["env"]["model_sha"] == data["native"]["env"]["model_sha"]
            )
        assert data["compare"], entry["id"]
        for sha, cmp in data["compare"].items():
            solution = rows[sha]
            assert cmp["id"] == solution["id"] and cmp["budget_bits"] == solution["budget_bits"]
            limit = tb._fraction(solution["budget_bits"]) + 1e-9
            ids = {row["id"]: row for row in bundle["baselines"]}
            assert cmp["uniforms"] and all(ids[u]["bits"] <= limit for u in cmp["uniforms"])
            by_id = {setup["id"]: setup for setup in data["setups"].values()}
            for row in cmp["rows"]:
                assert row["best"] in cmp["uniforms"]
                assert row["ours"] == data["setups"][sha]["values"][row["bench"]]
                assert row["theirs"] == by_id[row["best"]]["values"][row["bench"]]
                spec = next(s for s in tb.BENCHMARKS if s["id"] == row["bench"])
                assert row["verdict"] == tb.verdict(spec, row["diff"], row["se"])
