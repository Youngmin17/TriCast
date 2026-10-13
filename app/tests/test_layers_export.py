"""app/export_layers.py, the layer map's exporter, on SYNTHETIC tricast report.json records
(app/tests/quantizer_records.py: 196 decoder Linears with made-up metrics, exact "inf" activations for a
weight-only recipe, "nan" / "-inf" tokens): every metric round-trips exactly with its type, non-finite values
are flagged not dropped, rows follow block order, the export is deterministic, bad reports are refused. The
committed bundle (app/web/demo/layers, exported from the GPU reports) is checked for internal consistency and
its report order.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from app import export_layers
from app.export_layers import (
    COLUMNS,
    KIND_ORDER,
    SCHEMA,
    ExportError,
    block_and_kind,
    layer_file,
    load_report,
    main,
)
from app.tests import quantizer_records as rec

REPO = Path(export_layers.__file__).resolve().parents[1]
BUNDLE = REPO / "app" / "web" / "demo" / "layers"
LABELS = [
    ("syn_int8", "INT8 (SYNTHETIC)"),
    ("syn_w4a16", "W4A16 (SYNTHETIC)"),
    ("syn_nvint4", "NVINT4 (SYNTHETIC)"),
]
KINDS = {"syn_int8": "int8", "syn_w4a16": "w4a16", "syn_nvint4": "nvint4"}
LABEL_ARGS = [arg for recipe, text in LABELS for arg in ("--label", f"{recipe}={text}")]
# The committed layer map: the 8 W1 reports and the Quantizer demo's NVINT4 preset after NVFP4.
COMMITTED = [
    ("int8_row_w8a8", "INT8 W8A8"),
    ("mxfp8_w_a", "MXFP8"),
    ("w4a16_g128_zp_gptq", "GPTQ INT4 (W4)"),
    ("nvfp4_awq_shared", "AWQ + NVFP4"),
    ("nvfp4_smoothquant", "SmoothQuant + NVFP4"),
    ("nvfp4_w_a", "NVFP4"),
    ("qd_nvint4", "NVINT4"),
    ("mxfp4_w_a", "MXFP4"),
    ("msfp12_bfp", "BFP (MSFP12)"),
]


def original_report(recipe: str) -> dict:
    return rec.layer_report(recipe, KINDS[recipe])


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("reports")
    for recipe, _ in LABELS:
        rec.write(root / recipe / "report.json", original_report(recipe))
    return root


def export(out: Path, inputs: list[str]) -> dict[str, bytes]:
    assert main([*inputs, "--out", str(out), *LABEL_ARGS]) == 0
    return {path.name: path.read_bytes() for path in sorted(out.iterdir())}


@pytest.fixture(scope="module")
def exported(reports: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bundle")
    export(out, [str(reports)])
    return out


@pytest.mark.parametrize("recipe", [recipe for recipe, _ in LABELS])
def test_every_exported_metric_round_trips(exported: Path, recipe: str) -> None:
    report = original_report(recipe)
    data = json.loads((exported / f"{recipe}.json").read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA and data["columns"] == list(COLUMNS)
    assert len(data["rows"]) == len(report["layers"]) == 196
    by_name = {layer["name"]: layer for layer in report["layers"]}
    assert sorted(row[0] for row in data["rows"]) == sorted(by_name)
    tokens = {(row, column): token for row, column, token in data["nonfinite"]}
    used: set[tuple[int, str]] = set()
    patch = {entry["name"]: entry for entry in report["patch_report"]["layers"]}
    for r, row in enumerate(data["rows"]):
        layer = by_name[row[0]]
        assert row[0] == f"model.layers.{row[1]}.{row[2]}"
        assert row[3] == layer["status"]
        for c, column in enumerate(COLUMNS[4:], start=4):
            group, key = column.split(".")
            source = layer[group][key] if layer[group] is not None else None
            value = row[c]
            if source is None:
                assert value is None and (r, column) not in tokens
            elif isinstance(source, str) or (isinstance(source, float) and not math.isfinite(source)):
                assert source in ("inf", "-inf", "nan"), source
                assert value is None and tokens[(r, column)] == source
                used.add((r, column))
            else:
                assert value == source and type(value) is type(source), (row[0], column, value, source)
        spec = data["specs"][data["spec"][r]]
        entry = patch[row[0]]
        assert spec["weight"] == {
            key: entry["weight"][key] for key in ("format", "granularity", "group_size")
        }
        assert spec["activation"] == (
            None
            if entry["activation"] is None
            else {key: entry["activation"][key] for key in ("format", "granularity", "group_size")}
        )
        assert (spec["weight_algo"], spec["transform"]) == (entry["weight_algo"], entry["transform"])
        assert (
            spec["mma"]["name"] == entry["mma"]["name"] and spec["ulp_format"] == layer["mma_ulp"]["format"]
        )
    assert used == set(tokens)  # every nonfinite flag points at a non-finite source value


def test_nonfinite_metrics_are_flagged_not_dropped(exported: Path) -> None:
    data = json.loads((exported / "syn_w4a16.json").read_text(encoding="utf-8"))
    flags = {(row, column) for row, column, token in data["nonfinite"] if token == "inf"}
    column = COLUMNS.index("activation.sqnr_db")
    assert len(flags) == 196 and all(
        row[column] is None for row in data["rows"]
    )  # weight-only: exact activations
    tokens = sorted(
        token for _, _, token in json.loads((exported / "syn_nvint4.json").read_text())["nonfinite"]
    )
    assert tokens == ["-inf", "nan"]


def test_default_order_is_block_then_q_k_v_o_gate_up_down(exported: Path) -> None:
    for recipe, _ in LABELS:
        rows = json.loads((exported / f"{recipe}.json").read_text(encoding="utf-8"))["rows"]
        keys = [(row[1], KIND_ORDER.index(row[2].rsplit(".", 1)[-1])) for row in rows]
        assert keys == sorted(keys) and len(set(keys)) == 196
        assert [row[2] for row in rows[:7]] == [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]


def test_export_twice_is_byte_identical_and_input_order_does_not_matter(
    reports: Path, tmp_path: Path
) -> None:
    first = export(tmp_path / "a", [str(reports)])
    second = export(tmp_path / "b", [str(reports)])
    third = export(
        tmp_path / "c", [str(path) for path in sorted(reports.glob("*/report.json"), reverse=True)]
    )
    assert first == second == third
    assert len(first) == len(LABELS) + 1


def test_index_lists_the_reports_in_label_order(exported: Path) -> None:
    index = json.loads((exported / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == SCHEMA and index["columns"] == list(COLUMNS)
    assert [(entry["id"], entry["label"]) for entry in index["reports"]] == LABELS
    for entry in index["reports"]:
        report = original_report(entry["id"])
        assert (exported / entry["file"]).is_file() and entry["layers"] == 196
        assert entry["metrics"] == {key: report["model"][key] for key in export_layers.MODEL_KEYS}
        assert entry["model"] == {"id": report["env"]["model_id"], "sha": report["env"]["model_sha"]}
        assert entry["env"]["git_sha"] == report["env"]["git_sha"] and entry["env"]["git_dirty"] is False
        assert (entry["samples"], entry["seqlen"], entry["dataset_fingerprint"]) == (
            report["samples"],
            report["seqlen"],
            report["dataset_fingerprint"],
        )
        assert entry["skipped"] == ["lm_head"]


def test_committed_bundle_is_consistent_and_in_page_order() -> None:
    files = {path.name for path in BUNDLE.iterdir()}
    index = json.loads((BUNDLE / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == SCHEMA and index["columns"] == list(COLUMNS)
    assert [(entry["id"], entry["label"]) for entry in index["reports"]] == COMMITTED
    assert files == {"index.json", *(entry["file"] for entry in index["reports"])}
    for entry in index["reports"]:
        data = json.loads((BUNDLE / entry["file"]).read_text(encoding="utf-8"))
        assert (data["schema"], data["id"], data["label"], data["columns"]) == (
            SCHEMA,
            entry["id"],
            entry["label"],
            list(COLUMNS),
        )
        keys = [(row[1], KIND_ORDER.index(row[2].rsplit(".", 1)[-1])) for row in data["rows"]]
        assert len(data["rows"]) == entry["layers"] == 196 and keys == sorted(keys)
        assert all(data["rows"][r][COLUMNS.index(column)] is None for r, column, _ in data["nonfinite"])
        assert all(0 <= i < len(data["specs"]) for i in data["spec"]) and len(data["spec"]) == 196
    assert sum((BUNDLE / name).stat().st_size for name in files) < 1_500_000


def write_variant(tmp_path: Path, change) -> Path:
    report = original_report("syn_int8")
    change(report)
    return rec.write(tmp_path / "bad" / "report.json", report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda report: report.update(status="failed"), "status is 'failed', expected 'complete'"),
        (lambda report: report.pop("status"), "status is None, expected 'complete'"),
        (lambda report: report.pop("model"), "missing keys model"),
        (lambda report: report["env"].pop("model_sha"), "env is missing model_sha"),
        (lambda report: report["layers"][0]["output"].pop("cosine"), "output is missing cosine"),
        (lambda report: report["layers"][0]["output"].update(mse="oops"), "expected a number"),
    ],
)
def test_bad_reports_are_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change, message: str
) -> None:
    path = write_variant(tmp_path, change)
    assert main([str(path), "--out", str(tmp_path / "out")]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_incomplete_report_raises_from_load_report(tmp_path: Path) -> None:
    path = write_variant(tmp_path, lambda report: report.update(status="running"))
    with pytest.raises(ExportError, match="expected 'complete'"):
        load_report(path)


def test_label_for_an_unknown_recipe_is_rejected(
    reports: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(reports), "--out", str(tmp_path / "out"), "--label", "nope=Nope"]) == 1
    assert "not among the reports: nope" in capsys.readouterr().err


def test_block_and_kind_are_generic() -> None:
    assert block_and_kind("model.layers.3.mlp.down_proj", "model.layers") == (3, "mlp.down_proj")
    assert block_and_kind("lm_head", "model.layers") == (None, "lm_head")
    assert block_and_kind("transformer.h.2.attn.c_attn", None) == (2, "attn.c_attn")
    assert block_and_kind("model.decoder.layers.11.fc1", None) == (11, "fc1")
    assert block_and_kind("classifier", None) == (None, "classifier")


def test_layers_outside_blocks_keep_declaration_order() -> None:
    metrics = {"mse": 0.0, "sqnr_db": "inf", "max_abs_error": 0.0, "relative_frobenius": 0.0, "cosine": 1.0}
    ulp = {"mean": 0.0, "max": 0, "p99": 0.0, "exact_fraction": 1.0, "format": "fp32"}
    declared = ["stem.proj", "head.fc", "aux.out"]
    report = {
        "layers": [
            {
                "name": name,
                "status": "measured",
                "weight": metrics,
                "activation": None,
                "output": metrics,
                "mma_ulp": ulp,
            }
            for name in sorted(declared)
        ],
        "patch_report": {"layers": [{"name": name} for name in declared]},
    }
    data = layer_file("toy", "Toy", report, Path("toy.json"))
    assert [row[0] for row in data["rows"]] == declared
    assert all(row[1] is None for row in data["rows"])


def test_exporter_imports_no_torch_or_tricast() -> None:
    code = (
        "import sys, app.export_layers\n"
        "bad = {'torch', 'tricast'} & {name.split('.')[0] for name in sys.modules}\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_demo_server_serves_the_layer_map() -> None:
    from fastapi.testclient import TestClient

    from app.server import DEFAULT_DEMO_DIR, create_app

    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    page = client.get("/layers.html")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert 'src="js/layers.js"' in page.text and 'href="index.html"' in page.text
    for path in ("/js/layers.js", "/layers.css", "/styles.css"):
        assert client.get(path).status_code == 200, path
    index = client.get("/demo/layers/index.json")
    assert index.status_code == 200 and index.json()["schema"] == SCHEMA
    for entry in index.json()["reports"]:
        assert client.get(f"/demo/layers/{entry['file']}").status_code == 200
    assert 'href="layers.html"' in client.get("/").text
