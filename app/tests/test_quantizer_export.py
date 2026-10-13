"""app/export_quantizer_demo.py: the Quantizer page bundle, the honesty check that attaches a recorded run
only to the preset whose exact recipe it ran, the PPL / lm-eval comparisons against BF16 and the demo routes.

Run records here are SYNTHETIC (app/tests/quantizer_records.py) apart from the BF16 PPL record; the committed
bundle (app/web/demo/quantizer/presets.json, exported from the GPU runs) is checked for agreement with the
generator and the pins.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr
from app.tests import quantizer_records as rec

REPO = Path(ex.__file__).resolve().parents[1]
BUNDLE = REPO / "app" / "web" / "demo" / "quantizer" / "presets.json"


def export(out: Path, runs: Path, native: Path = rec.NATIVE, lmeval: Path | None = None) -> int:
    extra = ["--lmeval", str(lmeval)] if lmeval else []
    return ex.main(["--runs", str(runs), "--native", str(native), "--out", str(out), *extra])


def bundle(out: Path) -> dict:
    return json.loads((out / "presets.json").read_text(encoding="utf-8"))


def preset_of(data: dict, preset_id: str) -> dict:
    return next(p for p in data["presets"] if p["id"] == preset_id)


def decode(value):
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value["items"]}
    return [decode(item) for item in value] if isinstance(value, list) else value


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return rec.synthetic_runs(tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="module")
def exported(synth: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("qz")
    assert export(out, synth, lmeval=synth / "lm") == 0
    return bundle(out)


def test_export_writes_valid_json(exported: dict) -> None:
    assert exported["schema"] == ex.SCHEMA
    assert [p["id"] for p in exported["presets"]] == list(qr.PRESETS)
    assert sorted(m["id"] for m in exported["methods"]) == sorted(qr.METHODS)
    assert [k["id"] for k in exported["kinds"]] == list(qr.KINDS)
    assert [kv["id"] for kv in exported["kv"]] == list(qr.KV)
    for preset in exported["presets"]:
        default, grid, kv, label = qr.PRESETS[preset["id"]]
        assert (preset["default"], preset["grid"], preset["kv"], preset["label"]) == (
            default,
            grid,
            kv,
            label,
        )
        assert preset["recipe_sha256"] == ex.PINS[preset["id"]][1]


def test_generator_tables_come_from_the_python_module_with_key_order(exported: dict) -> None:
    gen = exported["generator"]
    methods = decode(gen["methods"])
    assert methods == qr.METHODS and decode(gen["kv"]) == qr.KV and decode(gen["parent"]) == qr.PARENT
    for method, fields in qr.METHODS.items():  # key order is the YAML key order
        assert list(methods[method] or {}) == list(fields or {})
        for key, value in (fields or {}).items():
            if isinstance(value, dict):
                assert list(methods[method][key]) == list(value)
    assert gen["kinds"] == list(qr.KINDS) and gen["blocks"] == qr.BLOCKS
    assert gen["calibrated"] == sorted(qr.CALIBRATED) and gen["description"] == qr.DESCRIPTION
    assert decode(gen["options"]) == qr.OPTIONS and list(decode(gen["options"])) == list(qr.OPTIONS)
    assert decode(gen["method_options"]) == {m: list(o) for m, o in qr.METHOD_OPTIONS.items()}
    assert decode(gen["kv_options"]) == qr.KV_OPTIONS and decode(gen["calibration"]) == qr.CALIBRATION
    assert list(decode(gen["calibration"])) == list(qr.CALIBRATION)  # the calibration block's key order
    assert decode(gen["complete"]) == {"transform": qr.TRANSFORM_FIELDS, "weight_algo": qr.WEIGHT_ALGO_FIELDS}
    assert gen["calibrating_observers"] == list(qr.CALIBRATING_OBSERVERS) and gen["float_min"] == qr.FLOAT_MIN
    assert set(exported["option_info"]) == set(qr.OPTIONS) and set(exported["calibration_info"]) == set(
        qr.CALIBRATION
    )


@pytest.mark.parametrize("preset_id", list(qr.PRESETS))
def test_preset_yaml_equals_the_reference_cli_and_the_pins(exported: dict, preset_id: str) -> None:
    cli = subprocess.run(
        [sys.executable, "-m", "app.quantizer_recipe", "--preset", preset_id],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout
    assert preset_of(exported, preset_id)["yaml"] + "\n" == cli
    assert rec.preset_yaml(preset_id) == cli  # byte for byte the YAML the recorded GPU runs used
    assert hashlib.sha256(cli[:-1].encode("utf-8")).hexdigest() == ex.PINS[preset_id][0]


def test_missing_runs_are_marked_not_recorded(tmp_path: Path) -> None:
    assert export(tmp_path / "out", tmp_path / "empty", native=tmp_path / "none.json") == 0
    data = bundle(tmp_path / "out")
    assert data["native"] is None and data["native_status"] == "기록 없음"
    assert data["lmeval_native"] is None and data["lmeval_native_status"] == "기록 없음"
    for preset in data["presets"]:
        for what in ("run", "report", "lmeval"):
            assert preset[what] is None and preset[f"{what}_status"] == "기록 없음"
            assert preset[f"{what}_rejected"] is None


def test_export_twice_is_byte_identical(synth: Path, tmp_path: Path) -> None:
    assert export(tmp_path / "a", synth, lmeval=synth / "lm") == 0
    assert export(tmp_path / "b", synth, lmeval=synth / "lm") == 0
    assert (tmp_path / "a" / "presets.json").read_bytes() == (tmp_path / "b" / "presets.json").read_bytes()


def test_committed_bundle_agrees_with_the_generator_and_the_pins() -> None:
    """The page data in the repo was exported by this module from the recorded runs: its generator tables,
    catalog and preset YAMLs are what the module produces now, and every attached record carries its preset's
    pinned tricast SHA-256 (the honesty check passed when it was exported)."""
    data = json.loads(BUNDLE.read_text(encoding="utf-8"))
    assert data["schema"] == ex.SCHEMA and data["generator"] == json.loads(json.dumps(ex.generator()))
    assert {key: data[key] for key in ex.catalog()} == json.loads(json.dumps(ex.catalog()))
    for preset in data["presets"]:
        default, grid, kv, label = qr.PRESETS[preset["id"]]
        assert (preset["default"], preset["grid"], preset["kv"], preset["label"]) == (
            default,
            grid,
            kv,
            label,
        )
        assert (
            preset["yaml"] == qr.to_yaml(qr.preset(preset["id"]))
            and preset["recipe_sha256"] == ex.PINS[preset["id"]][1]
        )
        for what in ("run", "report", "lmeval"):
            if preset[what] is not None:
                assert preset[what]["recipe_hash"] == ex.PINS[preset["id"]][1], (preset["id"], what)
    assert BUNDLE.read_text(encoding="utf-8") == ex.dump(data)  # written by ex.dump


def test_a_matching_run_and_report_are_attached(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    rec.write(runs / "ppl_demo" / "qd_mixed.json", rec.ppl_record("qd_mixed", ppl=25.0))
    rec.write(runs / "report" / "qd_mixed" / "report.json", rec.report_record("qd_mixed"))
    assert export(tmp_path / "out", runs) == 0
    data = bundle(tmp_path / "out")
    mixed = preset_of(data, "qd_mixed")
    native = rec.native_record()["metrics"]["ppl"]["ppl"]
    assert mixed["run_status"] == "기록됨" and mixed["run"]["ppl"] == 25.0
    assert mixed["run"]["ppl_delta_pct"] == pytest.approx((25.0 / native - 1) * 100)
    assert mixed["run"]["recipe_hash"] == ex.PINS["qd_mixed"][1]
    assert mixed["run"]["source"] == str((runs / "ppl_demo" / "qd_mixed.json").resolve())
    sqnr = mixed["report"]["sqnr_db"]
    _, grid, _, _ = qr.PRESETS["qd_mixed"]
    assert len(sqnr) == 196
    assert [i for i, v in enumerate(sqnr) if v is None] == [
        b * 7 + k for b in range(28) for k in range(7) if grid[b][k] == "bf16"
    ]
    assert sqnr[1 * 7 + 3] == 11.3  # block 1, o_proj: block-major, q k v o gate up down
    assert all(p["run"] is None and p["report"] is None for p in data["presets"] if p["id"] != "qd_mixed")


def _set_group(recipe: dict) -> None:
    recipe["defaults"]["weight"]["group_size"] = 32


def _rename(recipe: dict) -> None:
    recipe["name"] = "qd_other"


@pytest.mark.parametrize(
    ("change", "rehash", "message"),
    [
        (_set_group, False, "does not match the recorded recipe"),
        (_set_group, True, "the recorded recipe is not preset qd_mixed"),
        (
            lambda recipe: recipe.update(description="edited"),
            True,
            "the recorded recipe is not preset qd_mixed",
        ),
        (lambda recipe: recipe["overrides"].pop(), True, "the recorded recipe is not preset qd_mixed"),
        (_rename, True, "recipe.name is 'qd_other', expected 'qd_mixed'"),
    ],
)
@pytest.mark.parametrize("kind", ["ppl", "report"])
def test_honesty_check_refuses_a_different_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change, rehash: bool, message: str, kind: str
) -> None:
    record = rec.ppl_record("qd_mixed") if kind == "ppl" else rec.report_record("qd_mixed")
    change(record["recipe"])
    if rehash:
        record["recipe_hash"] = ex.recipe_sha256(record["recipe"])
    path = "ppl_demo/qd_mixed.json" if kind == "ppl" else "report/qd_mixed/report.json"
    rec.write(tmp_path / "runs" / path, record)
    assert export(tmp_path / "out", tmp_path / "runs") == 0
    assert message in capsys.readouterr().err
    what = "run" if kind == "ppl" else "report"
    mixed = preset_of(bundle(tmp_path / "out"), "qd_mixed")
    assert (
        mixed[what] is None
        and mixed[f"{what}_status"] == "기록 없음"
        and message in mixed[f"{what}_rejected"]
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(status="failed"), "status is 'failed', expected 'complete'"),
        (
            lambda r: r["metrics"]["ppl"].update(n_windows=10),
            "not measured on the same windows as the BF16 run",
        ),
        (lambda r: r["env"].update(model_sha="0" * 40), "not measured on the same windows as the BF16 run"),
        (lambda r: r["env"].pop("git_sha"), "env is missing git_sha"),
        (lambda r: r["metrics"]["ppl"].update(ppl="nan"), "expected a finite number"),
    ],
)
def test_bad_ppl_runs_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change, message: str
) -> None:
    record = copy.deepcopy(rec.ppl_record("qd_nvfp4"))
    change(record)
    rec.write(tmp_path / "runs" / "ppl_fmt" / "qd_nvfp4.json", record)
    assert export(tmp_path / "out", tmp_path / "runs") == 0
    assert message in capsys.readouterr().err
    nvfp4 = preset_of(bundle(tmp_path / "out"), "qd_nvfp4")
    assert nvfp4["run"] is None and nvfp4["run_status"] == "기록 없음" and message in nvfp4["run_rejected"]


def test_report_layers_must_follow_the_grid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record = rec.report_record("qd_mxfp4_ends_bf16")
    record["layers"].pop(3)
    rec.write(tmp_path / "runs" / "report" / "qd_mxfp4_ends_bf16" / "report.json", record)
    assert export(tmp_path / "out", tmp_path / "runs") == 0
    assert "measured layers differ from the preset's quantized layers" in capsys.readouterr().err
    assert preset_of(bundle(tmp_path / "out"), "qd_mxfp4_ends_bf16")["report"] is None
    record = rec.report_record("qd_mxfp4_ends_bf16")
    record["layers"].append({"name": qr.layer_name(0, "q_proj"), "output": {"sqnr_db": 1.0}})  # a BF16 layer
    rec.write(tmp_path / "runs" / "report" / "qd_mxfp4_ends_bf16" / "report.json", record)
    assert export(tmp_path / "out", tmp_path / "runs") == 0
    assert "unexpected ['model.layers.0.self_attn.q_proj']" in capsys.readouterr().err


def test_a_run_recorded_twice_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rec.write(tmp_path / "runs" / "ppl_a" / "qd_nvint4.json", rec.ppl_record("qd_nvint4"))
    rec.write(tmp_path / "runs" / "ppl_b" / "qd_nvint4.json", rec.ppl_record("qd_nvint4"))
    assert export(tmp_path / "out", tmp_path / "runs") == 0
    assert "recorded more than once" in capsys.readouterr().err
    assert preset_of(bundle(tmp_path / "out"), "qd_nvint4")["run"] is None


def test_a_pin_from_an_older_preset_yaml_vouches_for_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rec.write(tmp_path / "runs" / "ppl_demo" / "qd_mixed.json", rec.ppl_record("qd_mixed"))
    stale = {**ex.PINS, "qd_mixed": ("0" * 64, ex.PINS["qd_mixed"][1])}  # as if METHODS had changed
    monkeypatch.setattr(ex, "PINS", stale)
    assert export(tmp_path / "out", tmp_path / "runs") == 0
    assert (
        "changed in app/quantizer_recipe.py since its tricast SHA-256 was pinned" in capsys.readouterr().err
    )
    mixed = preset_of(bundle(tmp_path / "out"), "qd_mixed")
    assert mixed["run"] is None and mixed["recipe_sha256"] is None


def test_lmeval_runs_attach_with_points_against_bf16(tmp_path: Path) -> None:
    lm = tmp_path / "w3"
    rec.write(lm / "lmeval_A" / "native.json", rec.lmeval_record(None))
    rec.write(lm / "lmeval_A" / "qd_mixed.json", rec.lmeval_record("qd_mixed", shift=0.02))
    rec.write(lm / "lmeval_B" / "qd_nvfp4.json", rec.lmeval_record("qd_nvfp4", shift=0.05))
    assert export(tmp_path / "out", tmp_path / "runs", lmeval=lm) == 0
    data = bundle(tmp_path / "out")
    assert data["lmeval_native_status"] == "기록됨"
    mixed = preset_of(data, "qd_mixed")["lmeval"]
    assert [(t["task"], t["metric"], t["n"], t["stderr"]) for t in mixed["tasks"]] == [
        ("arc_easy", "acc_norm", 300, 0.028),
        ("arc_challenge", "acc_norm", 300, 0.028),
        ("piqa", "acc_norm", 300, 0.028),
        ("winogrande", "acc", 300, 0.028),
        ("hellaswag", "acc_norm", 300, 0.028),
    ]
    assert all(t["delta_pts"] == pytest.approx(-2.0) for t in mixed["tasks"])
    assert mixed["macro_delta_pts"] == pytest.approx(-2.0) and mixed["limit"] == 300.0
    assert mixed["lm_eval_version"] == "0.4.13" and mixed["recipe_hash"] == ex.PINS["qd_mixed"][1]
    assert preset_of(data, "qd_nvfp4")["lmeval"]["macro_delta_pts"] == pytest.approx(-5.0)
    assert preset_of(data, "qd_nvint4")["lmeval"] is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["recipe"].update(name="qd_other"), "recipe.name is 'qd_other'"),
        (lambda r: r["metrics"]["lm_eval"]["results"].pop("piqa"), "lm-eval has no acc_norm for piqa"),
        (
            lambda r: r["metrics"]["lm_eval"]["n-samples"]["piqa"].update(effective=100),
            "not the same documents",
        ),
        (lambda r: r["metrics"]["lm_eval"]["config"].update(limit=100.0), "not the same documents"),
    ],
)
def test_bad_lmeval_runs_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change, message: str
) -> None:
    lm = tmp_path / "w3"
    rec.write(lm / "lmeval_A" / "native.json", rec.lmeval_record(None))
    record = rec.lmeval_record("qd_mixed")
    change(record)
    rec.write(lm / "lmeval_A" / "qd_mixed.json", record)
    assert export(tmp_path / "out", tmp_path / "runs", lmeval=lm) == 0
    assert message in capsys.readouterr().err
    mixed = preset_of(bundle(tmp_path / "out"), "qd_mixed")
    assert mixed["lmeval"] is None and message in mixed["lmeval_rejected"]


def test_native_must_be_the_bf16_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    native = rec.write(tmp_path / "native.json", rec.ppl_record("qd_nvfp4"))
    assert export(tmp_path / "out", tmp_path / "runs", native=native) == 1
    assert "expected 'native'" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_catalog_bits_include_scale_overhead(exported: dict) -> None:
    bits = {m["id"]: m["weight_bits"] for m in exported["methods"]}
    for method in ("nvfp4", "awq_nvfp4", "smoothquant_nvfp4", "nvint4", "bfp12"):
        assert set(bits[method].values()) == {4.5}
    assert set(bits["mxfp4"].values()) == {4.25} and set(bits["mxfp8"].values()) == {8.25}
    assert set(bits["gptq_int4"].values()) == {4 + 20 / 128} and set(bits["bf16"].values()) == {16}
    assert bits["int8"] == {
        "q_proj": 8 + 32 / 1024,
        "k_proj": 8 + 32 / 1024,
        "v_proj": 8 + 32 / 1024,
        "o_proj": 8 + 32 / 2048,
        "gate_proj": 8 + 32 / 1024,
        "up_proj": 8 + 32 / 1024,
        "down_proj": 8 + 32 / 3072,
    }
    params = {k["id"]: k["params"] for k in exported["kinds"]}
    assert params == {
        "q_proj": 1024 * 2048,
        "k_proj": 1024 * 1024,
        "v_proj": 1024 * 1024,
        "o_proj": 2048 * 1024,
        "gate_proj": 1024 * 3072,
        "up_proj": 1024 * 3072,
        "down_proj": 3072 * 1024,
    }
    flags = {m["id"]: m["calibration"] for m in exported["methods"]}
    assert {m for m, flag in flags.items() if flag} == qr.CALIBRATED


def test_exporter_imports_no_torch_or_tricast() -> None:
    code = (
        "import sys, app.export_quantizer_demo\n"
        "bad = {'torch', 'tricast'} & {name.split('.')[0] for name in sys.modules}\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_demo_server_serves_the_quantizer() -> None:
    from fastapi.testclient import TestClient

    from app.server import DEFAULT_DEMO_DIR, create_app

    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    page = client.get("/quantizer.html")
    assert page.status_code == 200 and 'src="js/quantizer.js"' in page.text
    assert 'href="layers.html"' in page.text and 'href="index.html"' in page.text
    for path in ("/js/quantizer.js", "/quantizer.css", "/styles.css", "/js/ui.js"):
        assert client.get(path).status_code == 200, path
    data = client.get("/demo/quantizer/presets.json")
    assert data.status_code == 200 and data.json()["schema"] == ex.SCHEMA
    assert 'href="quantizer.html"' in client.get("/").text
    assert 'href="quantizer.html"' in client.get("/layers.html").text
