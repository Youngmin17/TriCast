"""Evidence ledger (app/ledger.py) and its retrieval (tricast.rag.search_evidence) — SPEC AC23."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import ledger as lg
from tricast import rag

ENV = {"utc": "2026-10-13T14:29:15+00:00", "hostname": "turin", "git_sha": "73f7893", "git_dirty": False,
       "versions": {"torch": "2.8.0", "transformers": "4.55.2", "triton": "3.4.0"},
       "gpu_names": ["NVIDIA A100 80GB PCIe"], "model_id": "Qwen/Qwen3-0.6B", "model_sha": "c1899de"}
META = {"model_id": "Qwen/Qwen3-0.6B", "model_sha": "c1899de", "windows": 8, "seqlen": 512, "seed": 42,
        "dataset_fingerprint": "f1a7", "dtype": "bfloat16", "git_sha": "73f7893", "git_dirty": False,
        "gpu_names": ["NVIDIA A100 80GB PCIe"], "env": ENV,
        "method_recipe_sha256": {"nvfp4": "aa", "mxfp4": "bb"}}
O_PROJ, DOWN = "model.layers.13.self_attn.o_proj", "model.layers.2.mlp.down_proj"


def _runs(root: Path) -> Path:
    runs = root / "runs" / "w"
    runs.mkdir(parents=True)
    table = {"schema": "tricast.autotune.table/2", "meta": META, "layers": [
        {"name": O_PROJ, "block": 13, "kind": "o_proj", "kl": {"bf16": 0.0, "nvfp4": 0.0026136211788828816,
                                                               "mxfp4": 0.004, "awq_nvfp4": 0.0021},
         "floor": 0.00086},
        {"name": DOWN, "block": 2, "kind": "down_proj",
         "kl": {"bf16": 0.0, "nvfp4": 0.0110009, "mxfp4": 0.0262213, "awq_nvfp4": None}}]}
    (runs / "table.json").write_text(json.dumps(table))
    entries = [{"config": META}, {"layer": O_PROJ, "method": "nvfp4", "kl": 0.0026136211788828816}]
    (runs / "table.entries.jsonl").write_text("".join(json.dumps(line) + "\n" for line in entries))
    measure = {"schema": "tricast.autotune.measure/1", "meta": {**META, "seed": 43}, "results": [
        {"name": "qd_nvfp4", "recipe_sha256": "9f26", "kl": 0.2777044510710148, "patched": 196, "kv": []}]}
    (runs / "kl_seed43.json").write_text(json.dumps(measure))
    run = {"run_hash": "r1", "recipe_hash": "9f26", "recipe": {"name": "qd_nvfp4"}, "env": ENV,
           "metrics": {"ppl": {"ppl": 26.15988887580222, "n_tokens": 298862, "n_windows": 146},
                       "lm_eval": {"results": {"piqa": {"acc,none": 0.67, "acc_stderr,none": 0.027,
                                                        "acc_norm,none": 0.68, "sample_len": 300}}}}}
    (runs / "qd_nvfp4.json").write_text(json.dumps(run))
    report = {"recipe_hash": "9f26", "recipe": {"name": "qd_nvfp4"}, "env": ENV, "samples": 8, "seqlen": 512,
              "seed": 42, "model": {"logits_kl": 0.28, "top1_agreement": 0.81},
              "layers": [{"name": O_PROJ, "output": {"sqnr_db": 14.501970327819883, "cosine": 0.98}}]}
    (runs / "report.json").write_text(json.dumps(report))
    (runs / "env.json").write_text(json.dumps(ENV))  # not a result: skipped
    return runs


def test_every_number_becomes_one_record_that_reads_back_from_its_source(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    path = tmp_path / "ledger.jsonl"
    summary = lg.ingest([runs], path, tmp_path)
    # table: o_proj 4 methods + floor, down_proj 3 (awq not applicable); its entries.jsonl is the same data
    assert summary["added"] == {"layer_kl": 8, "lmeval": 2, "layer_report": 2, "model_report": 2, "ppl": 1,
                                "recipe_kl": 1}
    records = lg.read_ledger(path)
    assert all(lg.verify(record, tmp_path) for record in records)
    floor = [r for r in records if r["kind"] == "layer_kl" and r["method"] == "floor"]
    assert [(r["layer"], r["block"], r["layer_kind"], r["value"]) for r in floor] == \
        [(O_PROJ, 13, "o_proj", 0.00086)]
    assert not any(r.get("method") == "awq_nvfp4" and r["layer"] == DOWN for r in records)
    ppl = next(r for r in records if r["kind"] == "ppl")
    assert ppl["value"] == 26.15988887580222 and ppl["env"]["gpu_names"] == ["NVIDIA A100 80GB PCIe"]
    assert lg.ingest([runs], path, tmp_path)["added"] == {}  # same files, same ids
    assert len(lg.read_ledger(path)) == len(records)


def test_a_changed_source_no_longer_verifies(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    path = tmp_path / "ledger.jsonl"
    lg.ingest([runs], path, tmp_path)
    record = next(r for r in lg.read_ledger(path) if r["kind"] == "ppl")
    text = (runs / "qd_nvfp4.json").read_text()
    (runs / "qd_nvfp4.json").write_text(text.replace("26.15988887580222", "26.15988887580223"))
    assert not lg.verify(record, tmp_path)


def test_a_number_without_an_environment_is_refused(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    bare = json.loads((runs / "qd_nvfp4.json").read_text())
    bare["env"] = {"model_id": "Qwen/Qwen3-0.6B"}
    (runs / "qd_nvfp4.json").write_text(json.dumps(bare))
    with pytest.raises(lg.LedgerError, match="no environment capture"):
        lg.ingest([runs / "qd_nvfp4.json"], tmp_path / "ledger.jsonl", tmp_path)


def test_documents_cite_their_records_and_are_found_by_layer_and_method(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    path = tmp_path / "ledger.jsonl"
    lg.ingest([runs], path, tmp_path)
    records = lg.read_ledger(path)
    docs_file = lg.docs_path(path)
    count = lg.write_documents(records, docs_file)
    docs = [json.loads(line) for line in docs_file.read_text().splitlines()]
    assert count == len(docs) and {doc["schema"] for doc in docs} == {lg.DOCS_SCHEMA}
    ids = {record["id"] for record in records}
    for doc in docs:
        assert doc["ids"] and set(doc["ids"]) <= ids
        assert all(f"[{item}]" in doc["text"] for item in doc["ids"])
    hits = rag.search_evidence("block13 oproj nvfp4", 3, ledgers=[str(docs_file)])
    assert hits[0]["section"] == f"layer {O_PROJ}" and set(hits[0]) == {"source", "section", "text", "score"}
    assert "nvfp4 0.00261362" in hits[0]["text"] and "output sqnr_db 14.502" in hits[0]["text"]
    method = rag.search_evidence("mxfp4 most sensitive", 1, ledgers=[str(docs_file)])[0]
    assert method["section"].startswith("method mxfp4")
    assert method["text"].index(DOWN) < method["text"].index(O_PROJ)  # most sensitive first
    recipe = rag.search_evidence("qd_nvfp4 piqa", 1, ledgers=[str(docs_file)])[0]
    assert recipe["section"] == "recipe qd_nvfp4" and "WikiText-2 test PPL 26.1599" in recipe["text"]
    assert rag.search_evidence("block13", 3, ledgers=[str(tmp_path / "missing.jsonl")]) == []


def test_contract_search_ignores_the_ledgers_and_bad_ledger_lines_are_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.docs.jsonl"
    bad.write_text(json.dumps({"source": "x", "section": 3, "text": "nvfp4"}) + "\n")
    with pytest.raises(ValueError, match="string source, section, text"):
        rag.search_evidence("nvfp4", 3, ledgers=[str(bad)])
    assert not any(hit["source"].startswith("ledger:") for hit in rag.search("nvfp4 block13 o_proj", 50))


def test_method_cards_of_the_catalog_are_searchable_beside_the_evidence(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("methods:\n"
                       "- {id: wanda, name: Wanda, axes: [weight_sparsity], tricast: official_adapter,\n"
                       "   entry: 'lib/prune.py:prune_wanda', note: activation-aware pruning}\n"
                       "- {id: kivi, name: KIVI, axes: [KV], tricast: tricast_reimplementation}\n")
    hits = rag.search_evidence("activation-aware pruning", 2, ledgers=[], catalogs=[str(catalog)])
    assert [hit["section"] for hit in hits] == ["method card wanda"]
    assert "tricast: official_adapter" in hits[0]["text"] and hits[0]["source"] == str(catalog)
    catalog.write_text("methods:\n- {name: no id}\n")
    with pytest.raises(ValueError, match="string id"):
        rag.search_evidence("kivi", 2, ledgers=[], catalogs=[str(catalog)])
    repo = rag.search_evidence("Wanda official adapter", 5, ledgers=[])
    assert "method card wanda" in [hit["section"] for hit in repo]  # docs/research/methods_catalog.yaml


def test_a_resumed_log_keeps_its_ids_a_rewritten_value_is_refused_and_the_table_replaces_its_log(
        tmp_path: Path) -> None:
    """Review: ids hung on the file hash, so a resumed (appended) .entries.jsonl duplicated its records."""
    runs = tmp_path / "runs" / "t"
    runs.mkdir(parents=True)
    log, path = runs / "table.entries.jsonl", tmp_path / "ledger.jsonl"
    lines = [{"config": META}, {"layer": O_PROJ, "method": "nvfp4", "kl": 0.0026}]
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert lg.ingest([runs], path, tmp_path)["added"] == {"layer_kl": 1}
    first = lg.read_ledger(path)[0]
    lines.append({"layer": DOWN, "method": "nvfp4", "kl": 0.011})
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    summary = lg.ingest([runs], path, tmp_path)
    assert summary["added"] == {"layer_kl": 1} and summary["refreshed"] == 1
    records = lg.read_ledger(path)
    assert len(records) == 2 and first["id"] in {r["id"] for r in records}
    assert all(lg.verify(record, tmp_path) for record in records)
    lines[1]["kl"] = 0.0027
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(lg.LedgerError, match=r"changed in place \(\['value'\]\)"):
        lg.ingest([runs], path, tmp_path)
    assert lg.read_ledger(path) == records  # nothing written
    lines[1]["kl"] = 0.0026
    lines[0] = {"config": {**META, "model_sha": "another"}}  # same numbers, another model (re-check NEW-1)
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(lg.LedgerError, match="changed in place.*model_sha"):
        lg.ingest([runs], path, tmp_path)
    log.write_text(json.dumps({"config": META}) + "\n")  # every measured line gone (re-check NEW-3)
    with pytest.raises(lg.LedgerError, match="no longer in the file"):
        lg.ingest([runs], path, tmp_path)
    lines[0] = {"config": META}
    lines[1]["kl"] = 0.0026
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    table = {"schema": "tricast.autotune.table/2", "meta": META, "layers": [
        {"name": O_PROJ, "block": 13, "kind": "o_proj", "kl": {"nvfp4": 0.0026}},
        {"name": DOWN, "block": 2, "kind": "down_proj", "kl": {"nvfp4": 0.011}}]}
    (runs / "table.json").write_text(json.dumps(table))
    summary = lg.ingest([runs], path, tmp_path)
    assert summary["superseded"] == 2 and summary["added"] == {"layer_kl": 2}
    records = {r["id"]: r for r in lg.read_ledger(path)}
    assert len(records) == 4 and all(lg.verify(record, tmp_path) for record in records.values())
    # the log's ids still resolve (proposals cite them) and point at the table's record of the same number
    logged = records[first["id"]]
    assert records[logged["superseded_by"]]["source"]["path"] == "runs/t/table.json"
    assert records[logged["superseded_by"]]["value"] == logged["value"]
    cited = {item for doc in lg.documents(list(records.values())) for item in doc["ids"]}
    assert first["id"] not in cited and logged["superseded_by"] in cited


def _write_lines(path: Path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_a_log_resumed_after_its_ingest_still_verifies_once_its_table_supersedes_it(tmp_path: Path) -> None:
    """Review M5: superseded log records kept the log hash of their own ingest; lines the resumed run appended
    later (directory ingest skips the log once the table exists) left them failing verify."""
    runs = tmp_path / "runs" / "t"
    runs.mkdir(parents=True)
    log, path = runs / "table.entries.jsonl", tmp_path / "ledger.jsonl"
    lines = [{"config": META}, {"layer": O_PROJ, "method": "nvfp4", "kl": 0.0026}]
    _write_lines(log, lines)
    assert lg.ingest([runs], path, tmp_path)["added"] == {"layer_kl": 1}
    lines.append({"layer": DOWN, "method": "nvfp4", "kl": 0.011})  # the resumed run appends, then finishes
    _write_lines(log, lines)
    table = {"schema": "tricast.autotune.table/2", "meta": META, "layers": [
        {"name": O_PROJ, "block": 13, "kind": "o_proj", "kl": {"nvfp4": 0.0026}},
        {"name": DOWN, "block": 2, "kind": "down_proj", "kl": {"nvfp4": 0.011}}]}
    (runs / "table.json").write_text(json.dumps(table))
    summary = lg.ingest([runs], path, tmp_path)
    records = lg.read_ledger(path)
    assert all(lg.verify(record, tmp_path) for record in records)
    assert summary["added"] == {"layer_kl": 2} and summary["superseded"] == 1 and summary["refreshed"] == 1
    assert len(records) == 3 and sum("superseded_by" in record for record in records) == 1
    lines[1]["kl"] = 0.0027  # the log rewritten after its table: refused as in a re-ingest
    _write_lines(log, lines)
    with pytest.raises(lg.LedgerError, match=r"changed in place \(\['value'\]\)"):
        lg.ingest([runs], path, tmp_path)
    _write_lines(log, lines[:1])
    with pytest.raises(lg.LedgerError, match="no longer in the file"):
        lg.ingest([runs], path, tmp_path)
    assert lg.read_ledger(path) == records  # nothing written


def test_a_table_of_another_model_or_window_setting_does_not_supersede_a_log(tmp_path: Path) -> None:
    """Review M5: supersession matched (layer, method, value) only; bf16 is 0.0 for every model and seed."""
    runs = tmp_path / "runs" / "t"
    runs.mkdir(parents=True)
    path = tmp_path / "ledger.jsonl"
    _write_lines(runs / "table.entries.jsonl",
                 [{"config": META}, {"layer": O_PROJ, "method": "bf16", "kl": 0.0}])
    lg.ingest([runs], path, tmp_path)
    before = lg.read_ledger(path)
    for meta, differ in (({**META, "model_sha": "another"}, "model_sha"), ({**META, "seed": 7}, "setting"),
                         ({**META, "source": {"split": "test"}}, "setting")):
        table = {"schema": "tricast.autotune.table/2", "meta": meta,
                 "layers": [{"name": O_PROJ, "block": 13, "kind": "o_proj", "kl": {"bf16": 0.0}}]}
        (runs / "table.json").write_text(json.dumps(table))
        with pytest.raises(lg.LedgerError, match=rf"another model or window setting \(\['{differ}'\]\)"):
            lg.ingest([runs], path, tmp_path)
        assert lg.read_ledger(path) == before  # nothing written
    table["meta"] = META
    (runs / "table.json").write_text(json.dumps(table))
    assert lg.ingest([runs], path, tmp_path)["superseded"] == 1  # the same model and setting supersede


def test_an_exact_layer_name_finds_its_own_document_first(tmp_path: Path) -> None:
    """Method documents list layer names too; the layer's own document must still come first."""
    runs = _runs(tmp_path)
    path = tmp_path / "ledger.jsonl"
    lg.ingest([runs], path, tmp_path)
    docs_file = lg.docs_path(path)
    lg.write_documents(lg.read_ledger(path), docs_file)
    for name in (O_PROJ, DOWN):
        hits = rag.search_evidence(name, 3, ledgers=[str(docs_file)], catalogs=[])
        assert hits[0]["section"] == f"layer {name}"
    hits = rag.search_evidence("v_proj block 13", 3, ledgers=[str(docs_file)], catalogs=[])
    assert "block 13 block13" in next(hit["text"] for hit in hits if hit["section"] == f"layer {O_PROJ}")


def _ingest_each(paths: list[str], ledger: str, root: str) -> None:
    for path in paths:
        lg.ingest([Path(path)], Path(ledger), Path(root))


def test_concurrent_ingests_lose_no_record(tmp_path: Path) -> None:
    """A tune loop and a person ingest into the same ledger: every record of both must survive."""
    import multiprocessing

    runs = tmp_path / "runs"
    runs.mkdir()
    files = []
    for index in range(24):
        run = {"run_hash": f"r{index}", "recipe_hash": f"h{index}", "recipe": {"name": f"x{index}"},
               "env": ENV, "metrics": {"ppl": {"ppl": 20.0 + index, "n_tokens": 1}}}
        path = runs / f"run{index}.json"
        path.write_text(json.dumps(run))
        files.append(str(path))
    ledger = tmp_path / "ledger.jsonl"
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_ingest_each, args=(files[part::2], str(ledger), str(tmp_path)))
               for part in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    values = sorted(record["value"] for record in lg.read_ledger(ledger))
    assert values == [20.0 + index for index in range(24)]


def _document_slowly(path: str, ledger: str, root: str, writing, written) -> None:
    """ingest_and_document whose document writer stalls until the other process has documented (2 s at
    most)."""
    write = lg.write_documents

    def stalled(records, target):
        writing.set()
        written.wait(2)
        return write(records, target)

    lg.write_documents = stalled
    lg.ingest_and_document([Path(path)], Path(ledger), Path(root))


def _document_meanwhile(path: str, ledger: str, root: str, writing, written) -> None:
    assert writing.wait(120)
    lg.ingest_and_document([Path(path)], Path(ledger), Path(root))
    written.set()


def test_a_slow_document_writer_cannot_leave_older_documents_than_the_ledger(tmp_path: Path) -> None:
    """Review m3: documents were written outside the ingest lock, so a writer holding an older snapshot could
    replace the documents another ingest had just written."""
    import multiprocessing

    runs = tmp_path / "runs"
    runs.mkdir()
    files = []
    for index in range(2):
        run = {"run_hash": f"r{index}", "recipe_hash": f"h{index}", "recipe": {"name": f"x{index}"},
               "env": ENV, "metrics": {"ppl": {"ppl": 20.0 + index, "n_tokens": 1}}}
        (runs / f"run{index}.json").write_text(json.dumps(run))
        files.append(str(runs / f"run{index}.json"))
    ledger = tmp_path / "ledger.jsonl"
    context = multiprocessing.get_context("spawn")
    writing, written = context.Event(), context.Event()
    workers = [context.Process(target=target, args=(file, str(ledger), str(tmp_path), writing, written))
               for target, file in ((_document_slowly, files[0]), (_document_meanwhile, files[1]))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    records = lg.read_ledger(ledger)
    assert sorted(record["value"] for record in records) == [20.0, 21.0]
    docs = [json.loads(line) for line in lg.docs_path(ledger).read_text().splitlines()]
    assert docs == [{"schema": lg.DOCS_SCHEMA, **doc} for doc in lg.documents(records)]


def test_without_revisions_supersession_needs_the_same_model_id(tmp_path: Path) -> None:
    """Re-check M5: with no model SHA on either side, a table of another model id must not supersede a log."""
    runs = tmp_path / "runs" / "t"
    runs.mkdir(parents=True)
    meta = {**META, "model_sha": None, "env": {**ENV, "model_sha": None}}
    lines = [{"config": meta}, {"layer": O_PROJ, "method": "bf16", "kl": 0.0}]
    (runs / "table.entries.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    path = tmp_path / "ledger.jsonl"
    lg.ingest([runs], path, tmp_path)
    other = {**meta, "model_id": "meta-llama/Llama-3.2-1B"}
    table = {"schema": "tricast.autotune.table/2", "meta": other,
             "layers": [{"name": O_PROJ, "block": 13, "kind": "o_proj", "kl": {"bf16": 0.0}}]}
    (runs / "table.json").write_text(json.dumps(table))
    with pytest.raises(lg.LedgerError, match="model_id"):
        lg.ingest([runs], path, tmp_path)
