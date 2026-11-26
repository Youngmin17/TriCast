"""Small contract checks for target-aware solve. Fixtures are not runtime/quality evidence."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from app import autotune as at
from app import packed_targets as pt


def case(target: str) -> dict:
    names = pt.layer_names(target, 1)
    kinds = {"Q4_K": (144, 2), "Q8_0": (272, 1)} if target == pt.GGUF else {pt.CT_FORMAT: (132, 1)}
    rows = {
        name: {
            "shape_nk": [64, 256],
            "measurements": {kind: {"payload_bytes": size * 64, "relative_sse": err,
                                    "artifact_sha256": "1" * 64, "payload_sha256": "2" * 64}
                             for kind, (size, err) in kinds.items()},
        }
        for name in names
    }
    return {"schema": pt.SCHEMA, "objective": pt.OBJECTIVE, "target": target, "architecture": "qwen3",
            "reference_sha256": "0" * 64, "blocks": 1, "layers": rows}


def check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def run(table: dict, bits: str = "5.1", target: str = pt.GGUF) -> dict | None:
            path = root / "table.json"
            path.write_text(json.dumps(table))
            with tempfile.TemporaryDirectory(dir=root) as out:
                rc = at.main(["solve", "--target", target, "--table", str(path),
                              "--out", out, "--budget-bits", bits])
                files = list(Path(out).iterdir())
                if rc == 2 and not files:
                    return None
                assert rc == 0 and [p.name for p in files] == ["packed-plan.json"], (rc, files)
                return json.loads(files[0].read_text())

        table = case(pt.GGUF)
        plan = run(table)
        assert plan and not plan["runtime_verified"] and plan["avg_bits"] <= 5.1
        # 5.1 bpw allows exactly one 8.5 bpw tensor and six 4.5 bpw tensors.
        assert [v["layout"] for v in plan["layers"].values()].count("Q8_0") == 1
        assert plan["objective_value"] == 13
        assert run(table, bits="4.49") is None
        assert run(table, target=pt.CT) is None
        assert run({"schema": "tricast.autotune.table/2"}) is None
        for mutation in ("unmeasured", "missing", "layout", "shape", "cost", "score", "sha"):
            t = copy.deepcopy(table)
            rows = t["layers"]
            name = next(iter(rows))
            row = rows[name]
            ms = row["measurements"]
            if mutation == "unmeasured":
                ms.clear()
            elif mutation == "missing":
                del rows[name]
            elif mutation == "layout":
                ms["gptq_int4"] = ms.pop("Q4_K")
            elif mutation == "shape":
                row["shape_nk"][1] = 255
            elif mutation == "cost":
                ms["Q8_0"]["payload_bytes"] -= 1
            elif mutation == "score":
                ms["Q8_0"]["relative_sse"] = None
            else:
                ms["Q4_K"]["artifact_sha256"] = ""
            assert run(t) is None, mutation
        assert run(case(pt.CT), bits="4.125", target=pt.CT)
        assert run(case(pt.CT), bits="4", target=pt.CT) is None
    print("PACKED_TARGET_CONTRACT_PASS")


if __name__ == "__main__":
    check()
