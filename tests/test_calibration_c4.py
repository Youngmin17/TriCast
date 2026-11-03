"""The C4 calibration loader passes no config name (datasets 4.x: config "en" fails split verification on a
single shard, ExpectedMoreSplitsError) and reads exactly the first official shard of the split."""

from __future__ import annotations

import pytest

from tricast import calibration


@pytest.mark.parametrize(("split", "shards"), [("train", "01024"), ("validation", "00008")])
def test_c4_calibration_loads_one_shard_without_a_config_name(monkeypatch: pytest.MonkeyPatch, split: str,
                                                              shards: str) -> None:
    datasets = pytest.importorskip("datasets")
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"text": "a"}]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    assert list(calibration._dataset_texts("c4", split)) == [{"text": "a"}]
    assert calls == [(("allenai/c4",), {"split": split, "keep_in_memory": False,
                                        "data_files": {split: f"en/c4-{split}.00000-of-{shards}.json.gz"}})]


def test_c4_calibration_refuses_the_test_split() -> None:
    pytest.importorskip("datasets")
    with pytest.raises(ValueError, match="train or validation"):
        calibration._dataset_texts("c4", "test")
