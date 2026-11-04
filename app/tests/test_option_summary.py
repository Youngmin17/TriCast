"""Option summaries: a layer card's option space covers the Python-only methods (METHOD_OPTIONS resolves them
through __missing__, which dict.get skips), and the Auto-tune bundle keeps of a proposal's options only those
away from their defaults, checked by app.quantizer_recipe.method_options. CPU only."""

from __future__ import annotations

from app import export_autotune_demo as ed
from app import layer_cards as lc

GPTQ = {"weight.group_size": 128, "weight.rounding": "rne", "weight_algo.act_order": True,
        "weight_algo.block_size": 128, "weight_algo.damp": 0.05}


def test_a_python_only_method_has_its_option_space() -> None:
    assert lc._option_space("gptq_int2") == {"weight.group_size": 128, "weight.rounding": "rne",
                                             "weight_algo.damp": 0.01, "weight_algo.act_order": False,
                                             "weight_algo.block_size": 128}
    assert lc._option_space("hqq_int3") == {} and lc._option_space("bf16") == {}


def test_only_options_away_from_their_defaults_are_kept() -> None:
    options = {"gptq_int2": GPTQ, "rht_gptq_int3": GPTQ, "gptq_int4": {**GPTQ, "weight_algo.act_order": False,
                                                                      "weight_algo.damp": 0.01}}
    assert ed.changed_options(options) == {
        "gptq_int2": {"weight_algo.damp": 0.05, "weight_algo.act_order": True},
        "rht_gptq_int3": {"weight_algo.damp": 0.05, "weight_algo.act_order": True}}
    assert ed.changed_options({"gptq_int2": {"weight.not_an_option": 1}}) is None  # a refused reply's options
    assert ed.changed_options(None) is None
