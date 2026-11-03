"""Opt-in per-layer calibration cache: bit-identical reuse, partial hits, invalidation, uncacheable layers."""

from __future__ import annotations

import copy
import json
import zipfile

import pytest
import torch
from torch import nn

from tricast.calibration import _sha256_json, calibrate
from tricast.nn import iter_emulinear
from tricast.recipe import load_recipe

IDS = torch.arange(64).remainder(32).reshape(2, 32)
PROBE = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
WINDOWS = {"input_ids": IDS, "samples": 3, "seqlen": 8, "seed": 0}
# Attention: AWQ fitted jointly over q/k/v; MLP: SmoothQuant over gate/up with static EMA activations
# (the diagonal observer path); o_proj: static EMA observer; down_proj: GPTQ (the defaults).
AWQ = {"weight": "nvfp4", "activation": "nvfp4", "weight_algo": "rtn",
       "transform": {"kind": "awq", "grid": 3, "share_inputs": True}}
SMOOTH = {"weight": "nvfp4", "activation": "fp8_tensor_ema", "weight_algo": "rtn",
          "transform": {"kind": "smoothquant", "alpha": 0.5}}
STATIC = {"weight": "fp8_tensor", "activation": "fp8_tensor_ema", "weight_algo": "rtn"}
GPTQ = {"weight": "int4_g128_zp", "weight_algo": "gptq"}
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


@pytest.fixture(params=["llama", "qwen3"])
def tiny(request):
    transformers = pytest.importorskip("transformers")
    config_cls, model_cls = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM) if request.param == "llama"
        else (transformers.Qwen3Config, transformers.Qwen3ForCausalLM)
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = config_cls(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=32, max_position_embeddings=32)
    config._attn_implementation = "eager"
    yield model_cls(config).eval()
    torch.set_num_threads(old_threads)


def _recipe(overrides=(), defaults=None, **extra):
    return load_recipe({"name": "cache", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"},
                                                      **(GPTQ if defaults is None else defaults)},
                        "overrides": list(overrides), "backend": "reference", **extra})


def _mixed(**changes):
    rules = {"qkv": {"modules": ["q_proj", "k_proj", "v_proj"], **AWQ},
             "gate_up": {"modules": ["gate_proj", "up_proj"], **SMOOTH},
             "o": {"modules": ["o_proj"], **STATIC}}
    for rule, change in changes.items():
        rules[rule] = change
    return _recipe(rules.values())


def _calibrate(base, recipe, cache_dir=None, **options):
    model = copy.deepcopy(base)  # cold start: unpatched and never calibrated
    result = calibrate(model, recipe, cache_dir=cache_dir, **{**WINDOWS, **options})
    return model, result


def _operand(operand):
    if operand is None:
        return None
    return {key: getattr(operand, key)
            for key in ("values", "fmt", "scale", "scale_fmt", "scale_kind", "k_domain", "alpha")}


def _state(model) -> dict:
    """Every field calibrate() snapshots, per layer (the weight-state tuple as whether it is current)."""
    state = {}
    for name, layer in iter_emulinear(model):
        transform = layer.transform
        state[name] = {
            "mode": layer.mode, "fitted": layer._fitted, "calibrated": layer._calibrated,
            "current": layer._weight_current(),
            "transform": {"kind": transform.kind, "block": transform.block, "diag": transform.diag},
            "observer": None if layer.observer is None else layer.observer.state_dict(),
            "weight_operand": _operand(layer._weight_operand),
            "outlier_operand": _operand(layer._outlier_operand), "weight_noise": layer._weight_noise,
        }
    return state


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _assert_same(a, b, path="state") -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert (a.dtype, a.shape, a.stride(), a.device) == (b.dtype, b.shape, b.stride(), b.device), path
        assert torch.equal(a, b) and torch.equal(_bits(a), _bits(b)), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{path}[{index}]")
    else:
        assert type(a) is type(b) and a == b, path


def _assert_same_model(reference, model) -> None:
    _assert_same(_state(reference), _state(model))
    _assert_same(reference(PROBE).logits, model(PROBE).logits, "logits")


def _names(model, modules) -> list[str]:
    return [name for name, _ in iter_emulinear(model) if name.rsplit(".", 1)[-1] in modules]


# Fields that describe the work done in this call rather than the calibrated model.
WORK = ("full_model_forwards", "hessian_bytes", "hessian_unshared_bytes", "hessian_groups", "cache")


def test_cache_hit_is_bit_identical_and_runs_no_forward(tiny, tmp_path):
    recipe = _mixed()
    reference, plain = _calibrate(tiny, recipe)
    filled_model, filled = _calibrate(tiny, recipe, tmp_path)
    hit_model, hit = _calibrate(tiny, recipe, tmp_path)
    assert "cache" not in plain
    assert filled["cache"]["hits"] == [] and filled["cache"]["uncacheable"] == {}
    assert filled["cache"]["misses"] == filled["layers"]
    assert hit["cache"]["hits"] == hit["layers"] and hit["cache"]["misses"] == []
    assert hit["cache"]["uncacheable"] == {} and len(hit["cache"]["source_sha256"]) == 64
    assert plain["full_model_forwards"] == filled["full_model_forwards"] == 4  # GPTQ preflight + 3 windows
    assert hit["full_model_forwards"] == 0
    assert {k: v for k, v in filled.items() if k != "cache"} == plain  # nothing cached yet: same call
    same = [key for key in plain if key not in WORK]
    assert {key: hit[key] for key in same} == {key: plain[key] for key in same}
    assert any(len(group) == 3 for group in hit["groups"]) and any(len(group) == 2 for group in hit["groups"])
    _assert_same_model(reference, filled_model)
    _assert_same_model(reference, hit_model)
    entries = sorted(tmp_path.glob("layer-*.pt"))
    assert len(entries) == len(plain["layers"]) and len(list(tmp_path.glob("inputs-*.json"))) == 1
    assert torch.load(entries[0], weights_only=True)["state"]["mode"] == "frozen"
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_hit_fits_only_changed_layers(tiny, tmp_path):
    recipe_a = _mixed()
    recipe_b = _mixed(gate_up={"modules": ["gate_proj", "up_proj"], **SMOOTH,
                               "transform": {"kind": "smoothquant", "alpha": 0.75}},
                      down={"modules": ["down_proj"], **GPTQ, "weight_algo": {"kind": "gptq", "damp": 0.05}})
    _calibrate(tiny, recipe_a, tmp_path)
    reference, plain = _calibrate(tiny, recipe_b)
    model, result = _calibrate(tiny, recipe_b, tmp_path)
    assert result["cache"]["hits"] == _names(model, ATTENTION)
    assert result["cache"]["misses"] == _names(model, MLP) and result["cache"]["uncacheable"] == {}
    assert result["full_model_forwards"] == plain["full_model_forwards"] == 4
    assert result["groups"] == plain["groups"]
    _assert_same_model(reference, model)
    again_model, again = _calibrate(tiny, recipe_b, tmp_path)
    assert again["cache"]["hits"] == again["layers"] and again["full_model_forwards"] == 0
    _assert_same_model(reference, again_model)


def test_partial_hit_inside_a_shared_gptq_hessian(tiny, tmp_path):
    # q/k/v (and gate/up) share one Hessian accumulator; refitting k_proj alone makes it the owner.
    _calibrate(tiny, _recipe(), tmp_path)
    recipe = _recipe([{"modules": ["k_proj"], **GPTQ, "weight_algo": {"kind": "gptq", "damp": 0.05}}])
    reference, plain = _calibrate(tiny, recipe)
    model, result = _calibrate(tiny, recipe, tmp_path)
    assert any(len(group) == 3 for group in plain["hessian_groups"])
    assert result["cache"]["misses"] == _names(model, ("k_proj",))
    assert result["hessian_groups"] == [[name] for name in _names(model, ("k_proj",))]
    _assert_same_model(reference, model)


@pytest.mark.parametrize("change", ["damp", "seed", "samples", "weight", "spec", "group_member",
                                    "group_shrinks", "kv"])
def test_any_key_field_change_misses(tiny, tmp_path, change):
    recipe = _mixed()
    _, first = _calibrate(tiny, recipe, tmp_path)
    base, options = tiny, {}
    if change == "damp":  # on down_proj only (RTN overrides would inherit a changed default damp)
        recipe = _mixed(down={"modules": ["down_proj"], **GPTQ,
                              "weight_algo": {"kind": "gptq", "damp": 0.05}})
    elif change == "seed":
        options = {"seed": 1}
    elif change == "samples":
        options = {"samples": 2}
    elif change == "weight":
        base = copy.deepcopy(tiny)
        with torch.no_grad():
            base.model.layers[1].mlp.down_proj.weight[0, 0] += 1.0
    elif change == "spec":
        recipe = _mixed(o={"modules": ["o_proj"], **STATIC, "weight": "fp8_row"})
    elif change == "group_member":  # k_proj alone changes; the first matching override applies
        recipe = _recipe([{"modules": ["k_proj"], **AWQ, "weight": "mxfp4"}, *_mixed().overrides])
    elif change == "group_shrinks":  # k_proj leaves the AWQ group: q/v keep their specs, not their group
        recipe = _recipe([{"modules": ["k_proj"], **STATIC}, *_mixed().overrides])
    else:
        recipe = _recipe(_mixed().overrides, kv={"preset": "kivi2", "mode": "fakequant"})
    model, result = _calibrate(base, recipe, tmp_path, **options)
    misses = set(result["cache"]["misses"])
    expected = {
        "damp": set(_names(model, ("down_proj",))),
        "spec": set(_names(model, ("o_proj",))),
        "group_member": set(_names(model, ("q_proj", "k_proj", "v_proj"))),
        "group_shrinks": set(_names(model, ("q_proj", "k_proj", "v_proj"))),
    }.get(change, set(first["layers"]))  # windows, weights and KV all reach every layer's inputs
    assert misses == expected and set(result["cache"]["hits"]) == set(result["layers"]) - expected
    if change == "weight":
        assert "model.layers.1.mlp.down_proj" in misses


def test_uncacheable_layers_are_reported_and_still_exact(tiny, tmp_path):
    recipe = _recipe([
        {"modules": ["q_proj", "k_proj"], **AWQ, **GPTQ},  # fitted-diagonal GPTQ: transformed replay
        {"modules": ["v_proj"], **AWQ},  # same transform group as q/k, RTN
        {"modules": ["o_proj"], **STATIC, "activation": "fp8_tensor_delayed"},  # history observer
        {"modules": ["gate_proj"], "weight": {"format": "int2", "granularity": "row", "rounding": "sr"},
         "weight_algo": "rtn"},
        {"modules": ["up_proj"], **STATIC},
    ])
    for _ in range(2):
        reference, _ = _calibrate(tiny, recipe)  # afresh: the history observer changes on every forward
        model, result = _calibrate(tiny, recipe, tmp_path)
        uncacheable = result["cache"]["uncacheable"]
        cacheable = _names(model, ("up_proj", "down_proj"))
        assert set(uncacheable) == set(result["layers"]) - set(cacheable)
        for name in _names(model, ("q_proj", "k_proj")):
            assert "_hessian_replay" in uncacheable[name]
        for name in _names(model, ("v_proj",)):
            assert "jointly with uncacheable" in uncacheable[name]
        for name in _names(model, ("o_proj",)):
            assert "history observer" in uncacheable[name]
        for name in _names(model, ("gate_proj",)):
            assert "stochastic weight rounding" in uncacheable[name]
        assert result["full_model_forwards"] > 0
        _assert_same_model(reference, model)  # the second call loads up/down and refits the rest
    assert result["cache"]["hits"] == cacheable and result["cache"]["misses"] == []
    assert len(list(tmp_path.glob("layer-*.pt"))) == len(cacheable)


def test_sequential_calibration_is_uncacheable(tiny, tmp_path):
    recipe = _mixed()
    reference, plain = _calibrate(tiny, recipe, sequential=True)
    for _ in range(2):
        model, result = _calibrate(tiny, recipe, tmp_path, sequential=True)
        assert result["mode"] == "sequential" and result["cache"]["hits"] == []
        assert result["cache"]["misses"] == [] and list(result["cache"]["uncacheable"]) == result["layers"]
        assert all("sequential" in reason for reason in result["cache"]["uncacheable"].values())
        assert result["full_model_forwards"] == plain["full_model_forwards"]
        _assert_same_model(reference, model)
    assert not list(tmp_path.glob("layer-*.pt")) and not list(tmp_path.glob("inputs-*.json"))


@pytest.mark.parametrize("context", ["autocast_hit", "inference_mode_fill"])
def test_caller_autocast_or_inference_mode_is_uncacheable(tiny, tmp_path, context):
    if context == "autocast_hit":  # filled normally; a hit under autocast would load fp32-fitted states
        recipe = _mixed()
        _calibrate(tiny, recipe, tmp_path)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reference, plain = _calibrate(tiny, recipe)
            model, result = _calibrate(tiny, recipe, tmp_path)
        recorded, reason = result, "autocast"
    else:  # filled under inference mode, whose tensors hide that q/k/v and gate/up share an input
        recipe = _recipe(defaults=SMOOTH)
        filled_model = copy.deepcopy(tiny)
        with torch.inference_mode():
            recorded = calibrate(filled_model, recipe, cache_dir=tmp_path, **WINDOWS)
        reference, plain = _calibrate(tiny, recipe)
        model, result = _calibrate(tiny, recipe, tmp_path)
        reason = "inference_mode"
    assert result["groups"] == plain["groups"]
    _assert_same_model(reference, model)
    assert recorded["cache"]["hits"] == [] and recorded["cache"]["misses"] == []
    assert list(recorded["cache"]["uncacheable"]) == recorded["layers"]
    assert all(reason in text for text in recorded["cache"]["uncacheable"].values())
    if context == "inference_mode_fill":
        assert result["cache"]["hits"] == [] and result["cache"]["inputs"] == "written"


class _Doubling(nn.Module):  # a model whose class is not under torch, transformers or tricast
    def __init__(self) -> None:
        super().__init__()
        self.config = None
        self.embed = nn.Embedding(32, 16)
        self.proj = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids) * 2)


@pytest.mark.parametrize("change", ["module_swap", "forward_hook", "global_hook", "no_config",
                                    "patching_alone"])
def test_model_structure_and_hooks_reach_the_key(tiny, tmp_path, change):
    recipe, base, handle = _mixed(), copy.deepcopy(tiny), None
    if change == "patching_alone":  # guard: patching o_proj that the filling recipe left a Linear keeps hits
        _calibrate(tiny, _mixed(o={"modules": ["o_proj"], "skip": True}), tmp_path)
    elif change == "no_config":
        recipe, base = _recipe(defaults=STATIC), _Doubling()
        _calibrate(base, recipe, tmp_path)
    else:
        _calibrate(tiny, recipe, tmp_path)
    activation = type(base.model.layers[0].mlp.act_fn) if change != "no_config" else None
    if change == "module_swap":  # same class, weights and config; one inner module differs
        base.model.layers[0].mlp.act_fn = nn.ReLU()
    elif change == "forward_hook":
        base.model.layers[0].mlp.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
    elif change == "global_hook":
        handle = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda module, args: (args[0] * 2,) if isinstance(module, activation) else None)
    try:
        reference, _ = _calibrate(base, recipe)
        model, result = _calibrate(base, recipe, tmp_path)
        _assert_same(_state(reference), _state(model))
        if change == "no_config":
            _assert_same(reference(PROBE), model(PROBE), "output")
        else:
            _assert_same(reference(PROBE).logits, model(PROBE).logits, "logits")
    finally:
        if handle is not None:
            handle.remove()
    cache = result["cache"]
    if change == "patching_alone":
        assert cache["misses"] == _names(model, ("o_proj",)) and cache["uncacheable"] == {}
        assert set(cache["hits"]) == set(result["layers"]) - set(cache["misses"])
    elif change == "module_swap":
        assert cache["hits"] == [] and cache["misses"] == result["layers"]
    else:
        reason = {"forward_hook": "forward hooks", "global_hook": "global module forward hooks",
                  "no_config": "no config"}[change]
        assert cache["hits"] == [] and list(cache["uncacheable"]) == result["layers"]
        assert all(reason in text for text in cache["uncacheable"].values())


@pytest.mark.parametrize("tamper", ["edited", "re_signed"])
def test_input_sharing_record_is_verified(tiny, tmp_path, tamper):
    q, k = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"

    def only(*names):
        recipe = _recipe(defaults=SMOOTH)
        recipe.include = list(names)
        return recipe

    for name in (q, k):  # each alone: singleton entries keyed under the true record (k reads q's input)
        _calibrate(tiny, only(name), tmp_path)
    (record,) = tmp_path.glob("inputs-*.json")
    text = record.read_text()
    edited = text.replace(f'"{k}": "{q}"', f'"{k}": "{k}"')  # claim k reads its own input
    assert edited != text
    if tamper == "re_signed":  # a valid digest over the false content
        data = json.loads(edited)
        if "classes" in data:
            data["digest"] = _sha256_json(data["classes"])
        edited = json.dumps(data)
    record.write_text(edited)
    reference, plain = _calibrate(tiny, only(q, k))
    model, result = _calibrate(tiny, only(q, k), tmp_path)
    assert result["groups"] == plain["groups"] == [[q, k]]
    _assert_same_model(reference, model)
    assert result["cache"]["hits"] == [] and result["full_model_forwards"] == plain["full_model_forwards"]
    assert result["cache"]["inputs"] == "rewritten"
    refused = {k: "stored under a different input-sharing record",
               q: "stored under a different input-sharing record"} if tamper == "re_signed" else {}
    assert result["cache"]["corrupt"] == refused
    again_model, again = _calibrate(tiny, only(q, k), tmp_path)
    assert again["cache"]["hits"] == [q, k] and again["full_model_forwards"] == 0
    _assert_same_model(reference, again_model)


def _flip_storage_bit(path) -> None:
    """Flip one exponent bit of the first element of the largest tensor storage, in place in the zip."""
    with zipfile.ZipFile(path) as archive:
        info = max((item for item in archive.infolist() if "/data/" in item.filename),
                   key=lambda item: item.file_size)
    raw = bytearray(path.read_bytes())
    header = info.header_offset
    start = (header + 30 + int.from_bytes(raw[header + 26:header + 28], "little")
             + int.from_bytes(raw[header + 28:header + 30], "little"))
    raw[start + 1] ^= 0x40
    path.write_bytes(bytes(raw))


@pytest.mark.parametrize("damage", ["bit_flip", "truncated"])
def test_corrupt_entry_is_refitted_and_overwritten(tiny, tmp_path, damage):
    recipe, target = _mixed(), "model.layers.0.mlp.down_proj"
    _calibrate(tiny, recipe, tmp_path)
    (entry,) = [path for path in tmp_path.glob("layer-*.pt")
                if torch.load(path, weights_only=True)["name"] == target]
    if damage == "bit_flip":  # key and name unchanged; torch.load(weights_only=True) still accepts it
        _flip_storage_bit(entry)
        assert torch.load(entry, weights_only=True)["name"] == target
    else:
        entry.write_bytes(entry.read_bytes()[: entry.stat().st_size // 2])
    reference, _ = _calibrate(tiny, recipe)
    model, result = _calibrate(tiny, recipe, tmp_path)
    _assert_same_model(reference, model)
    assert list(result["cache"]["corrupt"]) == [target] and result["cache"]["misses"] == [target]
    assert set(result["cache"]["hits"]) == set(result["layers"]) - {target}
    problem = result["cache"]["corrupt"][target]
    assert problem == "content digest mismatch" if damage == "bit_flip" else problem.startswith("unreadable")
    again_model, again = _calibrate(tiny, recipe, tmp_path)  # the refitted entry overwrote the damaged one
    assert again["cache"]["hits"] == again["layers"] and again["cache"]["corrupt"] == {}
    assert again["full_model_forwards"] == 0
    _assert_same_model(reference, again_model)


def test_a_changed_instance_attribute_reaches_the_key(tiny, tmp_path):
    """Re-check of M3 (2026-11-03): an attention module's ``scaling`` changed after construction (same
    class, config, weights, hooks) misses every layer and matches a fresh calibration."""
    recipe = _mixed()
    _calibrate(tiny, recipe, tmp_path)
    base = copy.deepcopy(tiny)
    base.model.layers[0].self_attn.scaling *= 16
    reference, _ = _calibrate(base, recipe)
    model, result = _calibrate(base, recipe, tmp_path)
    _assert_same(_state(reference), _state(model))
    _assert_same(reference(PROBE).logits, model(PROBE).logits, "logits")
    assert result["cache"]["hits"] == [] and result["cache"]["misses"] == result["layers"]
