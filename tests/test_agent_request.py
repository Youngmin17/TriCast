"""Structured requests preserve explicit arithmetic and report every default."""

from __future__ import annotations

import copy
import json
from typing import get_args, get_type_hints

import pytest
from jsonschema import Draft202012Validator

from tricast.agent.request import (
    REQUEST_SCHEMA,
    EmulationRequest,
    RequestValidationError,
    api_schema,
    estimate_cost,
)
from tricast.formats import REGISTRY
from tricast.mma.spec import PRESETS, Algorithm, MMASpec
from tricast.quant.spec import (
    KV_PRESETS,
    SCHEMES,
    Granularity,
    KVMode,
    ObserverKind,
    ScaleMethod,
    TransformKind,
    WeightAlgo,
    ZeroPoint,
)
from tricast.recipe import load_recipe
from tricast.rounding import Rounding


def choice(kind: str, **values) -> dict:
    return {**dict.fromkeys(REQUEST_SCHEMA["$defs"][kind]["properties"]), **values}


def request_dict(**values) -> dict:
    return {
        "intent": "evaluate",
        "model": "Qwen/Qwen3-0.6B",
        "recipes": [choice("RecipeChoice", name="hopper", base="hopper_fp8_w8a8")],
        "sweep": None,
        "tasks": ["wikitext2_ppl"],
        "limits": {"max_windows": None, "limit": None},
        "assumptions": [],
        "questions": [],
        "topic": None,
        **values,
    }


def test_strict_objects_recursively() -> None:
    def visit(value: object) -> None:
        if isinstance(value, dict):
            kind = value.get("type", [])
            if kind == "object" or isinstance(kind, list) and "object" in kind:
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(REQUEST_SCHEMA)


def test_schema_enums_match_specs() -> None:
    defs = REQUEST_SCHEMA["$defs"]
    quant = defs["QuantChoice"]["properties"]
    mma = defs["MMAChoice"]["properties"]
    recipe = defs["RecipeChoice"]["properties"]
    expected = [
        (quant["scheme"], SCHEMES),
        (quant["granularity"], get_args(Granularity)),
        (quant["scale_method"], get_args(ScaleMethod)),
        (quant["zero_point"], get_args(ZeroPoint)),
        (quant["observer"], get_args(ObserverKind)),
        (quant["rounding"], [mode.value for mode in Rounding]),
        (mma["preset"], PRESETS),
        (mma["algorithm"], get_args(Algorithm)),
        (mma["c_mode"], get_args(get_type_hints(MMASpec)["c_mode"])),
        (recipe["transform"], get_args(TransformKind)),
        (recipe["weight_algo"], get_args(WeightAlgo)),
    ]
    for schema, values in expected:
        assert set(schema["enum"]) == {*values, None}


@pytest.mark.parametrize('field', ['format', 'scale_format'])
def test_format_schema_accepts_registry_and_custom_format_grammar(field: str) -> None:
    schema = REQUEST_SCHEMA['$defs']['QuantChoice']['properties'][field]
    for name in [*REGISTRY, 'int3', 'uint3', 'e3m2', 'e4m3:fn', None]:
        Draft202012Validator(schema).validate(name)
        entry = choice('RecipeChoice', name='custom', weight=choice('QuantChoice', **{field: name}))
        request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
        assert request.recipes[0]['weight'][field] == name


@pytest.mark.parametrize('field', ['format', 'scale_format'])
@pytest.mark.parametrize('name', ['e0m3', 'int0', 'uint0', 'eXmY', 'not_a_format'])
def test_invalid_custom_format_is_rejected_with_its_request_path(field: str, name: str) -> None:
    entry = choice('RecipeChoice', name='custom', weight=choice('QuantChoice', **{field: name}))
    with pytest.raises(RequestValidationError, match=rf'recipes\[0\].weight.{field}'):
        EmulationRequest.from_dict(request_dict(recipes=[entry]))


def test_round_trip_and_input_isolation() -> None:
    data = request_dict()
    request = EmulationRequest.from_dict(data)
    assert request.to_dict() == data
    data["recipes"][0]["name"] = "changed"
    result = request.to_dict()
    result["recipes"][0]["name"] = "also_changed"
    assert request.recipes[0]["name"] == "hopper"


@pytest.mark.parametrize(("field", "value"), [("model", 7), ("intent", "guess"), ("unexpected", True)])
def test_schema_error_has_path(field: str, value: object) -> None:
    with pytest.raises(RequestValidationError) as caught:
        EmulationRequest.from_dict(request_dict(**{field: value}))
    assert caught.value.errors and field in str(caught.value)


def test_missing_key_rejected() -> None:
    data = request_dict()
    del data["topic"]
    with pytest.raises(RequestValidationError, match="topic"):
        EmulationRequest.from_dict(data)


def test_new_recipe_and_nulls_preserved() -> None:
    entry = choice(
        "RecipeChoice",
        name="custom",
        weight=choice("QuantChoice", scheme="mxfp4"),
        activation=choice("QuantChoice", scheme="mxfp4"),
        mma=choice("MMAChoice", preset="nvidia_blackwell_fp4"),
    )
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    (recipe,) = request.to_recipes()
    assert recipe.defaults.weight == SCHEMES["mxfp4"]
    assert recipe.defaults.activation == SCHEMES["mxfp4"]
    assert recipe.defaults.mma == PRESETS["nvidia_blackwell_fp4"]
    assert request.to_dict()["recipes"] == [entry]
    assert any("transform" in item for item in request.assumptions)


def test_partial_base_overrides_and_provenance_reset() -> None:
    entry = choice(
        "RecipeChoice",
        name="custom",
        base="hopper_fp8_w8a8",
        weight=choice("QuantChoice", rounding="rtz"),
        mma=choice("MMAChoice", f_bits=11),
    )
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    (recipe,) = request.to_recipes()
    base = load_recipe("hopper_fp8_w8a8")
    assert recipe.defaults.weight.format == base.defaults.weight.format
    assert recipe.defaults.weight.rounding is Rounding.RTZ
    assert recipe.defaults.mma.f_bits == 11
    assert recipe.defaults.mma.chunk_size == 32
    assert recipe.defaults.mma.name == recipe.defaults.mma.provenance == ""


def test_new_selectors_replace_old_defaults() -> None:
    entry = choice(
        "RecipeChoice",
        name="new",
        base="hopper_fp8_w8a8",
        weight=choice("QuantChoice", scheme="bf16"),
        mma=choice("MMAChoice", preset="nvidia_blackwell_fp4"),
    )
    (recipe,) = EmulationRequest.from_dict(request_dict(recipes=[entry])).to_recipes()
    assert recipe.defaults.weight == SCHEMES["bf16"]
    assert recipe.defaults.mma == PRESETS["nvidia_blackwell_fp4"]


@pytest.mark.parametrize("component", ["weight", "activation"])
def test_quant_error_keeps_request_path(component: str) -> None:
    entry = choice("RecipeChoice", name="bad", **{component: choice("QuantChoice", granularity="group")})
    with pytest.raises(RequestValidationError, match=rf"recipes\[0\].{component}"):
        EmulationRequest.from_dict(request_dict(recipes=[entry])).to_recipes()


def test_errors_accumulate_across_recipes() -> None:
    entries = [
        choice("RecipeChoice", name="bad1", base="not_a_recipe"),
        choice("RecipeChoice", name="bad2", mma=choice("MMAChoice", algorithm="gdfs", group_size=3)),
    ]
    with pytest.raises(RequestValidationError) as caught:
        EmulationRequest.from_dict(request_dict(recipes=entries)).to_recipes()
    assert len(caught.value.errors) == 2
    assert "recipes[0].base" in caught.value.errors[0]
    assert "recipes[1].mma" in caught.value.errors[1]


def test_empty_evaluation_not_silently_completed() -> None:
    with pytest.raises(RequestValidationError, match="recipes"):
        EmulationRequest.from_dict(request_dict(recipes=[])).to_recipes()
    for intent in ("explain", "inspect"):
        request = EmulationRequest.from_dict(request_dict(intent=intent, recipes=[], tasks=[]))
        assert request.to_recipes() == []


def test_mma_defaults_recorded_not_inserted_in_request() -> None:
    entry = choice("RecipeChoice", name="custom", mma=choice("MMAChoice", f_bits=13))
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    (recipe,) = request.to_recipes()
    assert recipe.defaults.mma.f_bits == 13
    assert request.recipes[0]["mma"]["chunk_size"] is None
    assert any("chunk_size" in item and "32" in item for item in request.assumptions)
    before = list(request.assumptions)
    request.to_recipes()
    assert request.assumptions == before


def test_sweep_expanded_and_original_preserved() -> None:
    request = EmulationRequest.from_dict(
        request_dict(intent="sweep", sweep={"axis": "f_bits", "values": [7, 13]})
    )
    config = request.to_run_config()
    assert len(config["recipes"]) == 2
    recipes = [load_recipe(data) for data in config["recipes"]]
    assert [item.defaults.mma.f_bits for item in recipes] == [7, 13]
    assert len({item.name for item in recipes}) == 2
    assert all(item.defaults.mma.provenance == "" for item in recipes)
    assert request.recipes[0]["mma"] is None


def test_sweep_overrides_explicit_base_choice() -> None:
    data = request_dict(intent="sweep", sweep={"axis": "chunk_size", "values": [8, 16]})
    data["recipes"][0]["mma"] = choice("MMAChoice", f_bits=10)
    recipes = [load_recipe(item) for item in EmulationRequest.from_dict(data).to_run_config()["recipes"]]
    assert [(item.defaults.mma.f_bits, item.defaults.mma.chunk_size) for item in recipes] == [
        (10, 8),
        (10, 16),
    ]


@pytest.mark.parametrize(
    "sweep",
    [
        {"axis": "f_bits", "values": ["13"]},
        {"axis": "c_mode", "values": [13]},
        {"axis": "c_mode", "values": ["unknown"]},
        {"axis": "f_bits", "values": []},
    ],
)
def test_invalid_sweep_value_rejected(sweep: dict) -> None:
    with pytest.raises(RequestValidationError, match="sweep"):
        EmulationRequest.from_dict(request_dict(intent="sweep", sweep=sweep)).to_run_config()


def test_invalid_sweep_precision_has_value_path() -> None:
    with pytest.raises(RequestValidationError, match=r"sweep.values\[1\]"):
        EmulationRequest.from_dict(
            request_dict(intent="sweep", sweep={"axis": "f_bits", "values": [13, 99]})
        ).to_run_config()


def test_task_mapping_and_limits() -> None:
    request = EmulationRequest.from_dict(
        request_dict(tasks=["wikitext2_ppl", "hellaswag", "piqa"], limits={"max_windows": 2, "limit": 0.25})
    )
    config = request.to_run_config()
    assert config["tasks"]["ppl"]["dataset"] == "wikitext2"
    assert config["tasks"]["ppl"]["max_windows"] == 2
    assert config["tasks"]["lm_eval"] == {"tasks": ["hellaswag", "piqa"], "limit": 0.25}
    json.dumps(config, allow_nan=False)


def test_questions_prevent_run_config() -> None:
    with pytest.raises(RequestValidationError, match="questions"):
        EmulationRequest.from_dict(request_dict(questions=["어떤 형식인가요?"])).to_run_config()


def test_cost_is_monotone_and_explicitly_estimated() -> None:
    request = EmulationRequest.from_dict(request_dict())
    base = request.estimate_cost(model_parameters=100, tokens=10, throughput_macs_per_second=50)
    assert base["label"] == "추정" and base["is_estimate"] is True
    assert base["emulated_macs"] == 1000 and base["estimated_seconds"] == 20
    larger = request.estimate_cost(model_parameters=200, tokens=20, throughput_macs_per_second=50)
    assert larger["emulated_macs"] == 4 * base["emulated_macs"]
    data = request_dict()
    data["recipes"].append({**copy.deepcopy(data["recipes"][0]), "name": "second"})
    doubled = EmulationRequest.from_dict(data).estimate_cost(model_parameters=100, tokens=10)
    assert doubled["emulated_macs"] == 2 * base["emulated_macs"]


def test_unknown_cost_is_not_fabricated() -> None:
    cost = estimate_cost({"model": "someone/Mystery-7B", "recipes": ["hopper_fp8_w8a8"], "tasks": {}})
    assert cost["model_parameters"] is None
    assert cost["emulated_macs"] is None and cost["estimated_seconds"] is None
    assert cost["assumptions"]


def test_window_token_bound_and_unknown_lm_eval_tokens() -> None:
    config = {"recipes": ["hopper_fp8_w8a8"], "tasks": {"ppl": {"max_windows": 2}}}
    cost = estimate_cost(config, model_parameters=100)
    assert cost["tokens_per_recipe"] == 4096
    assert any("upper bound" in item for item in cost["assumptions"])
    config["tasks"]["lm_eval"] = {"tasks": ["piqa"], "limit": 2}
    assert estimate_cost(config, model_parameters=100)["tokens_per_recipe"] is None


@pytest.mark.parametrize(
    "kwargs",
    [{"throughput_macs_per_second": 0}, {"tokens": -1}, {"model_parameters": float("inf")}, {"tokens": True}],
)
def test_invalid_estimate_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        estimate_cost({"recipes": []}, **kwargs)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_all_schemes_resolve_without_changing_arithmetic(scheme: str) -> None:
    entry = choice("RecipeChoice", name="selected", weight=choice("QuantChoice", scheme=scheme))
    (recipe,) = EmulationRequest.from_dict(request_dict(recipes=[entry])).to_recipes()
    assert recipe.defaults.weight == SCHEMES[scheme]


@pytest.mark.parametrize("preset", PRESETS)
def test_all_presets_preserve_datapath_and_provenance(preset: str) -> None:
    entry = choice("RecipeChoice", name="selected", mma=choice("MMAChoice", preset=preset))
    (recipe,) = EmulationRequest.from_dict(request_dict(recipes=[entry])).to_recipes()
    assert recipe.defaults.mma == PRESETS[preset]
    assert recipe.defaults.mma.provenance == PRESETS[preset].provenance


def test_compare_requires_multiple_alternatives() -> None:
    with pytest.raises(RequestValidationError, match="compare needs at least two"):
        EmulationRequest.from_dict(request_dict(intent="compare")).to_recipes()
    request = EmulationRequest.from_dict(
        request_dict(intent="compare", sweep={"axis": "f_bits", "values": [7, 13]})
    )
    assert len(request.to_run_config()["recipes"]) == 2


def test_observer_defaults_are_not_hidden() -> None:
    entry = choice(
        "RecipeChoice", name="observed", activation=choice("QuantChoice", scheme="fp8_tensor", observer="ema")
    )
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    request.to_recipes()
    assert any(
        '"observer"' in item and '"decay": 0.99' in item and '"history_len": 16' in item
        for item in request.assumptions
    )


def test_unlimited_evaluations_are_recorded_as_assumptions() -> None:
    request = EmulationRequest.from_dict(request_dict(tasks=["wikitext2_ppl", "piqa"]))
    request.to_run_config()
    assert any("limits.max_windows" in item and "no window limit" in item for item in request.assumptions)
    assert any("limits.limit" in item and "no sample limit" in item for item in request.assumptions)
    before = list(request.assumptions)
    request.to_run_config()
    assert request.assumptions == before


@pytest.mark.parametrize("model", ["", " ", " org/model", "org/model ", "org/model name"])
def test_empty_or_whitespace_model_rejected(model: str) -> None:
    with pytest.raises(RequestValidationError, match="model:"):
        EmulationRequest.from_dict(request_dict(model=model))


@pytest.mark.parametrize(
    ("component", "field"),
    [
        ("mma", "f_bits"),
        ("mma", "chunk_size"),
        ("mma", "g_bits"),
        ("mma", "group_size"),
        ("mma", "promote_interval"),
        ("weight", "group_size"),
        ("activation", "group_size"),
        ("limits", "max_windows"),
    ],
)
def test_integral_float_rejected_before_integer_arithmetic(component: str, field: str) -> None:
    data = request_dict()
    if component == "limits":
        data["limits"][field] = 13.0
        path = rf"limits\.{field}"
    else:
        kind = "MMAChoice" if component == "mma" else "QuantChoice"
        data["recipes"][0][component] = choice(kind, **{field: 13.0})
        path = rf"recipes\[0\]\.{component}\.{field}"
    with pytest.raises(RequestValidationError, match=path):
        EmulationRequest.from_dict(data)


def test_api_schema_removes_constraints_recursively_without_mutating_local_schema() -> None:
    forbidden = {
        'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
        'minLength', 'maxLength', 'pattern', 'minItems', 'maxItems', 'uniqueItems',
        'contains', 'minContains', 'maxContains', 'prefixItems', 'unevaluatedItems',
        'minProperties', 'maxProperties', 'propertyNames', 'patternProperties',
    }
    original = copy.deepcopy(REQUEST_SCHEMA)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            if 'object' in value.get('type', []):
                assert value['additionalProperties'] is False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(api_schema())
    assert REQUEST_SCHEMA == original
    assert REQUEST_SCHEMA['properties']['limits']['properties']['max_windows']['minimum'] == 1
    with pytest.raises(RequestValidationError, match='max_windows'):
        EmulationRequest.from_dict(request_dict(limits={'max_windows': 0, 'limit': None}))


def test_api_projection_strips_complex_constraints_but_preserves_property_names() -> None:
    schema = {
        'type': 'object', 'required': ['minimum', 'values'], 'additionalProperties': True,
        'properties': {
            'minimum': {'type': 'integer', 'minimum': 1, 'multipleOf': 2},
            'values': {
                'type': 'array', 'items': {'type': 'string', 'minLength': 2, 'pattern': '^a'},
                'contains': {'const': 'a'}, 'prefixItems': [{'const': 'b'}],
                'uniqueItems': True, 'maxItems': 4,
            },
        },
    }
    projected = api_schema(schema)
    assert projected['additionalProperties'] is False
    assert projected['properties']['minimum'] == {'type': 'integer'}
    assert projected['properties']['values'] == {'type': 'array', 'items': {'type': 'string'}}


@pytest.mark.parametrize('schema', [
    {'$ref': '#'},
    {'$ref': '#/$defs/A', '$defs': {'A': {'$ref': '#/$defs/B'}, 'B': {'$ref': '#/$defs/A'}}},
])
def test_api_schema_rejects_recursive_references(schema: dict) -> None:
    with pytest.raises(ValueError, match='recursive'):
        api_schema(schema)


def test_api_schema_preserves_nonrecursive_shared_references() -> None:
    schema = {
        'type': 'object', 'properties': {'a': {'$ref': '#/$defs/A'}, 'b': {'$ref': '#/$defs/A'}},
        'required': ['a', 'b'], 'additionalProperties': False,
        '$defs': {'A': {'type': 'integer', 'minimum': 0}},
    }
    assert api_schema(schema)['$defs']['A'] == {'type': 'integer'}


def test_api_schema_does_not_interpret_property_names_or_constants_as_references() -> None:
    schema = {
        'type': 'object', 'properties': {
            '$ref': {'type': 'string'},
            'constant': {'const': {'$ref': 'literal-data'}},
        },
    }
    projected = api_schema(schema)
    assert projected['properties'] == schema['properties']


def test_new_schema_enums_match_kv_and_scale_specs() -> None:
    defs = REQUEST_SCHEMA['$defs']
    assert set(defs['KVChoice']['properties']['preset']['enum']) == {*KV_PRESETS, None}
    assert set(defs['KVChoice']['properties']['mode']['enum']) == {*get_args(KVMode), None}
    assert set(defs['QuantChoice']['properties']['scale_rounding']['enum']) == {
        *(mode.value for mode in Rounding), None,
    }


@pytest.mark.parametrize('format_name', ['int3', 'uint3', 'e3m2', 'e4m3:fn'])
def test_custom_format_syntax_is_not_erased(format_name: str) -> None:
    entry = choice('RecipeChoice', name='custom', weight=choice('QuantChoice', format=format_name))
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    (recipe,) = request.to_recipes()
    from tricast.formats import get_format
    assert recipe.defaults.weight.format == get_format(format_name)
    assert request.recipes[0]['weight']['format'] == format_name


def test_scale_rounding_does_not_change_element_rounding() -> None:
    entry = choice(
        'RecipeChoice', name='scale_only', weight=choice('QuantChoice', scheme='nvfp4', scale_rounding='rtz')
    )
    (recipe,) = EmulationRequest.from_dict(request_dict(recipes=[entry])).to_recipes()
    assert recipe.defaults.weight.scale.rounding is Rounding.RTZ
    assert recipe.defaults.weight.rounding is Rounding.RNE


@pytest.mark.parametrize('preset', KV_PRESETS)
def test_kv_choice_preserves_explicit_options_in_recipe_data(preset: str) -> None:
    entry = choice(
        'RecipeChoice', name='cache', kv=choice('KVChoice', preset=preset, mode='fakequant', residual=64)
    )
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    data = request._recipe_data(entry, 0)
    assert data['kv'] == {'preset': preset, 'mode': 'fakequant', 'residual': 64}
    (recipe,) = request.to_recipes()
    assert recipe.kv.key == KV_PRESETS[preset].key
    assert recipe.kv.value == KV_PRESETS[preset].value
    assert recipe.kv.mode == 'fakequant'
    assert recipe.kv.residual == 64
    assert recipe.spec_for('model.layers.0.self_attn.q_proj') is None


def test_layer_selection_and_calibration_preserve_explicit_values() -> None:
    calibration = choice(
        'CalibrationChoice', dataset='wikitext2', samples=2, seqlen=8, seed=42, sequential=True
    )
    entry = choice('RecipeChoice', name='selected', base='hopper_fp8_w8a8', layers='0-3',
                   modules=['q_proj', 'v_proj'], calibration=calibration)
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    data = request._recipe_data(entry, 0)
    assert data['calibration'] == calibration
    assert data['overrides'] == [
        {'layers': '0-3', 'modules': ['q_proj', 'v_proj']}, {'match': '*', 'skip': True},
    ]
    assert any('skip all others' in item for item in request.assumptions)
    (recipe,) = request.to_recipes()
    assert recipe.spec_for('model.layers.0.self_attn.q_proj') is not None
    assert recipe.spec_for('model.layers.3.self_attn.v_proj') is not None
    assert recipe.spec_for('model.layers.4.self_attn.q_proj') is None
    assert recipe.spec_for('model.layers.1.self_attn.k_proj') is None
    assert recipe.calibration['sequential'] is True


def test_skip_requires_and_preserves_selection() -> None:
    entry = choice('RecipeChoice', name='selected', base='hopper_fp8_w8a8', layers='0,27', skip=True)
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    assert request._recipe_data(entry, 0)['overrides'] == [{'layers': '0,27', 'skip': True}]
    (recipe,) = request.to_recipes()
    assert recipe.spec_for('model.layers.0.self_attn.q_proj') is None
    assert recipe.spec_for('model.layers.1.self_attn.q_proj') is not None
    entry['layers'] = None
    with pytest.raises(ValueError, match='skip: requires'):
        request._recipe_data(entry, 0)


@pytest.mark.parametrize(('kind', 'field', 'value'), [
    ('KVChoice', 'residual', -1), ('KVChoice', 'residual', 1.0),
    ('CalibrationChoice', 'samples', 0), ('CalibrationChoice', 'samples', 1.0),
    ('CalibrationChoice', 'seqlen', 0), ('CalibrationChoice', 'seed', 0.0),
    ('CalibrationChoice', 'sequential', 'true'),
])
def test_new_options_keep_full_local_validation(kind: str, field: str, value: object) -> None:
    key = 'kv' if kind == 'KVChoice' else 'calibration'
    entry = choice(
        'RecipeChoice', name='selected', base='hopper_fp8_w8a8', **{key: choice(kind, **{field: value})}
    )
    with pytest.raises(RequestValidationError, match=field):
        EmulationRequest.from_dict(request_dict(recipes=[entry]))


def test_report_plan_needs_no_evaluation_task_or_invented_input() -> None:
    request = EmulationRequest.from_dict(request_dict(intent='report', tasks=[]))
    config = request.to_run_config()
    assert config['model'] == request.model
    assert len(config['recipes']) == 1
    assert config['tasks'] == {'report': {}}


def test_kv_layer_selection_never_changes_unrequested_linear_arithmetic() -> None:
    entry = choice('RecipeChoice', name='cache', kv=choice('KVChoice', preset='kivi2'), layers='0-3')
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    (recipe,) = request.to_recipes()
    assert recipe.kv_layers == '0-3'
    assert recipe.spec_for('model.layers.0.self_attn.q_proj') is None
    assert recipe.spec_for('model.layers.4.self_attn.q_proj') is None


@pytest.mark.parametrize('selectors', [{'modules': ['q_proj']}, {'layers': '0-3', 'skip': True}])
def test_unsupported_kv_selectors_are_not_ignored(selectors: dict) -> None:
    entry = choice('RecipeChoice', name='cache', kv=choice('KVChoice', preset='kivi2'), **selectors)
    with pytest.raises(RequestValidationError, match='modules|skip'):
        EmulationRequest.from_dict(request_dict(recipes=[entry])).to_recipes()


def test_report_sweep_is_rejected_instead_of_silently_dropped() -> None:
    request = EmulationRequest.from_dict(request_dict(
        intent='report', tasks=[], sweep={'axis': 'f_bits', 'values': [7, 13]},
    ))
    with pytest.raises(RequestValidationError, match='sweep'):
        request.to_run_config()


def test_implicit_and_partial_calibration_defaults_are_recorded() -> None:
    for calibration in (None, choice('CalibrationChoice', samples=2, sequential=True)):
        entry = choice('RecipeChoice', name='gptq', weight=choice('QuantChoice', scheme='int4_g128'),
                       weight_algo='gptq', calibration=calibration)
        request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
        (recipe,) = request.to_recipes()
        effective = recipe.calibration_options
        assert effective['dataset'] == 'wikitext2'
        assert effective['seed'] == 0
        assert effective['seqlen'] == 2048
        assert any('calibration' in item and 'wikitext2' in item and '2048' in item
                   for item in request.assumptions)


@pytest.mark.parametrize('axis', ['g_bits', 'group_size'])
@pytest.mark.parametrize('algorithm', [None, 'cofda', 'fp32_fma', 'fp64', 'int_exact'])
def test_non_gdfs_group_settings_rejected_by_shared_validator(axis: str, algorithm: str | None) -> None:
    entry = choice('RecipeChoice', name='explicit', weight=choice('QuantChoice', scheme='mxfp4'),
                   mma=choice('MMAChoice', algorithm=algorithm, **{axis: 6 if axis == 'g_bits' else 4}))
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    with pytest.raises(RequestValidationError, match=rf'recipes\[0\].mma.{axis}.*gdfs'):
        request.to_recipes()


@pytest.mark.parametrize('axis,values', [('g_bits', [5, 6]), ('group_size', [4, 8])])
def test_non_gdfs_sweep_is_rejected_before_live_parser_acceptance(axis: str, values: list[int]) -> None:
    from tricast.agent.parser import _validate

    data = request_dict(intent='sweep', sweep={'axis': axis, 'values': values})
    with pytest.raises(RequestValidationError, match='sweep.axis.*gdfs'):
        _validate(data)
    with pytest.raises(RequestValidationError, match='sweep.axis.*gdfs'):
        EmulationRequest.from_dict(data).to_run_config()


@pytest.mark.parametrize('axis,values', [('g_bits', [5, 6]), ('group_size', [8, 16])])
def test_explicit_gdfs_sweeps_preserve_every_point(axis: str, values: list[int]) -> None:
    entry = choice('RecipeChoice', name='gdfs', weight=choice('QuantChoice', scheme='mxfp4'),
                   mma=choice('MMAChoice', algorithm='gdfs', g_bits=6, group_size=8))
    request = EmulationRequest.from_dict(request_dict(
        recipes=[entry], intent='sweep', sweep={'axis': axis, 'values': values},
    ))
    specs = [load_recipe(item).defaults.mma for item in request.to_run_config()['recipes']]
    assert [getattr(spec, axis) for spec in specs] == values
    assert all(spec.algorithm == 'gdfs' for spec in specs)


@pytest.mark.parametrize('alias,canonical', [('fp8', 'fp8_e4m3'), ('fp4', 'fp4_e2m1'),
                                             ('float', 'fp32'), ('half', 'fp16')])
@pytest.mark.parametrize('operand', ['weight', 'activation'])
@pytest.mark.parametrize('field', ['format', 'scale_format'])
def test_ambiguous_format_aliases_have_path_qualified_assumptions(
    alias: str, canonical: str, operand: str, field: str,
) -> None:
    quant = choice('QuantChoice', format='fp32')
    quant[field] = alias
    entry = choice('RecipeChoice', name='alias', **{operand: quant})
    request = EmulationRequest.from_dict(request_dict(recipes=[entry]))
    message = f'recipes[0].{operand}.{field}: {alias} -> {canonical}'
    assert any(message in item for item in request.assumptions)
    before = list(request.assumptions)
    assert EmulationRequest.from_dict(request.to_dict()).assumptions == before
    assert request.recipes[0][operand][field] == alias


@pytest.mark.parametrize('text,canonical', [('format=fp8 평가', 'fp8_e4m3'),
                                           ('nvfp4 scale.format=fp8', 'fp8_e4m3')])
def test_scoped_offline_aliases_record_canonical_choice(text: str, canonical: str) -> None:
    from tricast.agent.parser import parse_request

    parsed = parse_request(text, llm='offline')
    assert not parsed.errors and not parsed.request.questions
    assert any('fp8 -> ' + canonical in item for item in parsed.request.assumptions)


def test_evaluation_options_do_not_change_calibration_or_claim_defaults() -> None:
    request = EmulationRequest.from_dict(request_dict(evaluation={'seqlen': 512, 'seed': 7}))
    config = request.to_run_config()
    assert config['tasks']['ppl']['seqlen'] == 512 and config['seed'] == 7
    assert config['recipes'][0].get('calibration') is None
    assert not any('seqlen=2048' in item or 'seed=42' in item for item in request.assumptions)
    assert request.to_dict()['evaluation'] == {'seqlen': 512, 'seed': 7}


@pytest.mark.parametrize('evaluation', [{'seqlen': 1, 'seed': None}, {'seqlen': 512.0, 'seed': None},
                                         {'seqlen': None, 'seed': True}, {'seqlen': None, 'seed': -1},
                                         {'seqlen': None, 'seed': 2**32}])
def test_evaluation_options_are_validated_at_request_boundary(evaluation: dict) -> None:
    with pytest.raises(RequestValidationError, match='evaluation'):
        EmulationRequest.from_dict(request_dict(evaluation=evaluation))


def test_lm_eval_seqlen_cannot_be_silently_discarded() -> None:
    request = EmulationRequest.from_dict(request_dict(tasks=['hellaswag'],
                                                      evaluation={'seqlen': 512, 'seed': 7}))
    with pytest.raises(RequestValidationError, match='evaluation.seqlen'):
        request.to_run_config()


def test_report_cannot_silently_discard_evaluation_tasks() -> None:
    request = EmulationRequest.from_dict(request_dict(intent='report', tasks=['hellaswag']))
    with pytest.raises(RequestValidationError, match='tasks'):
        request.to_run_config()


@pytest.mark.parametrize('inputs', [{'texts': ['text'], 'input_ids': [[1, 2]]},
                                    {'texts': ['   '], 'input_ids': None},
                                    {'texts': None, 'input_ids': [[1, 2], [1, 2, 3]]},
                                    {'texts': None, 'input_ids': [[1.0, 2]]}])
def test_report_inputs_are_validated_before_execution(inputs: dict) -> None:
    with pytest.raises(RequestValidationError, match='report_inputs'):
        EmulationRequest.from_dict(request_dict(intent='report', tasks=[], report_inputs=inputs))


@pytest.mark.parametrize('use_texts', [False, True])
def test_agent_report_executes_explicit_inputs_through_live_schema(
    monkeypatch: pytest.MonkeyPatch, use_texts: bool,
) -> None:
    from types import SimpleNamespace

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tricast.agent.loop import run_agent
    from tricast.eval import envinfo, runner

    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=1, max_position_embeddings=16)).eval()
    class Tokenizer:
        def __call__(self, text: str, *, return_tensors: str) -> dict:
            assert text == 'explicit report text' and return_tensors == 'pt'
            return {'input_ids': torch.tensor([[1, 2, 3, 4]])}

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(runner, '_load_model', lambda *args: (model, Tokenizer()))
    monkeypatch.setattr(envinfo, 'capture_env', lambda **kw: {'git_sha': 'fixture', **kw.get('extra', {})})
    inputs = {'texts': ['explicit report text'] if use_texts else None,
              'input_ids': None if use_texts else [[1, 2, 3, 4]]}
    data = request_dict(intent='report', tasks=[], report_inputs=inputs,
                        evaluation={'seqlen': 4, 'seed': 42},
                        recipes=[choice('RecipeChoice', name='report', base='fp64_reference')])
    replies = iter([json.dumps(data), '명시된 입력의 레이어 분석을 실행했습니다.'])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        stop_reason='end_turn', content=[{'type': 'text', 'text': next(replies)}],
    )))
    report = run_agent('report using supplied inputs', execute=True, llm='anthropic', client=client)
    assert not report.errors and report.numbers_traceable
    assert report.plan['tasks']['report']['seqlen'] == 4
    assert report.plan['tasks']['report']['texts' if use_texts else 'input_ids'] == inputs[
        'texts' if use_texts else 'input_ids']
    record = report.results[0]
    assert record['metrics']['report']['layers'] and record['env']
    assert record['metrics']['report']['model']['ppl_reference'] > 0


def test_cost_counts_already_expanded_sweep_recipes() -> None:
    request = EmulationRequest.from_dict(request_dict(intent='sweep',
                                                      sweep={'axis': 'f_bits', 'values': [7, 13]}))
    cost = request.estimate_cost(model_parameters=100, tokens=10)
    assert cost['recipe_count'] == 2 and cost['emulated_macs'] == 2000


@pytest.mark.parametrize('intent', ['explain', 'inspect'])
def test_explanation_cannot_silently_discard_evaluation_options_or_recipe_tasks(intent: str) -> None:
    from tricast.agent.parser import _validate

    with pytest.raises(RequestValidationError, match='evaluation'):
        _validate(request_dict(intent=intent, recipes=[], tasks=[],
                               evaluation={'seqlen': 512, 'seed': 7}))
    with pytest.raises(RequestValidationError, match='tasks'):
        _validate(request_dict(intent=intent, tasks=['hellaswag']))
    request = _validate(request_dict(intent=intent, recipes=[], tasks=['hellaswag']))
    assert request.tasks == ['hellaswag']
    assert any('no evaluation will be executed' in item for item in request.assumptions)
