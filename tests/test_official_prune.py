"""Official pruning code (wanda repository, pinned) called unchanged on transformers>=4.48 decoders.

Covers the pin/clean-tree refusal, isolated imports, the 2023-call shim (bit identity), the pruning
structure each official method produces, firing evidence, and the recipe / eval-runner hook-in. Tiny
models are built from configs and data comes from a loader override or faked ``load_dataset`` (no network).
"""

from __future__ import annotations

import copy
import functools
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("datasets")  # official lib/data.py imports it

from tricast.official import compat, repos  # noqa: E402
from tricast.official import prune as official_prune  # noqa: E402
from tricast.recipe import load_recipe  # noqa: E402

SEQLEN, NSAMPLES = 32, 4
KINDS = ("llama", "qwen3")
LINEARS = 7  # q, k, v, o, gate, up, down


def _tiny(kind: str) -> nn.Module:
    torch.manual_seed(0)
    common = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "intermediate_size": 128, "vocab_size": 256, "max_position_embeddings": SEQLEN}
    if kind == "llama":
        return transformers.LlamaForCausalLM(transformers.LlamaConfig(**common)).eval()
    return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(head_dim=16, **common)).eval()


def _loader(n: int = NSAMPLES, length: int = SEQLEN) -> list:
    generator = torch.Generator().manual_seed(1)
    return [(torch.randint(4, 256, (1, length), generator=generator), None) for _ in range(n)]


def _linears(model: nn.Module) -> list[torch.Tensor]:
    return [module.weight.data for layer in model.model.layers
            for module in layer.modules() if type(module) is nn.Linear]


def _count_calls(monkeypatch, dotted: str, owner: str, method: str) -> dict:
    """Count calls of a method of an official class (the module object official lib.prune uses)."""
    target = getattr(repos.load_module("wanda", dotted), owner)
    original = getattr(target, method)
    counter = {"calls": 0}

    def counted(self, *args, **kwargs):
        counter["calls"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(target, method, counted)
    return counter


def _tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i + 4 for i in range(252)})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )


def _head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def test_ensure_refuses_other_commit_and_edited_tree(tmp_path, monkeypatch):
    pinned = repos.REPOS["wanda"]
    source = repos.ensure("wanda")
    assert _head(source) == pinned.commit
    # An absent clone is cloned and checked out at the pin (here from the local clone: no network).
    monkeypatch.setenv("TRICAST_OFFICIAL_DIR", str(tmp_path))
    monkeypatch.setitem(repos.REPOS, "wanda", replace(pinned, url=str(source)))
    clone = repos.ensure("wanda")
    assert clone == tmp_path / "wanda"
    assert _head(clone) == pinned.commit
    (clone / "untracked.txt").write_text("not part of the pinned tree\n", encoding="utf-8")
    assert repos.ensure("wanda") == clone
    monkeypatch.setitem(repos.REPOS, "wanda", replace(pinned, url=str(source), commit="0" * 40))
    with pytest.raises(RuntimeError, match="not the pinned commit"):
        repos.ensure("wanda")
    monkeypatch.setitem(repos.REPOS, "wanda", replace(pinned, url=str(source)))
    with (clone / "lib" / "prune.py").open("a", encoding="utf-8") as handle:
        handle.write("# local edit\n")
    with pytest.raises(RuntimeError, match="tracked files"):
        repos.ensure("wanda")
    with pytest.raises(ValueError, match="unknown official repository"):
        repos.ensure("not-registered")


def test_load_module_is_isolated():
    path = list(sys.path)
    before = {key: sys.modules.get(key) for key in ("lib", "quant", "sparsegpt")}
    prune_module = repos.load_module("wanda", "lib.prune")
    sparsegpt = repos.load_module("sparsegpt", "sparsegpt")  # imports its sibling: from quant import *
    assert prune_module.__name__ == "_tricast_official_wanda.lib.prune"
    assert Path(prune_module.__file__).resolve() == (repos.ensure("wanda") / "lib" / "prune.py").resolve()
    assert sys.path == path
    assert {key: sys.modules.get(key) for key in before} == before
    assert repos.load_module("sparsegpt", "quant").quantize is sparsegpt.quantize
    assert repos.load_module("wanda", "lib.prune") is prune_module
    for name in ("wanda", "sparsegpt"):  # no bytecode or other file was written into the clones
        status = subprocess.run(["git", "-C", str(repos.ensure(name)), "status", "--porcelain", "--ignored"],
                                capture_output=True, text=True, check=True).stdout
        assert status == ""


def test_load_module_ignores_foreign_sibling_names(monkeypatch):
    root = repos.ensure("sparsegpt").resolve()
    for key in [key for key in sys.modules if key.startswith("_tricast_official_sparsegpt")]:
        monkeypatch.delitem(sys.modules, key)  # a fresh import (restored after the test)
    fakes = {}
    for key in ("quant", "lib"):
        fake = types.ModuleType(key)
        fake.__file__ = f"/elsewhere/{key}.py"
        fake.quantize = lambda *args: "foreign"
        fake.Quantizer = type("Quantizer", (), {})
        fakes[key] = (fake, dict(vars(fake)))
        monkeypatch.setitem(sys.modules, key, fake)
    quant = repos.load_module("sparsegpt", "quant")  # loaded first: sparsegpt must reuse this very module
    sparsegpt = repos.load_module("sparsegpt", "sparsegpt")  # from quant import *
    assert Path(quant.__file__).resolve() == root / "quant.py"
    assert sparsegpt.quantize is quant.quantize and sparsegpt.Quantizer is quant.Quantizer
    assert Path(sparsegpt.quantize.__code__.co_filename).resolve() == root / "quant.py"
    for key, (fake, attributes) in fakes.items():
        assert sys.modules[key] is fake and vars(fake) == attributes


def test_sibling_check_rejects_a_foreign_binding(tmp_path):
    foreign = types.ModuleType("quant")
    exec(compile("def quantize(x):\n    return x\n", str(tmp_path / "quant.py"), "exec"), vars(foreign))
    official = types.ModuleType("_tricast_official_sparsegpt.sparsegpt")
    official.quantize = foreign.quantize
    root = repos.ensure("sparsegpt").resolve()
    with pytest.raises(RuntimeError, match="outside the pinned clone"):
        repos._check_siblings([official], root, {"quant"})
    official.quantize = repos.load_module("sparsegpt", "quant").quantize
    repos._check_siblings([official], root, {"quant"})


@pytest.mark.parametrize("kind", KINDS)
def test_compat_official_layer_call_is_bit_identical(kind):
    model = _tiny(kind)
    attention = model.config._attn_implementation
    calls = []
    def record(module, args, kwargs, output):
        calls.append((module, args, kwargs, output))

    hooks = [layer.register_forward_hook(record, with_kwargs=True) for layer in model.model.layers]
    with torch.no_grad(), compat.official_compat(model, SEQLEN):
        assert model.seqlen == SEQLEN
        model(_loader(1)[0][0])
        own = list(calls)
        assert [call[0] for call in own] == list(model.model.layers)
        for layer, args, kwargs, output in own:
            assert "position_embeddings" in kwargs  # the model's own call passed through unchanged
            assert kwargs["attention_mask"].dim() == 4  # eager: a real mask tensor, as the 2023 code expects
            mask, positions = kwargs["attention_mask"], kwargs["position_ids"]
            result = layer(args[0], attention_mask=mask, position_ids=positions)
            assert isinstance(result, tuple) and len(result) == 1
            assert torch.equal(result[0], output)
    for hook in hooks:
        hook.remove()
    assert model.config._attn_implementation == attention
    assert type(model.model.layers) is nn.ModuleList
    assert all("forward" not in vars(layer) for layer in model.model.layers)
    assert "seqlen" not in vars(model) and "hf_device_map" not in vars(model)


@pytest.mark.parametrize("kind", KINDS)
def test_wanda_unstructured_prunes_half_of_every_row(kind, monkeypatch):
    model = _tiny(kind)
    statistics = _count_calls(monkeypatch, "lib.layerwrapper", "WrappedGPT", "add_batch")
    report = official_prune.prune(model, None, "wanda", 0.5, nsamples=NSAMPLES, loader=_loader())
    weights = _linears(model)
    assert len(weights) == 2 * LINEARS
    for weight in weights:  # official wanda compares weights within each output row
        assert ((weight == 0).sum(dim=1) == weight.shape[1] // 2).all()
    # official activation statistics: one add_batch per layer x linear x calibration sample
    assert statistics["calls"] == 2 * LINEARS * NSAMPLES
    assert report["check_sparsity"] == {"overall": 0.5, "layers": [0.5, 0.5]}
    assert report["data"] == {"source": "loader override", "batches": NSAMPLES, "requests": ["c4"]}
    assert report["official"]["commit"] == repos.REPOS["wanda"].commit
    assert report["official_commit"] == _head(repos.ensure("wanda")) == repos.REPOS["wanda"].commit
    assert report["official"]["function"] == "lib.prune.prune_wanda"
    assert type(model.model.layers) is nn.ModuleList and "seqlen" not in vars(model)


def test_wanda_2_4_zeros_two_of_every_four():
    model = _tiny("llama")
    report = official_prune.prune(model, None, "wanda", 0.5, prune_n=2, prune_m=4, nsamples=NSAMPLES,
                                  loader=_loader())
    for weight in _linears(model):
        groups = weight.reshape(weight.shape[0], -1, 4)  # groups of 4 along the input dimension
        assert ((groups == 0).sum(dim=-1) >= 2).all()
    assert report["nm"] == {"n": 2, "m": 4, "short_groups": 0, "ok": True}
    assert report["args"]["sparsity_type"] == "2:4"


def test_sparsegpt_unstructured_half_density(monkeypatch):
    if not torch.cuda.is_available():  # official fasterprune calls torch.cuda.synchronize(): no-op on CPU
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    model = _tiny("llama")
    solves = _count_calls(monkeypatch, "lib.sparsegpt", "SparseGPT", "fasterprune")
    report = official_prune.prune(model, None, "sparsegpt", 0.5, nsamples=NSAMPLES, loader=_loader())
    assert solves["calls"] == 2 * LINEARS
    layer_zeros, layer_numel = [0, 0], [0, 0]
    for index, (entry, weight) in enumerate(zip(report["linears"], _linears(model), strict=True)):
        rows, columns = weight.shape
        # Official mask per 128-column block: tmp <= sorted(tmp)[int(numel * 0.5)], so half of the block
        # plus the threshold weight itself.
        expected = sum(int(rows * (min(start + 128, columns) - start) * 0.5) + 1
                       for start in range(0, columns, 128))
        assert entry["zeros"] == int((weight == 0).sum()) == expected
        layer_zeros[index // LINEARS] += expected
        layer_numel[index // LINEARS] += weight.numel()
    sparsity = [round(zeros / numel, 6) for zeros, numel in zip(layer_zeros, layer_numel, strict=True)]
    assert report["check_sparsity"]["layers"] == sparsity
    assert all(abs(value - 0.5) < 2e-4 for value in sparsity)


def test_wanda_mask_differs_from_magnitude_mask():
    base = _tiny("llama")
    wanda, magnitude = copy.deepcopy(base), copy.deepcopy(base)
    official_prune.prune(wanda, None, "wanda", 0.5, prune_n=2, prune_m=4, nsamples=NSAMPLES, loader=_loader())
    report = official_prune.prune(magnitude, None, "magnitude", 0.5, prune_n=2, prune_m=4)
    assert report["data"]["requests"] == []  # magnitude reads no data
    assert report["nm"]["ok"]
    differing = [int(((a == 0) != (b == 0)).sum())
                 for a, b in zip(_linears(wanda), _linears(magnitude), strict=True)]
    assert sum(differing) > 0  # wanda weighs |w| by activation norms; same 2:4 structure, other weights


class _Rows:
    """The slice of ``datasets.Dataset`` official get_c4 uses: len, row access, column slice."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, key):
        return {"text": self.texts[key]}


def test_default_data_is_official_c4_sampling(monkeypatch):
    data = repos.load_module("wanda", "lib.data")
    calls = []

    def load_dataset(path, name=None, *, data_files, split):
        if name == "allenai--c4":
            raise ValueError("BuilderConfig 'allenai--c4' not found.")  # datasets>=2.x on the hub repo
        calls.append((path, name, data_files, split))
        return _Rows([" ".join(f"t{(row * 7 + column) % 252}" for column in range(80)) for row in range(10)])

    monkeypatch.setattr(data, "load_dataset", load_dataset)
    model = _tiny("llama")
    report = official_prune.prune(model, _tokenizer(), "wanda", 0.5, nsamples=NSAMPLES)
    train, validation = "en/c4-train.00000-of-01024.json.gz", "en/c4-validation.00000-of-00008.json.gz"
    assert calls == [("allenai/c4", "en", {"train": train}, "train"),
                     ("allenai/c4", "en", {"validation": validation}, "validation")]
    assert report["data"] == {"source": "official lib.data.get_loaders('c4')", "dataset": "c4",
                              "requests": ["c4"]}
    assert any("allenai--c4" in note for note in report["deviations"])
    assert data.load_dataset is load_dataset
    assert report["check_sparsity"]["overall"] == 0.5


def test_seqlen_maps_the_official_default_to_shorter_windows(monkeypatch):
    data = repos.load_module("wanda", "lib.data")
    texts = [" ".join(f"t{(row * 7 + column) % 252}" for column in range(80)) for row in range(10)]
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: _Rows(texts))
    windows = []
    get_c4 = data.get_c4

    def recording_get_c4(nsamples, seed, seqlen, tokenizer):  # official get_loaders calls it by name
        loader, validation = get_c4(nsamples, seed, seqlen, tokenizer)
        windows.extend(tuple(batch[0].shape) for batch in loader)
        return loader, validation

    monkeypatch.setattr(data, "get_c4", recording_get_c4)
    model = _tiny("llama")
    report = official_prune.prune(model, _tokenizer(), "wanda", 0.5, nsamples=NSAMPLES, seqlen=16)
    assert windows == [(1, 16)] * NSAMPLES  # the official sampler cut windows of the mapped length
    mappings = [entry for entry in report["deviations"] if isinstance(entry, dict)]
    assert len(mappings) == 1
    assert {key: mappings[0][key] for key in ("param", "official", "used")} == {
        "param": "seqlen", "official": SEQLEN, "used": 16}
    assert "official default infeasible" in mappings[0]["reason"]
    assert report["seqlen"] == 16 and "seqlen" not in vars(model)
    for weight in _linears(model):
        assert ((weight == 0).sum(dim=1) == weight.shape[1] // 2).all()
    default = official_prune.prune(_tiny("llama"), None, "wanda", 0.5, nsamples=NSAMPLES, loader=_loader())
    assert default["seqlen"] == SEQLEN and not any(isinstance(entry, dict) for entry in default["deviations"])
    for seqlen in (1, SEQLEN + 1):
        with pytest.raises(ValueError, match="seqlen"):
            official_prune.prune(_tiny("llama"), None, "wanda", 0.5, nsamples=NSAMPLES, seqlen=seqlen)


def test_recipe_prune_block():
    block = {"method": "wanda", "ratio": 0.5}
    plain = load_recipe({"name": "pruned", "defaults": {}})
    recipe = load_recipe({"name": "pruned", "defaults": {}, "prune": block})
    assert recipe.prune == {**block, "n": 0, "m": 0, "nsamples": 128, "seed": 0, "dataset": "c4"}
    assert load_recipe(recipe.to_dict()).sha256 == recipe.sha256 != plain.sha256
    # recorded before prune.seqlen existed: a block without it keeps its hash
    assert recipe.sha256 == "bf33bedbe424c37eb24dba3a2f5bf5980d361aeb8e917525e8cd1482120739e4"
    mapped = load_recipe({"name": "pruned", "defaults": {}, "prune": {**block, "seqlen": 2048}})
    assert mapped.prune["seqlen"] == 2048 and mapped.sha256 != recipe.sha256
    assert load_recipe(mapped.to_dict()).sha256 == mapped.sha256
    assert "prune" not in plain.to_dict()
    with pytest.raises(ValueError, match="two pruning paths"):
        load_recipe({"name": "x", "defaults": {"sparsity": {"kind": "n:m", "n": 2, "m": 4}}, "prune": block})
    with pytest.raises(ValueError, match="two pruning paths"):
        override = {"match": "*q_proj", "sparsity": {"kind": "unstructured", "ratio": 0.5}}
        load_recipe({"name": "x", "defaults": {}, "overrides": [override], "prune": block})
    with pytest.raises(ValueError, match=r"prune\.ratio: must be 0.5"):
        load_recipe({"name": "x", "defaults": {}, "prune": {**block, "ratio": 0.6, "n": 2, "m": 4}})
    with pytest.raises(ValueError, match=r"prune\.n, m"):
        load_recipe({"name": "x", "defaults": {}, "prune": {**block, "n": 1, "m": 4}})
    with pytest.raises(ValueError, match=r"prune\.nsamples"):
        load_recipe({"name": "x", "defaults": {}, "prune": {**block, "nsamples": 256}})
    with pytest.raises(ValueError, match="prune"):
        load_recipe({"name": "x", "defaults": {}, "prune": {**block, "blocksize": 128}})  # unknown key


def test_runner_prunes_before_patching_and_reloads(tmp_path, monkeypatch):
    runner = pytest.importorskip("tricast.eval.runner")
    from tricast.eval.envinfo import capture_env
    from tricast.nn import patch

    model, tokenizer = _tiny("llama"), _tokenizer()
    loads = []
    monkeypatch.setattr(runner, "_load_model", lambda *args: loads.append(args) or (model, tokenizer))
    env = capture_env(None, {"model_id": "offline/tiny", "model_sha": "model-commit", "git_sha": "a" * 40,
                             "git_dirty": False, "src_sha256": None, "seed": 42, "dtype": "fp32",
                             "device": "cpu"})
    monkeypatch.setattr(runner, "capture_env", lambda *args: env.copy())
    official = functools.partial(official_prune.prune, loader=_loader(length=16))
    monkeypatch.setattr(official_prune, "prune", official)
    seen = []
    patch_model = patch.patch_model

    def recording_patch(target, recipe, **kwargs):
        weight = target.model.layers[0].self_attn.q_proj.weight
        seen.append(float((weight == 0).float().mean()))
        return patch_model(target, recipe, **kwargs)

    monkeypatch.setattr(patch, "patch_model", recording_patch)
    linear = {"defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
              "include": ["model.layers.0.self_attn.q_proj"]}
    pruned = {"name": "pruned", **linear,
              "prune": {"method": "wanda", "ratio": 0.5, "nsamples": NSAMPLES, "seqlen": 16}}
    cfg = {"model": "offline/tiny", "dtype": "fp32", "device": "cpu", "seed": 42,
           "recipes": [pruned, {"name": "plain", **linear}],
           "tasks": {"ppl": {"texts": ["t0 t1 t2 t3"], "seqlen": 4}}, "output_dir": str(tmp_path)}
    results = runner.run_config(cfg)
    assert [result["status"] for result in results] == ["complete", "complete"]
    assert results[0]["prune"]["check_sparsity"] == {"overall": 0.5, "layers": [0.5, 0.5]}
    assert results[0]["prune"]["seqlen"] == 16  # the recipe's seqlen reached official prune
    assert "prune" not in results[1]
    assert seen[0] == 0.5  # quantization patching saw the pruned weights
    assert len(loads) == 2  # the model was reloaded after the run that pruned it
    with pytest.raises(ValueError, match="unpruned model"):
        patch_model(_tiny("llama"), load_recipe(pruned))
