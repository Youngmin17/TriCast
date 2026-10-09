"""CPU checks for the runners and the demo driver (reference backend, tiny random models).

The LLM tests need transformers and tokenizers and skip without them. The vision runners need the pinned
checkpoints, Ultralytics and torchvision, so here they are covered by their letterbox geometry, checkpoint
selection, the shared baseline/emulated comparison on a tiny Conv2d model, and the demo's screening,
selection and bundle writing with stand-in detector/classifier results.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from app import demo
from app.catalog import MODELS, compose_recipe, preset_mma
from app.metrics import match_boxes
from app.runners import common, llm, vision
from app.runners.vision import letterbox, unletterbox
from tricast import load_recipe
from tricast.eval.envinfo import _tree_hash
from tricast.nn import iter_emuconv2d, iter_emulinear

QWEN, LLAMA = "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B"
CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                 "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@pytest.fixture(scope="module")
def tokenizer():
    """Byte-level tokenizer: one token per UTF-8 byte, plus pad and chat markers."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {char: index for index, char in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    for special in ("<pad>", "<|im_start|>", "<|im_end|>"):
        vocab[special] = len(vocab)
    model = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    model.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    model.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=model, pad_token="<pad>",
                                                     eos_token="<|im_end|>",
                                                     additional_special_tokens=["<|im_start|>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def tiny_model(family: str, vocab_size: int, pad: int) -> nn.Module:
    transformers = pytest.importorskip("transformers")
    config_class = transformers.Qwen3Config if family == "qwen3" else transformers.LlamaConfig
    model_class = transformers.Qwen3ForCausalLM if family == "qwen3" else transformers.LlamaForCausalLM
    config = config_class(vocab_size=vocab_size, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                          max_position_embeddings=512, bos_token_id=None, eos_token_id=None, pad_token_id=pad)
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return model_class(config).float().eval()


@pytest.fixture
def qwen(tokenizer) -> nn.Module:
    return tiny_model("qwen3", len(tokenizer), tokenizer.pad_token_id)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode() + tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_token_pieces_hold_multibyte_characters_until_complete(tokenizer) -> None:
    ids = tokenizer("가나a", add_special_tokens=False).input_ids
    assert len(ids) == 7

    def decode(part: list[int]) -> str:
        return tokenizer.decode(part, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    assert llm.token_pieces(decode, ids) == ["", "", "가", "", "", "나", "a"]


def test_run_generate_compares_and_restores_the_model(tokenizer, qwen) -> None:
    sample = torch.tensor([[5, 9, 17, 33]])
    with torch.no_grad():
        before_logits = qwen(sample).logits
    before_state, originals = state_hash(qwen), dict(qwen.named_modules())
    stages = []

    def progress(stage: str, fraction: float, partial: dict | None = None) -> None:
        stages.append((stage, partial))

    result = llm.run_generate(QWEN, "hopper_fp8_w8a8", None, "Hi, 가?", 6, "cpu", model=qwen,
                              tokenizer=tokenizer, progress=progress)
    assert set(result) == {"baseline", "emulated", "metrics", "timing", "evidence", "cached_baseline"}
    for output in (result["baseline"], result["emulated"]):
        assert len(output["tokens"]) == 6
        assert "".join(token["piece"] for token in output["tokens"]) == output["text"]
        for token in output["tokens"]:
            probabilities = [item["p"] for item in token["top"]]
            assert len(probabilities) == 5 and probabilities == sorted(probabilities, reverse=True)
            assert token["logprob"] <= 0
    forced, divergence = result["metrics"]["teacher_forced"], result["metrics"]["first_divergence"]
    assert forced["positions"] == len(forced["kl"]) == len(forced["top1"]) == 6
    # Forced decoding sees the generation's own prefixes: shared tokens agree, the first split does not.
    assert all(forced["top1"][:divergence])
    assert divergence is None or forced["top1"][divergence] is False
    # 2 decoder layers x 7 projections; free and forced decoding each run 1 prefill + 5 decode steps.
    assert result["evidence"] == {"patched_linear": 14, "patched_conv2d": 0, "emulated_calls": 14 * 12,
                                  "backend": "reference"}
    assert result["cached_baseline"] is False
    assert ("baseline", {"baseline": result["baseline"]}) in stages and stages[-1][0] == "metrics"
    assert state_hash(qwen) == before_state and not list(iter_emulinear(qwen))
    assert all(module is originals[name] for name, module in qwen.named_modules())
    with torch.no_grad():
        assert torch.equal(qwen(sample).logits, before_logits)
    again = llm.run_generate(QWEN, "fp8_f7_lowacc", None, "Hi, 가?", 6, "cpu", model=qwen,
                             tokenizer=tokenizer)
    assert again["cached_baseline"] is True and again["baseline"] == result["baseline"]


def test_forced_decoding_reproduces_generation_logits(tokenizer, qwen) -> None:
    prompt_ids = llm.encode_prompt(tokenizer, "precision?", chat=True)
    identity = compose_recipe({"algorithm": "fp64"}, None, "llm.generate")  # unquantized, FP64 accumulation
    for recipe in (None, identity):
        with nullcontext() if recipe is None else common.emulated(qwen, recipe, "reference") as evidence:
            ids, logits = llm._generate(qwen, tokenizer, prompt_ids, 6)
            again, forced = llm._generate(qwen, tokenizer, prompt_ids, 6, forced=ids)
        assert again == ids and logits.shape[0] == 6 and torch.equal(forced, logits)
    assert evidence["emulated_calls"] == 14 * 12


def test_forced_decoding_is_causal(tokenizer, qwen) -> None:
    """Under per-tensor activation scales a later token must not reach earlier positions."""
    prompt_ids = llm.encode_prompt(tokenizer, "precision?", chat=True)
    tokens = tokenizer("accumulate!", add_special_tokens=False).input_ids
    changed = [*tokens[:5], tokenizer("Z", add_special_tokens=False).input_ids[0], *tokens[6:]]
    assert changed != tokens
    with common.emulated(qwen, "hopper_fp8_w8a8", "reference"):
        _, first = llm._generate(qwen, tokenizer, prompt_ids, len(tokens), forced=tokens)
        _, second = llm._generate(qwen, tokenizer, prompt_ids, len(changed), forced=changed)
    # Row t predicts token t from the prompt and tokens[:t]: rows 0..5 never see index 5.
    assert torch.equal(first[:6], second[:6])
    assert not torch.equal(first[6:], second[6:])


def test_same_recipe_as_baseline_gives_identical_outputs(tokenizer, qwen) -> None:
    recipe = compose_recipe(preset_mma("hopper"), "fp8_tensor", "llm.generate")
    result = llm.run_generate(QWEN, recipe, recipe, "precision", 5, "cpu", model=qwen, tokenizer=tokenizer)
    assert result["metrics"]["first_divergence"] is None and result["metrics"]["prefix_match"] == 5
    assert result["metrics"]["teacher_forced"]["kl"] == [0.0] * 5
    assert result["metrics"]["teacher_forced"]["top1"] == [True] * 5
    assert result["baseline"]["text"] == result["emulated"]["text"]


def test_base_model_continues_the_raw_prompt(tokenizer) -> None:
    assert MODELS[LLAMA].chat is False and MODELS[QWEN].chat is True
    prompt = "The accumulator"
    assert llm.encode_prompt(tokenizer, prompt, chat=False).tolist() == [tokenizer(prompt).input_ids]
    model = tiny_model("llama", len(tokenizer), tokenizer.pad_token_id)
    result = llm.run_generate(LLAMA, "fp8_f7_lowacc", None, prompt, 4, "cpu", model=model,
                              tokenizer=tokenizer)
    assert len(result["emulated"]["tokens"]) == 4 and result["evidence"]["emulated_calls"] == 14 * 8


def test_emulated_context_unpatches_on_error() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    with (pytest.raises(RuntimeError, match="stop"),
          common.emulated(model, "hopper_fp8_w8a8", "reference") as evidence):
        assert evidence["patched_linear"] == 2 and len(list(iter_emulinear(model))) == 2
        raise RuntimeError("stop")
    assert not list(iter_emulinear(model))


def test_baseline_cache_keeps_the_four_most_recent_per_model() -> None:
    cache, model, other = common.BaselineCache(), nn.Linear(1, 1), nn.Linear(1, 1)
    for index in range(6):
        cache.put(model, (index,), index)
    assert len(cache) == 4 and cache.get(model, (1,)) is None and cache.get(model, (5,)) == 5
    assert cache.get(model, (2,)) == 2  # now the most recent; (3,) is the oldest
    cache.put(model, (6,), 6)
    assert cache.get(model, (3,)) is None and cache.get(model, (2,)) == 2
    assert cache.get(other, (6,)) is None


def test_run_env_pins_the_model_without_a_hub_lookup(tmp_path: Path, monkeypatch) -> None:
    looked_up = []
    capture = common.capture_env
    monkeypatch.setattr(common, "capture_env",
                        lambda model_id, extra: looked_up.append(model_id) or capture(model_id, extra))
    recipe = compose_recipe(preset_mma("hopper"), "fp8_tensor", "llm.generate")
    env = common.run_env("llm.generate", QWEN, recipe, None, {"model_commit": "abc"}, device="cpu")
    assert (env["model_id"], env["model_kind"], env["model_commit"]) == (QWEN, "hf", "abc")
    assert env["model_sha"] == env["model_revision"] == MODELS[QWEN].revision
    assert (env["dtype"], env["device"], env["attn_implementation"]) == ("float32", "cpu", "eager")
    assert env["recipe_sha256"] == load_recipe(recipe).sha256 and env["baseline_recipe_sha256"] is None
    assert env["app"] == "tricast-studio" and re.fullmatch("[0-9a-f]{64}", env["app_sha256"])
    assert env["seed"] == 42 and isinstance(env["deterministic_algorithms"], bool)
    assert set(env["tf32"]) == {"matmul", "cudnn"} and env["generation"]["enable_thinking"] is False
    checkpoint = tmp_path / "yolo11n.pt"
    checkpoint.write_bytes(b"weights")
    detect = compose_recipe(preset_mma("hopper"), "fp8_tensor", "vision.detect")
    fp64 = compose_recipe({"algorithm": "fp64"}, "fp8_tensor", "vision.detect")
    env = common.run_env("vision.detect", "yolo11n", detect, fp64, checkpoint=checkpoint)
    digest = hashlib.sha256(b"weights").hexdigest()
    assert env["checkpoint"] == {"path": str(checkpoint.resolve()), "sha256": digest}
    assert (env["model_kind"], env["model_sha"], env["model_revision"]) == ("local", digest,
                                                                            MODELS["yolo11n"].revision)
    assert {key: env["detection"][key] for key in common.DETECTION} == common.DETECTION
    assert env["baseline_recipe_sha256"] == load_recipe(fp64).sha256 and env["dtype"] == "float32"
    assert looked_up == [None, None]
    json.dumps(env, allow_nan=False)


def test_app_sha256_hashes_the_app_tree_like_tricast_without_bundle_or_caches(tmp_path: Path,
                                                                              monkeypatch) -> None:
    app = tmp_path / "app"
    for name in ("x.py", "web/js/ui.js"):
        (app / name).parent.mkdir(parents=True, exist_ok=True)
        (app / name).write_text(name)
    expected = _tree_hash(app, tmp_path)
    for name in ("web/demo/runs/r.json", "__pycache__/x.cpython-311.pyc", "web/.DS_Store", "runners/y.pyc"):
        (app / name).parent.mkdir(parents=True, exist_ok=True)
        (app / name).write_text("ignored")
    monkeypatch.setattr(common, "APP", app)
    monkeypatch.setattr(common, "REPO", tmp_path)
    common.app_sha256.cache_clear()
    try:
        assert common.app_sha256() == expected
        (app / "web" / "js" / "ui.js").write_text("changed")
        common.app_sha256.cache_clear()
        assert common.app_sha256() != expected
    finally:
        common.app_sha256.cache_clear()


@pytest.mark.parametrize(("shape", "resized", "target", "pad", "gain"), [
    ((480, 640), (480, 640), (512, 672), (16, 16), 1.0),
    ((427, 640), (427, 640), (448, 672), (16, 10), 1.0),
    ((640, 427), (640, 427), (672, 448), (10, 16), 1.0),
    ((960, 1280), (480, 640), (512, 672), (16, 16), 0.5),
    ((240, 320), (480, 640), (512, 672), (16, 16), 2.0),
    ((500, 500), (640, 640), (672, 672), (16, 16), 1.28),
])
def test_letterbox_matches_ultralytics_rect_validation(shape, resized, target, pad, gain) -> None:
    plan = letterbox(*shape, imgsz=640, stride=32)
    assert (plan.original, plan.resized, plan.target, plan.pad) == (shape, resized, target, pad)
    assert plan.gain == pytest.approx(gain)


def test_unletterbox_maps_back_to_original_pixels_and_clips() -> None:
    plan = letterbox(960, 1280, imgsz=640, stride=32)
    boxes = torch.tensor([[16.0, 16.0, 336.0, 266.0], [-10.0, 400.0, 700.0, 600.0]])
    expected = torch.tensor([[0.0, 0.0, 640.0, 500.0], [0.0, 768.0, 1280.0, 960.0]])
    assert torch.equal(unletterbox(boxes, plan), expected)
    assert torch.equal(boxes[0], torch.tensor([16.0, 16.0, 336.0, 266.0]))


def test_checkpoint_path_prefers_explicit_then_environment_then_repository(monkeypatch) -> None:
    monkeypatch.delenv("TRICAST_YOLO_CHECKPOINT", raising=False)
    assert vision.checkpoint_path("yolo11n") == common.REPO / "checkpoints" / "yolo11n.pt"
    monkeypatch.setenv("TRICAST_YOLO_CHECKPOINT", "/env/yolo11n.pt")
    assert vision.checkpoint_path("yolo11n") == Path("/env/yolo11n.pt")
    assert vision.checkpoint_path("yolo11n", "/given/yolo11n.pt") == Path("/given/yolo11n.pt")
    monkeypatch.delenv("TRICAST_RESNET_CHECKPOINT", raising=False)
    assert vision.checkpoint_path("resnet18", Path("/given/r18.pth")) == Path("/given/r18.pth")
    assert vision.checkpoint_path("resnet18").name == "resnet18-f37072fd.pth"


def test_explicit_checkpoint_reaches_the_hash_checked_loaders(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "resnet18.pth"
    fake.write_bytes(b"not the pinned weights")
    with pytest.raises(ValueError, match="SHA256"):
        vision.load_classifier("cpu", checkpoint=fake)
    with pytest.raises(FileNotFoundError):
        vision.load_detector("cpu", checkpoint=tmp_path / "missing.pt")
    seen = []

    def loader(device: str, checkpoint: Path | None = None) -> None:
        seen.append(checkpoint)
        raise LookupError

    monkeypatch.setattr(vision, "load_detector", loader)
    monkeypatch.setattr(vision, "load_classifier", loader)
    for run, model_id in ((vision.run_detect, "yolo11n"), (vision.run_classify, "resnet18")):
        with pytest.raises(LookupError):
            run(model_id, "hopper_fp8_w8a8", None, tmp_path / "unused.jpg", "cpu", checkpoint=fake)
    assert seen == [fake, fake]


def test_vision_comparison_patches_a_copy_and_caches_the_baseline() -> None:
    torch.manual_seed(0)
    native = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16, 5)).eval()
    batch = torch.randn(1, 3, 4, 4)
    before = state_hash(native)
    recipe = compose_recipe(preset_mma("hopper"), "fp8_tensor", "vision.classify")
    partials = []

    def compute(model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            return model(batch)

    def progress(stage: str, fraction: float, partial: dict | None = None) -> None:
        if partial is not None:
            partials.append(partial)

    def run() -> tuple:
        return vision._compare(native, recipe, None, ("test", "image"), "cpu", None, progress, compute,
                               lambda label, value: {"label": label})

    baseline, value, _, evidence, cached = run()
    assert evidence == {"patched_linear": 1, "patched_conv2d": 1, "emulated_calls": 2, "backend": "reference"}
    assert cached is False and torch.equal(baseline.value, compute(native))
    assert not torch.equal(value, baseline.value)
    assert partials == [{"baseline": {"label": "원본 (native)"}}]
    assert state_hash(native) == before and not list(iter_emuconv2d(native))
    again = run()
    assert again[4] is True and torch.equal(again[0].value, baseline.value) and torch.equal(again[1], value)


@pytest.fixture
def restore_torch_flags() -> Iterator[None]:
    deterministic = torch.are_deterministic_algorithms_enabled()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    yield
    torch.use_deterministic_algorithms(deterministic)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


@pytest.fixture
def checkpoints(tmp_path: Path, monkeypatch) -> list[str]:
    """Stand-in checkpoint files for the demo's flags (main() sets the variables monkeypatch restores)."""
    flags = []
    for model_id, flag in (("yolo11n", "--yolo-checkpoint"), ("resnet18", "--resnet-checkpoint")):
        path = tmp_path / f"{model_id}.ckpt"
        path.write_bytes(model_id.encode())
        monkeypatch.setenv(common.CHECKPOINT_ENV[model_id], str(path))
        flags += [flag, str(path)]
    return flags


def test_demo_quick_llm_bundle(tmp_path: Path, monkeypatch, capsys, tokenizer, qwen, checkpoints,
                               restore_torch_flags) -> None:
    monkeypatch.setattr(llm, "load_llm", lambda model_id, device, revision: (qwen, tokenizer))
    out = tmp_path / "bundle"
    args = ["--out", str(out), "--device", "cpu", "--cases", "llm", "--quick",
            "--cache", str(tmp_path / "cache"), *checkpoints]
    assert demo.main(args) == 0
    assert capsys.readouterr().out.strip().endswith("DEMO_BUNDLE_DONE runs=2 failed=0")
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))["runs"]
    assert [entry["preview"]["baseline"] for entry in index] == [index[0]["preview"]["baseline"]] * 2
    hopper = index[1]
    assert set(hopper) == {"id", "task", "model", "mma", "format", "baseline", "input", "title", "selection",
                           "role", "summary", "preview"}
    assert hopper["mma"] == preset_mma("hopper") and hopper["input"] == {"prompt": demo.QWEN_PROMPTS[0]}
    assert hopper["role"] is None
    assert set(hopper["summary"]) == {"first_divergence", "prefix_match", "top1_agreement", "kl_mean"}
    run = json.loads((out / "runs" / f"{hopper['id']}.json").read_text(encoding="utf-8"))
    assert (run["status"], run["preset"]) == ("done", "hopper")
    assert run["mma_label"] == "CoFDA · F13 · CS32 · fused"
    assert run["recipe"]["bundled"] == "hopper_fp8_w8a8"
    assert run["env"]["recipe_sha256"] == load_recipe(yaml.safe_load(run["recipe"]["yaml"])).sha256
    assert run["env"]["model_revision"] == MODELS[QWEN].revision and run["env"]["app_sha256"]
    assert run["request"]["input"] == {"prompt": demo.QWEN_PROMPTS[0], "max_new_tokens": 48}
    assert run["evidence"]["emulated_calls"] == 14 * 96 and run["cached_baseline"] is True
    assert json.loads((out / "catalog.json").read_text(encoding="utf-8"))["mode"] == "demo"
    monkeypatch.setattr(llm, "run_generate", lambda *args, **kwargs: pytest.fail("resume ran a recorded run"))
    assert demo.main([*args, "--resume"]) == 0
    assert capsys.readouterr().out.strip().endswith("DEMO_BUNDLE_DONE runs=2 failed=0")


def stub_detect(model_id: str, recipe: dict, baseline: dict | None, image_path: Path, device: str,
                **settings: object) -> dict:
    """Four baseline boxes; the emulated side drops ``image id % 4`` of them."""
    image_id = int(Path(image_path).stem.removeprefix("coco_"))
    picture = {"media": Path(image_path).name, "width": 32, "height": 24}
    boxes = [{"cls": "person", "cls_id": 0, "conf": 0.9 - 0.1 * i, "xyxy": [2.0 * i, 2.0, 2.0 * i + 4, 8.0]}
             for i in range(4)]
    kept = boxes[:4 - image_id % 4]
    return {"baseline": {"label": "native", "image": picture, "boxes": boxes},
            "emulated": {"label": recipe["description"], "image": picture, "boxes": kept},
            "metrics": match_boxes(boxes, kept), "timing": {"baseline_s": 0.0, "emulated_s": 0.0},
            "evidence": {"patched_linear": 0, "patched_conv2d": 88, "emulated_calls": 88, "backend": "stub"},
            "cached_baseline": False}


def stub_classify(model_id: str, recipe: dict, baseline: dict | None, image_path: Path, device: str) -> dict:
    picture = {"media": Path(image_path).name, "width": 32, "height": 24}
    top = [{"label": f"class{i}", "class_id": i, "p": 0.5 / (i + 1)} for i in range(5)]
    return {"baseline": {"label": "native", "image": picture, "top": top},
            "emulated": {"label": recipe["description"], "image": picture, "top": top},
            "metrics": {"top1_same": True, "top5_overlap": 5, "kl": 0.0, "baseline_top1_p": 0.5,
                        "emulated_top1_p": 0.5}, "timing": {"baseline_s": 0.0, "emulated_s": 0.0},
            "evidence": {"patched_linear": 1, "patched_conv2d": 20, "emulated_calls": 21, "backend": "stub"},
            "cached_baseline": False}


def test_demo_screens_selects_and_sweeps_vision_cases(tmp_path: Path, monkeypatch, capsys, checkpoints,
                                                      restore_torch_flags) -> None:
    from PIL import Image

    root = tmp_path / "val2017"
    root.mkdir()
    images = [{"id": image_id, "file_name": f"{image_id:012d}.jpg", "license": 1 if image_id == 20 else 4,
               "width": 32, "height": 24, "flickr_url": f"http://flickr.test/{image_id}",
               "coco_url": f"http://coco.test/{image_id}"} for image_id in (20, 15, 11, 14, 13, 12)]
    for image in images:
        Image.new("RGB", (32, 24), (image["id"] * 10, 90, 160)).save(root / image["file_name"])
    licenses = [{"id": 1, "name": "Attribution-NonCommercial-ShareAlike License", "url": "http://nc.test"},
                {"id": 4, "name": "Attribution License", "url": "http://creativecommons.org/licenses/by/2.0/"}]
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({"images": images, "licenses": licenses}), encoding="utf-8")
    monkeypatch.setattr(vision, "run_detect", stub_detect)
    monkeypatch.setattr(vision, "run_classify", stub_classify)
    monkeypatch.setattr(vision, "load_classifier", lambda device: None)
    out = tmp_path / "bundle"
    assert demo.main(["--out", str(out), "--device", "cpu", "--cases", "detect,classify",
                      "--coco-root", str(root), "--coco-ann", str(annotations), "--images", "5",
                      "--cache", str(tmp_path / "cache"), "--render", *checkpoints]) == 0
    # Screening scores (emulated drops id % 4 boxes): 11 -> 3, 15 -> 3, 14 -> 2, 13 -> 1, 12 -> 0.
    scores = json.loads((out / "demo_scores.json").read_text(encoding="utf-8"))
    assert [row["image_id"] for row in scores["images"]] == [11, 12, 13, 14, 15]
    assert [(row["image_id"], row["role"]) for row in scores["selected"]] == [(11, "top1"), (15, "top2"),
                                                                             (14, "median")]
    # 3 images x (8 fused + 4 decoupled + FP64) detection runs and 3 x (5 fused + FP64) classification runs.
    assert capsys.readouterr().out.strip().endswith("DEMO_BUNDLE_DONE runs=57 failed=0")
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))["runs"]
    detect = [entry for entry in index if entry["task"] == "vision.detect"]
    assert len(detect) == 39 and {entry["input"]["image"] for entry in detect} == {
        "coco_000000000011.jpg", "coco_000000000015.jpg", "coco_000000000014.jpg"}
    roles = {entry["input"]["image"]: entry["role"] for entry in index}
    assert roles == {"coco_000000000011.jpg": "차이 점수 1위 (3.0)",
                     "coco_000000000015.jpg": "차이 점수 2위 (3.0)",
                     "coco_000000000014.jpg": "차이 점수 중앙값 (2.0)"}
    assert detect[0]["preview"] == {"baseline_boxes": 4, "emulated_boxes": 1}
    assert set(detect[0]["summary"]) == {"matched", "baseline_only", "emulated_only", "mean_iou"}
    classify = [entry for entry in index if entry["task"] == "vision.classify"]
    assert len(classify) == 18
    assert classify[0]["preview"] == {"baseline_top1": "class0", "emulated_top1": "class0",
                                      "emulated_top1_p": 0.5}
    media = out / "media"
    assert len(list(media.glob("coco_*.jpg"))) == 3 and len(list(media.glob("*_compare.png"))) == 39
    assert "http://flickr.test/15" in (media / "SOURCES.md").read_text(encoding="utf-8")
    run = json.loads((out / "runs" / f"{detect[0]['id']}.json").read_text(encoding="utf-8"))
    assert run["request"]["input"] == {"image": "coco_000000000011.jpg"} and run["status"] == "done"
    assert run["env"]["model_kind"] == "local"
    assert run["env"]["checkpoint"]["sha256"] == hashlib.sha256(b"yolo11n").hexdigest()
