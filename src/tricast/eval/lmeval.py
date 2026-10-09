"""lm-eval's Hugging Face adapter with TriCast linear arithmetic.

Use ``python -m tricast.eval.lmeval --model tricast --model_args
pretrained=MODEL,recipe=RECIPE --tasks TASKS`` to register the adapter before the
lm-eval CLI starts. The standalone ``lm_eval`` command does not load this module.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import lm_eval
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_name_from_object
except ImportError as exc:
    raise ImportError("TriCast lm-eval integration requires the 'tricast[eval]' dependencies") from exc


def batch_size_for(model, batch_size):
    """lm-eval right-pads loglikelihood batches without a mask, so a scale that spans tokens
    (``EmuLinear.per_sequence``) or a quantized KV cache would also see the padding. Such models
    are evaluated one request per forward; returns ``(batch_size, reason or None)``."""
    from ..nn.patch import iter_emulinear

    if getattr(model, "_tricast_kv_patch", None) is not None:
        return 1, "the KV cache quantization spans tokens"
    names = [name for name, layer in iter_emulinear(model) if layer.per_sequence]
    if names:
        return 1, f"{len(names)} layers use activation scales that span tokens (e.g. {names[0]})"
    return batch_size, None


@register_model("tricast")
class TriCastLM(HFLM):
    """Patch a loaded HF model before lm-eval constructs evaluation requests."""

    def __init__(self, pretrained, recipe=None, backend: str | None = None, calibrate=True, **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        if recipe is not None:
            from ..calibration import calibrate as calibrate_model
            from ..nn.patch import patch_model, unpatch_model
            from ..recipe import load_recipe

            self.recipe = load_recipe(recipe)
            self.patch_report = patch_model(self.model, self.recipe, backend=backend)
            self.calibration = None
            try:
                if not self.patch_report.patched and not self.patch_report.kv:
                    raise ValueError("recipe selected no modules: zero patches were applied")
                if calibrate and self.recipe.needs_calibration:
                    self.calibration = calibrate_model(self.model, self.recipe, tokenizer=self.tokenizer)
            except Exception:
                unpatch_model(self.model)
                raise
        self.batch_size_reason = None
        if self.batch_size != 1:
            size, self.batch_size_reason = batch_size_for(self.model, self.batch_size)
            if self.batch_size_reason:
                self.batch_size_per_gpu = size

    def get_model_info(self) -> dict:
        """lm-eval stores this under ``config``: the recipe, what was patched and calibrated, the
        exact source tree, and the batch size actually used (SPEC AC3)."""
        from dataclasses import asdict

        from .envinfo import source_identity

        info = super().get_model_info()
        info["tricast"] = {"source": source_identity(), "batch_size_reason": self.batch_size_reason}
        if getattr(self, "recipe", None) is not None:
            info["tricast"].update(recipe=self.recipe.to_dict(), recipe_hash=self.recipe.sha256,
                                   patch_report=asdict(self.patch_report), calibration=self.calibration)
        return info

    def _model_call(
        self, inps: torch.Tensor, attn_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patch = getattr(self.model, "_tricast_kv_patch", None)
        if patch is None or patch.kv.mode != "cache":
            return super()._model_call(inps, attn_mask=attn_mask, labels=labels)
        if attn_mask is not None or labels is not None:
            raise ValueError("streaming KV evaluation requires a causal language model")
        from .ppl import _model_logits

        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype,
                                             enabled=self.mixed_precision_dtype is not None):
            return _model_logits(self.model, inps)


class _DatasetTaskManager(TaskManager):
    """Keep the evaluated datasets available for fingerprint capture."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded: dict = {}

    def load(self, *args: Any, **kwargs: Any) -> dict:
        tasks = super().load(*args, **kwargs)
        self.loaded.update(tasks["tasks"])
        return tasks

    def load_task_or_group(self, task_list: str | list | None = None) -> dict:
        tasks = super().load_task_or_group(task_list)
        self.loaded.update(tasks)
        return tasks

    def load_config(self, config: dict) -> dict:
        tasks = super().load_config(config)
        self.loaded.update(tasks)
        return tasks


def _dataset_fingerprints(tasks: dict) -> dict:
    result = {}
    for name, task in tasks.items():
        if isinstance(task, dict):
            result.update(_dataset_fingerprints(task))
        else:
            dataset = getattr(task, "dataset", None)
            result[str(name)] = ({split: getattr(data, "_fingerprint", None)
                                  for split, data in dataset.items()}
                                 if isinstance(dataset, dict) else getattr(dataset, "_fingerprint", None))
    return result


def evaluate(
    model,
    tokenizer,
    tasks,
    *,
    num_fewshot: int | None = None,
    limit: int | float | None = None,
    batch_size: int = 8,
    log_samples: bool = False,
) -> dict:
    """Evaluate a preloaded model; tasks may include offline lm-eval Task objects."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        seed = torch.initial_seed() % 2**32
        adapter = TriCastLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        manager = _DatasetTaskManager()
        task_list = [tasks] if isinstance(tasks, str) else list(tasks)
        for task in task_list:
            if not isinstance(task, (str, dict)):
                manager.loaded[get_task_name_from_object(task)] = task
        result = lm_eval.simple_evaluate(
            model=adapter,
            tasks=task_list,
            task_manager=manager,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            log_samples=log_samples,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )
        if result is None:
            raise RuntimeError("lm-eval returned no results on this process")
        result["dataset_fingerprints"] = _dataset_fingerprints(manager.loaded)
        result["batch_size"] = {"requested": batch_size, "used": adapter.batch_size,
                                "reason": adapter.batch_size_reason}
        patch = getattr(model, "_tricast_kv_patch", None)
        cache_mode = patch is not None and patch.kv.mode == "cache"
        result["forward_modes"] = {
            "loglikelihood": "streaming_cache" if cache_mode else "full_window",
            "generation": "prefill_then_decode_cache" if cache_mode else "model_default",
        }
        return result
    finally:
        for module, training in modes:
            module.training = training


def main() -> None:
    """Run lm-eval's CLI after registering the ``tricast`` model adapter."""
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
