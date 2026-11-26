"""Generate with stock vLLM 0.21 from a local compressed-tensors directory or GGUF file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", type=Path, help="local HF W4 directory, or a .gguf file")
    cli.add_argument(
        "--tokenizer", type=Path, help="original HF config/tokenizer directory (required for GGUF)"
    )
    cli.add_argument("--prompt", default="How many sides does a triangle have? Answer with one number.")
    cli.add_argument("--max-new-tokens", type=int, default=16)
    args = cli.parse_args()
    gguf = args.model.is_file() and args.model.suffix.lower() == ".gguf"
    if not args.model.is_dir() and not gguf:
        cli.error("expected an existing HF directory or GGUF file")
    tok_dir = args.tokenizer or args.model
    if not tok_dir.is_dir() or not (tok_dir / "config.json").is_file():
        cli.error("a local config/tokenizer directory is required; use --tokenizer for GGUF")
    if not 1 <= args.max_new_tokens < 512:
        cli.error("--max-new-tokens must be in [1, 511] for this 512-token example")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    opts = (
        {"load_format": "gguf", "quantization": "gguf", "hf_config_path": str(tok_dir)}
        if gguf
        else {"load_format": "safetensors", "quantization": "compressed-tensors"}
    )
    engine = LLM(
        model=str(args.model),
        tokenizer=str(tok_dir),
        dtype="bfloat16",
        seed=42,
        max_model_len=512,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.23,
        kv_cache_memory_bytes=384 << 20,
        **opts,
    )
    answer = engine.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=args.max_new_tokens),
        use_tqdm=False,
    )[0].outputs[0]
    if not answer.token_ids or not answer.text.strip():
        raise RuntimeError("empty generation")
    print(answer.text)


if __name__ == "__main__":
    main()
