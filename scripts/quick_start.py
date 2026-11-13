"""Minimal quick-start inference script for AIR on Dream-7B-Instruct or LLaDA-8B-Instruct."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air import AIRConfig, BackboneAdapter, air_generate, load_backbone


DEFAULT_MODELS = {
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-prompt AIR generation.")
    parser.add_argument("--backbone", choices=["dream", "llada"], default="llada")
    parser.add_argument("--model", default=None, help="HF model ID or local path.")
    parser.add_argument("--budget", type=int, default=64, help="Fixed NFE budget T.")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--heavy-ratio", type=int, default=2)
    parser.add_argument("--r-start", type=float, default=0.85)
    parser.add_argument("--r-end", type=float, default=0.60)
    parser.add_argument("--score-mode", choices=["prob", "margin", "entropy"], default="prob")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--prompt",
        default="Write a Python function `fibonacci(n: int) -> int` that returns the n-th Fibonacci number.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = args.model or DEFAULT_MODELS[args.backbone]
    dtype = "float32" if args.device == "cpu" and args.dtype == "bfloat16" else args.dtype

    model, tokenizer, detected_backbone = load_backbone(model_id, device=args.device, dtype=dtype)
    adapter = BackboneAdapter(model, backbone=detected_backbone)

    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"]
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    cfg = AIRConfig(
        budget=args.budget,
        gen_length=args.gen_length,
        interval=args.interval,
        heavy_ratio=args.heavy_ratio,
        r_start=args.r_start,
        r_end=args.r_end,
        score_mode=args.score_mode,
        backbone=detected_backbone,
    )

    tokens, stats = air_generate(adapter, tokenizer, prompt_ids, cfg, device=args.device)
    print(tokenizer.decode(tokens.tolist(), skip_special_tokens=True))
    print(
        f"\n[AIR Stats] NFE={stats['nfe']} | Remasked={stats['remasked_tokens']} "
        f"| HeavyCheckpoints={stats['heavy_checkpoints']} | RemainingMasks={stats['remaining_masks']}"
    )


if __name__ == "__main__":
    main()
