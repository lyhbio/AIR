"""Standalone CLI runner to reproduce AIR, Vanilla, and ReMDM paper configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air import (
    AIRConfig,
    EBSamplerConfig,
    ReMDMConfig,
    SlowFastConfig,
    VanillaConfig,
)


CONFIG_DIR = ROOT / "configs"


def build_paper_config(
    task: str,
    backbone: str,
    method: str,
    steps: int,
    heavy: str = "on",
    score_mode: str = "prob",
    schedule: str = "cosine",
    fixed_rate: float | None = None,
    interval_override: int | None = None,
    r_end_override: float | None = None,
    sampling_alg_override: str | None = None,
    temperature_override: float | None = None,
):
    """Constructs the exact sampler configuration matching Sections 4.1-4.7."""
    if task in {"humaneval", "mbpp"}:
        preset = json.loads((CONFIG_DIR / "code_generation.json").read_text(encoding="utf-8"))
        common = preset["common"]
        budget_cfg = preset["budgets"][str(steps)]
        interval = interval_override or common["interval"]
        r_start = common["r_start"]
        r_end = budget_cfg["r_end"] if r_end_override is None else r_end_override
    else:
        preset = json.loads((CONFIG_DIR / "reasoning_qa.json").read_text(encoding="utf-8"))
        common = preset["common"]
        budget_cfg = preset["budgets"][str(steps)]
        interval = interval_override or budget_cfg["interval"]
        r_start = common["r_start"]
        r_end = common["r_end"] if r_end_override is None else r_end_override

    heavy_ratio = common["heavy_ratio"] if heavy == "on" else -1
    model_id = preset["backbones"][backbone]
    sampling_alg = sampling_alg_override or common["sampling_alg"]
    temperature = common["temperature"] if temperature_override is None else temperature_override
    top_p = 1.0 if temperature == 0.0 else common["top_p"]

    if method == "air":
        cfg = AIRConfig(
            budget=steps,
            gen_length=common["gen_length"],
            interval=interval,
            heavy_ratio=heavy_ratio,
            r_start=r_start,
            r_end=r_end,
            schedule=schedule,
            fixed_rate=fixed_rate,
            score_mode=score_mode,
            sampling_alg=sampling_alg,
            temperature=temperature,
            top_p=top_p,
            backbone=backbone,
        )
    elif method == "vanilla":
        cfg = VanillaConfig(
            budget=steps,
            gen_length=common["gen_length"],
            sampling_alg=sampling_alg,
            temperature=temperature,
            top_p=top_p,
            backbone=backbone,
        )
    elif method == "remdm":
        cfg = ReMDMConfig(
            budget=steps,
            gen_length=common["gen_length"],
            sampling_alg=sampling_alg,
            temperature=temperature,
            top_p=top_p,
            backbone=backbone,
        )
    elif method == "eb":
        cfg = EBSamplerConfig(
            max_nfe=common["gen_length"],
            gen_length=common["gen_length"],
            sampling_alg=sampling_alg,
            temperature=temperature,
            top_p=top_p,
            backbone=backbone,
        )
    else:
        cfg = SlowFastConfig(
            max_nfe=common["gen_length"],
            gen_length=common["gen_length"],
            backbone=backbone,
        )
    return model_id, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or run AIR benchmark evaluations via lm-eval.")
    parser.add_argument("--task", choices=["humaneval", "mbpp", "gsm8k", "pubmedqa"], required=True)
    parser.add_argument("--backbone", choices=["dream", "llada"], default="dream")
    parser.add_argument("--method", choices=["air", "vanilla", "remdm", "eb", "slowfast"], default="air")
    parser.add_argument("--steps", type=int, choices=[32, 64, 128, 256], default=64)
    parser.add_argument("--heavy", choices=["on", "off"], default="on")
    parser.add_argument(
        "--score-mode",
        choices=["prob", "margin", "committed_margin", "top2_margin", "entropy"],
        default="prob",
    )
    parser.add_argument(
        "--sampling-alg",
        choices=["entropy", "argmax", "maskgit_plus", "topk_margin"],
        default=None,
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--schedule", choices=["cosine", "linear", "fixed"], default="cosine")
    parser.add_argument("--fixed-rate", type=float, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--r-end", type=float, default=None)
    parser.add_argument("--model", default=None, help="Optional local model directory override.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_results.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id, cfg = build_paper_config(
        task=args.task,
        backbone=args.backbone,
        method=args.method,
        steps=args.steps,
        heavy=args.heavy,
        score_mode=args.score_mode,
        schedule=args.schedule,
        fixed_rate=args.fixed_rate,
        interval_override=args.interval,
        r_end_override=args.r_end,
        sampling_alg_override=args.sampling_alg,
        temperature_override=args.temperature,
    )
    resolved_model = args.model or model_id

    if args.print_config_only:
        print(json.dumps({"model": resolved_model, "task": args.task, "method": args.method, "config": cfg.__dict__}, indent=2))
        return

    import lm_eval
    from air.lm_eval_adapter import DiffusionAIRHarnessLM

    metrics_path = args.output.with_suffix(".jsonl")
    lm = DiffusionAIRHarnessLM(
        pretrained=resolved_model,
        method=args.method,
        backbone=args.backbone,
        device=args.device,
        dtype=args.dtype,
        metrics_path=str(metrics_path),
        **{k: v for k, v in cfg.__dict__.items() if k != "backbone"},
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task],
        num_fewshot=0,
        limit=args.limit,
        confirm_run_unsafe_code=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results.get("results", {}), indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to {args.output}")


if __name__ == "__main__":
    main()
