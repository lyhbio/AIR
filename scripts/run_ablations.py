"""Run or inspect all paper ablation experiments (Tables 3-6, Appendix A)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def build_ablation_plan(ablation: str, output_root: Path, device: str = "cuda:0") -> List[Dict]:
    """Builds the exact list of evaluation jobs for the requested paper ablation."""
    spec = json.loads((CONFIG_DIR / "ablations.json").read_text(encoding="utf-8"))
    jobs: List[Dict] = []

    def _add(tag: str, args_list: List[str], section: str) -> None:
        out_path = output_root / section / f"{tag}.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_eval.py"),
            *args_list,
            "--device",
            device,
            "--output",
            str(out_path),
        ]
        jobs.append({"section": section, "tag": tag, "output": str(out_path), "command": cmd})

    if ablation in {"sensitivity", "all"}:
        s3 = spec["r_end_sensitivity_table3"]
        for budget in s3["budgets"]:
            for r_end in s3["r_end_values"]:
                _add(
                    f"he_step{budget}_rend{r_end}",
                    [
                        "--task", s3["task"],
                        "--backbone", s3["backbone"],
                        "--method", "air",
                        "--steps", str(budget),
                        "--r-end", str(r_end),
                        "--interval", str(s3["interval"]),
                    ],
                    "table3_sensitivity",
                )

    if ablation in {"schedule", "all"}:
        s4 = spec["schedule_and_interval_table4"]
        for budget in s4["budgets"]:
            for variant in s4["variants"]:
                args_list = [
                    "--task", s4["task"],
                    "--backbone", s4["backbone"],
                    "--method", "air",
                    "--steps", str(budget),
                    "--schedule", variant["schedule"],
                    "--interval", str(variant["interval"]),
                ]
                if "fixed_rate" in variant:
                    args_list.extend(["--fixed-rate", str(variant["fixed_rate"])])
                _add(f"{variant['name']}_step{budget}", args_list, "table4_schedule")

    if ablation in {"heavy", "all"}:
        s5 = spec["heavy_checkpoint_table5"]
        for task in s5["tasks"]:
            for budget in s5["budgets"]:
                for arm_name, heavy_flag in [("heavy_on", "on"), ("heavy_off", "off")]:
                    _add(
                        f"{task}_step{budget}_{arm_name}",
                        [
                            "--task", task,
                            "--backbone", s5["backbone"],
                            "--method", "air",
                            "--steps", str(budget),
                            "--heavy", heavy_flag,
                        ],
                        "table5_heavy",
                    )

    if ablation in {"score", "all"}:
        s6 = spec["verification_score_table6"]
        for budget in s6["budgets"]:
            for score_mode in ["prob", "margin", "top2_margin", "entropy"]:
                _add(
                    f"he_step{budget}_score_{score_mode}",
                    [
                        "--task", s6["task"],
                        "--backbone", s6["backbone"],
                        "--method", "air",
                        "--steps", str(budget),
                        "--score-mode", score_mode,
                    ],
                    "table6_score",
                )
            for score_mode in ["prob", "margin"]:
                _add(
                    f"he_step{budget}_argmax_score_{score_mode}",
                    [
                        "--task", s6["task"],
                        "--backbone", s6["backbone"],
                        "--method", "air",
                        "--steps", str(budget),
                        "--sampling-alg", "argmax",
                        "--temperature", "0.0",
                        "--score-mode", score_mode,
                    ],
                    "table6_score_argmax",
                )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate AIR paper ablations (Tables 3-6).")
    parser.add_argument("--ablation", choices=["sensitivity", "schedule", "heavy", "score", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true", help="Execute commands sequentially instead of dry-run.")
    args = parser.parse_args()

    jobs = build_ablation_plan(args.ablation, args.output_root, device=args.device)
    if not args.execute:
        print(json.dumps({"total_jobs": len(jobs), "jobs": jobs}, indent=2))
        return

    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] Running {job['section']}/{job['tag']}")
        subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
