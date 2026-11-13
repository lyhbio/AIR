"""Reproduces Figure 3: Accuracy-NFE Pareto frontier on HumanEval and MBPP (Dream-7B-Instruct)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 10.5
plt.rcParams["ytick.labelsize"] = 10.5
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def generate_fig3(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

    c_air = "#d62728"
    c_vanilla = "#666666"
    c_remdm = "#ff7f0e"
    c_eb_dyn = "#17becf"
    c_slowfast = "#9467bd"
    nfe_fixed = [32, 64, 128, 256]

    # Panel (a): HumanEval (Dream-7B-Instruct), exact Table 1 values
    ax0 = axes[0]
    air_he = [24.4, 40.9, 52.4, 57.9]
    vanilla_he = [4.2, 13.4, 27.4, 54.8]
    remdm_he = [9.7, 20.1, 31.7, 31.0]

    ax0.plot(nfe_fixed, air_he, marker="D", color=c_air, linewidth=2.8, markersize=8, zorder=10)
    ax0.plot(nfe_fixed, vanilla_he, marker="o", color=c_vanilla, linewidth=1.8, linestyle="--", zorder=4)
    ax0.plot(nfe_fixed, remdm_he, marker="s", color=c_remdm, linewidth=1.8, linestyle="-.", zorder=4)
    ax0.errorbar(
        [64.0], [15.9], xerr=[[14.2], [14.2]], fmt="h", color=c_eb_dyn, markersize=13,
        markerfacecolor=c_eb_dyn, markeredgecolor=c_eb_dyn, markeredgewidth=1.8,
        ecolor=c_eb_dyn, elinewidth=1.6, capsize=4.5, capthick=1.4, zorder=11,
    )
    ax0.errorbar(
        [127.0], [14.0], xerr=[[18.5], [18.5]], fmt="*", color=c_slowfast, markersize=14,
        markeredgecolor="black", markeredgewidth=0.8,
        ecolor=c_slowfast, elinewidth=1.6, capsize=4.5, capthick=1.4, zorder=8,
    )
    ax0.annotate(
        "AIR @ 64 NFE (40.9%)",
        xy=(64, 40.9),
        xytext=(35, 48.0),
        arrowprops=dict(facecolor=c_air, edgecolor=c_air, arrowstyle="->", lw=1.2),
        fontsize=9.0,
        fontweight="medium",
        color=c_air,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_air, linewidth=0.8, alpha=0.95),
    )
    ax0.set_title("(a) HumanEval (Dream-7B-Instruct)", fontweight="bold", pad=12)
    ax0.set_xlabel("Forward Evaluations (NFE, Log-Scale)", fontweight="bold")
    ax0.set_ylabel("Pass@1 Accuracy (%)", fontweight="bold")
    ax0.set_xscale("log", base=2)
    ax0.set_xticks(nfe_fixed)
    ax0.set_xticklabels([str(v) for v in nfe_fixed])
    ax0.set_xlim(21, 300)
    ax0.set_ylim(0, 60)
    ax0.grid(True, linestyle="--", alpha=0.45, linewidth=0.7)

    # Panel (b): MBPP (Dream-7B-Instruct), exact Table 1 values
    ax1 = axes[1]
    air_mbpp = [34.8, 47.0, 54.2, 58.2]
    vanilla_mbpp = [10.4, 19.8, 38.2, 58.0]
    remdm_mbpp = [24.2, 33.2, 40.0, 45.2]

    ax1.plot(nfe_fixed, air_mbpp, marker="D", color=c_air, linewidth=2.8, markersize=8, zorder=10)
    ax1.plot(nfe_fixed, vanilla_mbpp, marker="o", color=c_vanilla, linewidth=1.8, linestyle="--", zorder=4)
    ax1.plot(nfe_fixed, remdm_mbpp, marker="s", color=c_remdm, linewidth=1.8, linestyle="-.", zorder=4)
    ax1.errorbar(
        [32.0], [34.4], xerr=[[8.6], [8.6]], fmt="h", color=c_eb_dyn, markersize=13,
        markerfacecolor=c_eb_dyn, markeredgecolor=c_eb_dyn, markeredgewidth=1.8,
        ecolor=c_eb_dyn, elinewidth=1.6, capsize=4.5, capthick=1.4, zorder=11,
    )
    ax1.errorbar(
        [42.0], [9.8], xerr=[[11.2], [11.2]], fmt="*", color=c_slowfast, markersize=14,
        markeredgecolor="black", markeredgewidth=0.8,
        ecolor=c_slowfast, elinewidth=1.6, capsize=4.5, capthick=1.4, zorder=8,
    )
    ax1.annotate(
        "AIR @ 64 NFE (47.0%)",
        xy=(64, 47.0),
        xytext=(35, 53.0),
        arrowprops=dict(facecolor=c_air, edgecolor=c_air, arrowstyle="->", lw=1.2),
        fontsize=9.0,
        fontweight="medium",
        color=c_air,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_air, linewidth=0.8, alpha=0.95),
    )
    ax1.set_title("(b) MBPP (Dream-7B-Instruct)", fontweight="bold", pad=12)
    ax1.set_xlabel("Forward Evaluations (NFE, Log-Scale)", fontweight="bold")
    ax1.set_ylabel("Pass@1 Accuracy (%)", fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(nfe_fixed)
    ax1.set_xticklabels([str(v) for v in nfe_fixed])
    ax1.set_xlim(21, 300)
    ax1.set_ylim(0, 65)
    ax1.grid(True, linestyle="--", alpha=0.45, linewidth=0.7)

    handles = [
        Line2D([], [], color=c_air, marker="D", lw=2.8, markersize=7, label="AIR (Fixed)"),
        Line2D([], [], color=c_vanilla, marker="o", ls="--", lw=1.8, label="Vanilla Entropy (Fixed)"),
        Line2D([], [], color=c_remdm, marker="s", ls="-.", lw=1.8, label="ReMDM (Fixed)"),
        Line2D([], [], color=c_eb_dyn, marker="h", markerfacecolor=c_eb_dyn,
               markeredgewidth=1.8, markersize=10, ls="none", label="EB-Sampler (Dynamic)"),
        Line2D([], [], color=c_slowfast, marker="*", markeredgecolor="black",
               markeredgewidth=0.8, markersize=13, ls="none", label="SlowFast (Dynamic)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=10,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.5)

    fig.savefig(out_dir / "Fig3.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Fig3.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    generate_fig3(args.out_dir)
    print(f"Saved Fig3.pdf and Fig3.png to {args.out_dir}")
