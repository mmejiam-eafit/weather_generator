"""Task 07: summary plots across the three DMD-rank-selection experiments.

Reads the already-computed CSVs from `07_dmd_ou_generative_check.py` and
`07_dmd_rank_cv_check.py` (`tasks/07-tucker-dmd-rank-selection.md` §8-9) and
produces two static diagnostic PNGs -- no new experiments, no re-fitting.

Figure 1 (`dmd_rank_fidelity_tradeoff.png`): spectral fidelity
(`log_psd_distance`) and marginal fidelity (`ks_d_statistic`) vs `r_dmd`,
as two stacked panels sharing an x-axis -- deliberately NOT a dual-axis
chart (two different-scale measures on one plot is the classic
misleading-chart mistake; small multiples instead), each panel marking
its own argmin so the tradeoff between the two metrics is visible at a
glance.

Figure 2 (`dmd_rank_cv_predictive_error.png`): real-data 8-fold
cross-validated one-step predictive error vs `r_dmd` -- each fold drawn
as a thin, de-emphasized line (showing the replication directly) with the
cross-fold mean drawn bold on top.

Run directly (after both source CSVs exist):
`poetry run python notebooks/07_dmd_rank_summary_plots.py`.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTEBOOKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"
GENERATIVE_CSV = RESULTS_DIR / "dmd_ou_generative_check_wsh_r124x288_202401-202410.csv"
CV_CSV = RESULTS_DIR / "dmd_rank_cv_check_wsh_r124x288_202401-202410.csv"

ACCENT = "#3b6ea5"  # single accent hue, reused across both figures for the "main" series
FOLD_GRAY = "#b8bec7"  # recessive, de-emphasized individual-fold lines


def plot_fidelity_tradeoff(out_path: Path) -> None:
    df = pd.read_csv(GENERATIVE_CSV).dropna(subset=["log_psd_distance", "ks_d_statistic"])
    df = df.sort_values("r_dmd")

    fig, (ax_psd, ax_ks) = plt.subplots(
        2, 1, figsize=(6.5, 6.5), sharex=True, constrained_layout=True
    )

    for ax, col, label, title in (
        (
            ax_psd,
            "log_psd_distance",
            "log-PSD distance",
            "Spectral fidelity (lower = better spectral match)",
        ),
        (
            ax_ks,
            "ks_d_statistic",
            "KS D-statistic",
            "Marginal fidelity (lower = better marginal match)",
        ),
    ):
        ax.plot(df["r_dmd"], df[col], color=ACCENT, linewidth=2, marker="o", markersize=5)
        best_r = df.loc[df[col].idxmin(), "r_dmd"]
        best_val = df[col].min()
        ax.axvline(best_r, color=ACCENT, linewidth=1, linestyle="--", alpha=0.5)
        ax.annotate(
            f"min at r_dmd={int(best_r)}",
            xy=(best_r, best_val),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, which="both", axis="both", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_ks.set_xscale("log")
    ax_ks.set_xticks(df["r_dmd"])
    ax_ks.set_xticklabels([str(r) for r in df["r_dmd"]])
    ax_ks.set_xlabel("r_dmd", fontsize=9)

    fig.suptitle(
        "DMD+OU generative check: spectral vs. marginal fidelity, by r_dmd",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cv_predictive_error(out_path: Path) -> None:
    df = pd.read_csv(CV_CSV)
    ranks = sorted(df["r_dmd"].unique())

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)

    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold].sort_values("r_dmd")
        ax.plot(
            sub["r_dmd"],
            sub["median_err_pct_lead1"],
            color=FOLD_GRAY,
            linewidth=1,
            alpha=0.8,
            label="individual fold" if fold == 0 else None,
        )

    agg = df.groupby("r_dmd")["median_err_pct_lead1"].agg(["mean", "std"]).reindex(ranks)
    ax.plot(
        ranks,
        agg["mean"],
        color=ACCENT,
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="mean across 8 folds",
    )
    ax.fill_between(
        ranks,
        agg["mean"] - agg["std"],
        agg["mean"] + agg["std"],
        color=ACCENT,
        alpha=0.15,
        linewidth=0,
    )

    best_r = agg["mean"].idxmin()
    ax.axvline(best_r, color=ACCENT, linewidth=1, linestyle="--", alpha=0.5)
    ax.annotate(
        f"lowest mean at r_dmd={int(best_r)}\n(within 1 fold-std of every r_dmd>=80)",
        xy=(best_r, agg["mean"].min()),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
        color="#333333",
    )

    ax.set_xscale("log")
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("r_dmd", fontsize=9)
    ax.set_ylabel("median 1-step held-out error (%)", fontsize=9)
    ax.set_title(
        "Real-data 8-fold CV: 1-step predictive error is nearly flat across r_dmd",
        fontsize=10,
    )
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    for filename, plot_fn in (
        ("dmd_rank_fidelity_tradeoff.png", plot_fidelity_tradeoff),
        ("dmd_rank_cv_predictive_error.png", plot_cv_predictive_error),
    ):
        results_path = RESULTS_DIR / filename
        plot_fn(results_path)
        # Copied into notebooks/ too: these plots are the evidence backing
        # the rank-selection write-up, so they live alongside the scripts
        # that produced them, not only under results/.
        notebooks_path = NOTEBOOKS_DIR / filename
        shutil.copy2(results_path, notebooks_path)
        print(f"Copied to {notebooks_path}")


if __name__ == "__main__":
    main()
