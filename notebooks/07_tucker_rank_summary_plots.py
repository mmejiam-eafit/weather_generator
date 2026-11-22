"""Task 07: Tucker rank tradeoff plot -- overall error vs. error-distribution skew.

Mirrors `07_dmd_rank_summary_plots.py`'s style (two stacked panels sharing
an x-axis, never dual-axis) for the same kind of question that plot
answered for `r_dmd`: is there a rank where both an aggregate-fidelity
measure and a distribution-shape measure are simultaneously reasonable, or
does one keep trading off against the other?

IMPORTANT, and drawn directly on the chart, not just in this docstring:
unlike the DMD rank sweep (9 ranks, all evaluated the same way), these
three Tucker points are NOT at comparable convergence levels -- 20% is
fully converged (200/200 iterations); 30% and 40% are early snapshots
(80/300 and 50/80 iterations respectively) from runs that were still
actively improving when they were stopped/cut off. Treat this as
suggestive, not a rigorous knee-point analysis like the DMD one was.

Data hardcoded from `tasks/07-tucker-dmd-rank-selection.md` §5
(2026-09-26/27 reconstruction-check runs) -- there is no per-rank CSV for
this (unlike the DMD sweep), since each point came from a separate,
differently-configured VM run, not a single parametrized sweep script.

Run directly: `poetry run python notebooks/07_tucker_rank_summary_plots.py`.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# rank_pct, label (with convergence status), overall RMSE, skewness
DATA = [
    (20, "20%\n(converged,\niter 200/200)", 0.3357, 3.007),
    (30, "30%\n(NOT converged,\niter 80/300)", 0.2844, 3.349),
    (40, "40%\n(NOT converged,\niter 50/80)", 0.2438, 3.510),
]


def main() -> None:
    ranks = [d[0] for d in DATA]
    labels = [d[1] for d in DATA]
    rmses = [d[2] for d in DATA]
    skews = [d[3] for d in DATA]

    fig, (ax_rmse, ax_skew) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True, sharex=True)

    ax_rmse.plot(ranks, rmses, marker="o", markersize=8)
    for x, y in zip(ranks, rmses):
        ax_rmse.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax_rmse.set_ylabel("overall RMSE\n(standardized-anomaly units)")
    ax_rmse.set_title("Overall reconstruction error (lower = better fit)")
    ax_rmse.grid(True, alpha=0.3)

    ax_skew.plot(ranks, skews, marker="o", markersize=8, color="tab:red")
    for x, y in zip(ranks, skews):
        ax_skew.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax_skew.set_ylabel("skewness of open-ocean\n|error| distribution")
    ax_skew.set_title("Error-distribution skew (lower = less tail-concentrated, i.e. \"better\")")
    ax_skew.set_xticks(ranks)
    ax_skew.set_xticklabels(labels, fontsize=9)
    ax_skew.set_xlabel("Tucker rank (% of grid)")
    ax_skew.grid(True, alpha=0.3)

    fig.suptitle(
        "Tucker rank: aggregate error keeps improving, but error concentration (skew) keeps\n"
        "worsening -- NOT an apples-to-apples convergence comparison (see x-axis labels)",
        fontsize=11,
    )

    out_path = OUT_DIR / "tucker_rank_error_vs_skew_tradeoff.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
