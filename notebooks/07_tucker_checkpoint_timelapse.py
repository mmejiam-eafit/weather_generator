"""Task 07: plot a Tucker rank-sweep checkpoint CSV's error trajectory.

Four panels (2x2, never a dual-axis chart, per this repo's plotting
convention -- see the DMD rank fidelity/CV plots in
`07_dmd_rank_summary_plots.py`):
- top-left: raw `true_err_pct` vs. iteration (linear-linear).
- top-right: the SAME curve, log-y. Caution: `err_pct` converges to a
  positive floor (not zero) at any fixed rank, so log(err_pct) will look
  like it is "flattening" simply because it approaches ANY positive
  constant -- this is largely a restatement of the linear panel, not
  independent evidence about how much runway is left.
- bottom-left: per-checkpoint error delta vs. iteration (linear-linear).
- bottom-right: the delta, log-log. THIS is the more rigorous convergence
  diagnostic: if delta(t) decays as a power law t^-p (a straight line in
  log-log), fitting p directly answers "how much future improvement is
  left" -- p > 1 means the remaining (infinite-horizon) sum of future
  deltas converges to a small, boundable number; p <= 1 means it does not,
  and substantial further improvement could still accumulate even though
  each individual step keeps shrinking.

Run directly: `poetry run python notebooks/07_tucker_checkpoint_timelapse.py
--checkpoint_csv results/task07/gaussianize/anomalies_wsh_r186x432_checkpointed300_202401-202410_checkpoints_live.csv
--label 30pct_8hcap`.
"""

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_csv", type=str, required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--time_budget_hours", type=float, default=None,
        help="If set, project the reachable iteration count and error floor at this"
        " wall-clock cap, using the observed elapsed_s_cumulative pace.",
    )
    args = p.parse_args()

    iterations: list[int] = []
    err_pct: list[float] = []
    elapsed_s: list[float] = []
    with open(args.checkpoint_csv, newline="") as f:
        for row in csv.DictReader(f):
            iterations.append(int(row["iteration"]))
            err_pct.append(float(row["true_err_pct"]))
            elapsed_s.append(float(row["elapsed_s_cumulative"]))

    deltas = [err_pct[i - 1] - err_pct[i] for i in range(1, len(err_pct))]
    delta_iters = iterations[1:]

    fig, ((ax_err, ax_err_log), (ax_delta, ax_delta_log)) = plt.subplots(
        2, 2, figsize=(13, 8), constrained_layout=True
    )

    ax_err.plot(iterations, err_pct, marker="o", markersize=4)
    ax_err.set_ylabel("true_err_pct (%)")
    ax_err.set_xlabel("iteration")
    ax_err.set_title(f"Tucker rank {args.label}: error vs. iteration")
    ax_err.grid(True, alpha=0.3)

    ax_err_log.plot(iterations, err_pct, marker="o", markersize=4)
    ax_err_log.set_yscale("log")
    ax_err_log.set_xlabel("iteration")
    ax_err_log.set_title("Same curve, log-y\n(flattens near ANY positive floor -- see caveat)")
    ax_err_log.grid(True, alpha=0.3, which="both")

    ax_delta.plot(delta_iters, deltas, marker="o", markersize=4, color="tab:orange")
    ax_delta.axhline(0, color="gray", linewidth=0.8)
    ax_delta.set_ylabel("error drop since\nprevious checkpoint (pp)")
    ax_delta.set_xlabel("iteration")
    ax_delta.set_title("Rate of improvement")
    ax_delta.grid(True, alpha=0.3)

    log_iter = np.log(delta_iters)
    log_delta = np.log(deltas)
    slope, intercept = np.polyfit(log_iter, log_delta, 1)
    fit_line = np.exp(intercept) * np.array(delta_iters, dtype=float) ** slope

    ax_delta_log.plot(delta_iters, deltas, marker="o", markersize=4, color="tab:orange")
    ax_delta_log.plot(
        delta_iters, fit_line, linestyle="--", color="black", linewidth=1,
        label=f"power-law fit: delta ~ iter^{slope:.2f}",
    )
    ax_delta_log.set_xscale("log")
    ax_delta_log.set_yscale("log")
    ax_delta_log.set_xlabel("iteration")
    ax_delta_log.set_title("Delta, log-log\n(straight line = power-law decay; slope is the exponent)")
    ax_delta_log.legend(fontsize=8)
    ax_delta_log.grid(True, alpha=0.3, which="both")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tucker_checkpoint_timelapse_{args.label}.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")

    print(f"Last checkpoint: iter={iterations[-1]} err_pct={err_pct[-1]:.4f}")
    print(f"Last delta: {deltas[-1]:.4f} pp (first delta was {deltas[0]:.4f} pp)")
    print(f"Fitted power-law exponent (delta ~ iter^-p): p = {-slope:.3f}")
    if -slope > 1:
        power = -slope
        coeff = np.exp(intercept)
        last_iter = delta_iters[-1]
        remaining = coeff * last_iter ** (1 - power) / (power - 1)
        print(
            "  p > 1: the remaining (infinite-horizon) sum of future deltas converges --"
            " a real, quantifiable case for 'further iterations would yield negligible"
            " improvement', not just a visual impression."
        )
        print(
            f"  Extrapolating the fit to iter->infinity: total remaining improvement"
            f" (summed over ALL future checkpoints, forever) ~= {remaining:.3f} pp,"
            f" i.e. err_pct would bottom out around {err_pct[-1] - remaining:.3f}%"
            f" (current: {err_pct[-1]:.3f}%)."
        )
    else:
        print(
            "  p <= 1: the remaining sum of future deltas does NOT converge under this"
            " fit -- substantial further improvement could still accumulate even though"
            " each individual step keeps shrinking. Flattening in log-y (top-right) alone"
            " would have been misleading here."
        )

    if args.time_budget_hours is not None:
        # Steady-state pace: fit elapsed_s ~ a + b*iteration using all but the
        # first checkpoint (which carries one-time load/init overhead, not
        # representative of the steady per-iteration cost).
        pace_slope, pace_intercept = np.polyfit(iterations[1:], elapsed_s[1:], 1)
        time_budget_s = args.time_budget_hours * 3600.0
        projected_max_iter = (time_budget_s - pace_intercept) / pace_slope
        last_iter = delta_iters[-1]
        print(
            f"\nSteady-state pace: {pace_slope:.2f} s/iter. At a {args.time_budget_hours:.1f}h"
            f" cap ({time_budget_s:.0f}s), projected reachable iteration ~= {projected_max_iter:.0f}"
            f" (requested total_iters may not be reached)."
        )
        if -slope > 1 and projected_max_iter > last_iter:
            power = -slope
            coeff = np.exp(intercept)
            remaining_finite = (
                coeff
                / (power - 1)
                * (last_iter ** (1 - power) - projected_max_iter ** (1 - power))
            )
            projected_err = err_pct[-1] - remaining_finite
            print(
                f"  Projected error AT the time cap (iter~={projected_max_iter:.0f}):"
                f" ~= {projected_err:.3f}% (vs. current {err_pct[-1]:.3f}%,"
                f" vs. the iter->infinity floor of {err_pct[-1] - remaining:.3f}%)."
            )
            print(
                f"  I.e. this run should capture "
                f"{100 * remaining_finite / remaining:.1f}% of the theoretically-available"
                f" further improvement within the time budget -- the rest would need"
                f" iterations well beyond what {args.time_budget_hours:.0f}h can reach."
            )


if __name__ == "__main__":
    main()
