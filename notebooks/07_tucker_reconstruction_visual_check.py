"""Task 07: visual sanity check of a Tucker rank's reconstruction quality.

Loads a saved `(core, factor_lat, factor_lon)` bundle from one of the
rank-sweep scripts (`07_placement2_tucker_on_anomalies.py` /
`07_gaussianized_rank_sweep.py`) and a small set of true-field time-slices
(extracted separately, e.g. via `np.load(..., mmap_mode="r")` on the VM
where the full ~7.8GB cached tensor lives -- never loaded whole here), and
plots true vs. reconstructed vs. error side by side for each slice. Land
cells (per the static era-2 ocean mask) are shown as NaN, not data.

Time is mode 0 and is never decomposed (`modes=[1,2]` throughout this
task), so reconstructing a handful of timesteps is cheap: for each index
`t`, `recon[t] = factor_lat @ core[t] @ factor_lon.T` -- computed here via
tensorly's own `multi_mode_dot` for consistency with the rest of the repo.

Run directly: `poetry run python notebooks/07_tucker_reconstruction_visual_check.py
--tucker_npz results/task07/gaussianize/anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz
--true_slices_npz results/task07/gaussianize/demo_slices.npz --label 20pct`.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import kurtosis, skew
from tensorly.tenalg import multi_mode_dot

COASTAL_BAND_CELLS = 3

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCEAN_MASK = REPO_ROOT / "results" / "task07" / "gaussianize" / "era2_ocean_mask.npy"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tucker_npz", type=str, required=True)
    p.add_argument("--true_slices_npz", type=str, required=True)
    p.add_argument("--ocean_mask_path", type=str, default=str(DEFAULT_OCEAN_MASK))
    p.add_argument("--label", type=str, required=True, help="e.g. '20pct', used in filenames/titles")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    tucker = np.load(args.tucker_npz)
    core, factor_lat, factor_lon = tucker["core"], tucker["factor_lat"], tucker["factor_lon"]

    true_bundle = np.load(args.true_slices_npz)
    time_indices, true_slices = true_bundle["time_indices"], true_bundle["true"]

    ocean_mask = np.load(args.ocean_mask_path)  # True == ocean
    land = ~ocean_mask

    recon_slices = multi_mode_dot(core[time_indices], [factor_lat, factor_lon], modes=[1, 2])

    true_masked = np.where(land[None, :, :], np.nan, true_slices)
    recon_masked = np.where(land[None, :, :], np.nan, recon_slices)
    error_masked = true_masked - recon_masked

    n = len(time_indices)
    # A robust (percentile) range, not the raw max -- a handful of extreme
    # storm-event pixels otherwise stretch the color scale so far that the
    # bulk of the (standardized, mostly O(1)) field washes out to near-white.
    field_absmax = float(np.nanpercentile(np.abs(true_masked), 99))
    error_absmax = float(np.nanpercentile(np.abs(error_masked), 99))

    fig, axes = plt.subplots(
        n, 3, figsize=(13, 4 * n), constrained_layout=True, squeeze=False
    )
    col_titles = ["True", "Reconstructed", "Error (true − reconstructed)"]
    for row, t_idx in enumerate(time_indices):
        panels = [true_masked[row], recon_masked[row], error_masked[row]]
        for col, (field, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            vmax = error_absmax if col == 2 else field_absmax
            im = ax.imshow(
                field, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"
            )
            ax.set_title(f"{title} (t={t_idx})" if col == 0 else title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"Tucker rank {args.label} reconstruction check "
        f"(wsh, standardized log-energy anomaly space)"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tucker_reconstruction_check_{args.label}.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")

    rmse = float(np.sqrt(np.nanmean(error_masked**2)))
    print(f"RMSE over shown slices (standardized-anomaly units): {rmse:.4f}")

    # Quantify the "error concentrates at the coastline" visual impression:
    # coastal ocean = ocean cells within COASTAL_BAND_CELLS of a land cell.
    coastal_band = binary_dilation(land, iterations=COASTAL_BAND_CELLS) & ocean_mask
    open_ocean = ocean_mask & ~coastal_band
    coastal_rmse = float(np.sqrt(np.nanmean(error_masked[:, coastal_band] ** 2)))
    open_rmse = float(np.sqrt(np.nanmean(error_masked[:, open_ocean] ** 2)))
    n_coastal, n_open = int(coastal_band.sum()), int(open_ocean.sum())
    print(
        f"Coastal ocean (within {COASTAL_BAND_CELLS} cells of land, n={n_coastal}): "
        f"RMSE={coastal_rmse:.4f}"
    )
    print(f"Open ocean (n={n_open}): RMSE={open_rmse:.4f}")
    print(f"Ratio coastal/open: {coastal_rmse / open_rmse:.2f}x")

    # Does error concentrate at the edges of open-ocean PHENOMENA themselves
    # (storm/eddy boundaries), not just at the coastline? Correlate |error|
    # against the true field's local spatial-gradient magnitude, restricted
    # to open_ocean cells only (>= COASTAL_BAND_CELLS from land, so the
    # gradient stencil never touches the land=0 sentinel).
    grad_mags = []
    abs_errors = []
    for row in range(n):
        gy, gx = np.gradient(true_slices[row])
        grad_mag = np.sqrt(gy**2 + gx**2)
        grad_mags.append(grad_mag[open_ocean])
        abs_errors.append(np.abs(error_masked[row][open_ocean]))
    grad_mags = np.concatenate(grad_mags)
    abs_errors = np.concatenate(abs_errors)

    corr = float(np.corrcoef(grad_mags, abs_errors)[0, 1])
    print(f"\nOpen-ocean-only: correlation(|true field gradient|, |error|) = {corr:.3f}")

    quartiles = np.nanpercentile(grad_mags, [25, 50, 75])
    bins = np.digitize(grad_mags, quartiles)
    print("Open-ocean RMSE by local-gradient quartile of the true field (Q1=flattest, Q4=sharpest):")
    for q in range(4):
        sel = bins == q
        q_rmse = float(np.sqrt(np.mean(abs_errors[sel] ** 2)))
        print(f"  Q{q + 1} (n={int(sel.sum())}): RMSE={q_rmse:.4f}")

    lower_half = bins <= 1  # Q1+Q2: flatter 50% of open-ocean cells by true-field gradient
    upper_half = bins >= 2  # Q3+Q4: sharper 50%
    rmse_lower = float(np.sqrt(np.mean(abs_errors[lower_half] ** 2)))
    rmse_upper = float(np.sqrt(np.mean(abs_errors[upper_half] ** 2)))
    half_ratio = rmse_upper / rmse_lower
    print(
        f"\nRMSE(upper half, sharper 50%)={rmse_upper:.4f}  "
        f"RMSE(lower half, flatter 50%)={rmse_lower:.4f}  "
        f"ratio={half_ratio:.2f}  (~1 = uniform, higher = error concentrating in the sharp-transition half)"
    )

    # Shape of the open-ocean |error| distribution itself (not the 4
    # gradient-conditioned group-RMSEs above -- a genuine distribution of
    # ~229k values per quartile, enough to support moments/percentiles).
    # Two views of the same "is error concentrated in a heavy tail" question:
    # a robust percentile-based ratio, and the formal 3rd/4th moments.
    p25, p50, p75, p100 = np.percentile(abs_errors, [25, 50, 75, 100])
    iqr = p75 - p25
    tail_stretch = p100 - p50
    percentile_ratio = float(tail_stretch / iqr)
    error_skew = float(skew(abs_errors))
    error_kurtosis = float(kurtosis(abs_errors))  # Fisher (excess), normal = 0
    print(f"\nOpen-ocean |error| distribution shape (n={len(abs_errors)}):")
    print(f"  P25={p25:.4f} P50={p50:.4f} P75={p75:.4f} P100(max)={p100:.4f}  IQR={iqr:.4f}")
    print(f"  (max - median) / IQR = {percentile_ratio:.2f}  (~1 = symmetric/light-tailed, higher = heavy right tail)")
    print(f"  skewness = {error_skew:.3f}  excess kurtosis = {error_kurtosis:.3f}")


if __name__ == "__main__":
    main()
