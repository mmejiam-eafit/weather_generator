"""Task 07: full DMD+OU generative statistical-fidelity check.

Answers the same open question as `07_dmd_surrogate_rank_check.py` from
the other direction: does mode-OU, fit on the DMD residual at a given
r_dmd, already reproduce the true statistics (autocorrelation, spectrum,
marginal) of the held-out Gaussianized core -- or does statistical
fidelity keep improving as r_dmd grows, meaning DMD is still missing real
structure? Full reviewed design: `tasks/07-tucker-dmd-rank-selection.md`
§8.

Stays in Gaussianized-core space throughout (never calls
`GaussianCopula.inverse_transform` or `tucker_inverse`), isolating the
DMD/OU question from the copula's own documented lossiness (probability
clipping to [1e-6, 1-1e-6], saturating z at ~+/-4.75, linear extrapolation
off the outermost LUT bin -- see `07_gaussianized_rank_sweep.py`).
Explicit limitation: a clean result here does not by itself guarantee no
missing structure in physical units for extreme events specifically,
since `inverse_transform` is exactly where that information is discarded.

Generative model, per r_dmd candidate (this supersedes a first attempt
that sampled every DMD mode's OU residual independently: real DMD applied
to real data guarantees exact complex-conjugate mode pairs, but drawing
independent complex noise per mode ignores that pairing, so a
conjugate-pair's two "independent" synthetic residuals stop being exact
conjugates of each other -- confirmed empirically, `imag_real_ratio_mean`
came back ~0.93-1.04 at every rank in that first run, meaning the
generated signal was comparably imaginary as real rather than "real plus
negligible noise". Fix: for each conjugate pair (k, k'), |eta_k(t)| =
|eta_k'(t)| exactly, so only ONE real, non-negative quantity -- the
magnitude -- needs a stochastic model; the phase is taken as purely
deterministic, advancing at that mode's own frequency (`Lambda`'s
argument). Sharing one magnitude draw across a pair and negating the
phase for the partner makes them exact conjugates by construction, at
every step, with no explicit forcing needed. Because magnitude is
non-negative with a nonzero mean (unlike a zero-mean-reverting OU
residual), it's modeled in log-space (de-meaned, mean restored at
generation time) -- consistent with this project's existing convention
for physically-positive quantities (log-energy for wave height). Modes
with no genuine conjugate partner (Im(Lambda) ~= 0) keep the original
complex-residual OU treatment unchanged, since a real mode's residual can
already be positive or negative and has no separate phase to model.

Run directly (same cached core as `07_dmd_rank_sweep.py`):
`poetry run python notebooks/07_dmd_ou_generative_check.py --core_path
results/task07/gaussianize/anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz`.
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
import xarray as xr

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import DMDFit, ModeOUFit, dmd_fit
from weather.simulate import simulate_ou_forward

R_DMD_CANDIDATES = [5, 10, 20, 40, 50, 80, 100, 150, 200]
TRAIN_FRACTION = 0.8
N_BINS = 500
SEED = 42
N_REALIZATIONS = 10
MAX_ACF_LAG = 48
KS_SUBSAMPLE_SIZE = 200_000
WELCH_NPERSEG_CAP = 64  # matches `weather/validate.py`'s `spectra_check`
N_PLOT_CELLS = 12
CONJUGATE_PAIR_TOL = 1e-6  # |Lambda|<1 and O(1)-scale, per the existing n_growing check
LOG_MAGNITUDE_FLOOR = 1e-12  # purely defensive against log(0); negligible vs. real magnitudes


def _wrap(core: np.ndarray) -> xr.Dataset:
    return xr.Dataset({"core": (("time", "core_lat", "core_lon"), core)})


def _unwrap(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["core"].values)


def find_conjugate_pairs(
    lam: np.ndarray, tol: float = CONJUGATE_PAIR_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition DMD mode indices by conjugate-pair structure.

    Exact DMD on real data yields a real reduced operator, whose
    eigenvalues are algebraically exactly real or exactly conjugate pairs
    (up to eigensolver floating-point error) -- not approximate, so a
    tight tolerance reliably and unambiguously recovers the pairing.

    Returns
    -------
    real_idx : modes with Im(lam) ~= 0 -- no conjugate partner needed.
    rep_idx : one representative index per conjugate pair.
    partner_idx : rep_idx[i]'s exact conjugate partner.
    """
    n = lam.shape[0]
    used = np.zeros(n, dtype=bool)
    real_idx: list[int] = []
    rep_idx: list[int] = []
    partner_idx: list[int] = []
    for k in range(n):
        if used[k]:
            continue
        if abs(lam[k].imag) < tol:
            real_idx.append(k)
            used[k] = True
            continue
        candidates = np.where(~used)[0]
        candidates = candidates[candidates != k]
        if candidates.size == 0:
            real_idx.append(k)  # unpaired oscillatory mode -- fall back to the real path
            used[k] = True
            continue
        dists = np.abs(lam[candidates] - np.conj(lam[k]))
        best = int(candidates[np.argmin(dists)])
        if dists.min() < tol:
            rep_idx.append(k)
            partner_idx.append(best)
            used[k] = True
            used[best] = True
        else:
            real_idx.append(k)
            used[k] = True
    return (
        np.array(real_idx, dtype=int),
        np.array(rep_idx, dtype=int),
        np.array(partner_idx, dtype=int),
    )


def residual_series(gauss_train: np.ndarray, fit: DMDFit) -> np.ndarray:
    """The per-mode residual eta_k(t) = a_k(t) - a_det_k(t) for every training
    step -- the same quantity `train.dmd.fit_mode_ou` computes internally,
    recomputed here since it isn't exposed and this script needs the full
    series (not just the AR(1) parameters `fit_mode_ou` would return).

    Returns
    -------
    (r_dmd, n_train) complex128
    """
    n_train = gauss_train.shape[0]
    snapshots = gauss_train.reshape(n_train, -1).T
    a_t = np.linalg.pinv(fit.Phi) @ snapshots
    tt = np.arange(n_train, dtype=np.float64)
    a_det = fit.b0[:, None] * (fit.Lambda[:, None] ** tt[None, :])
    return np.asarray(a_t - a_det, dtype=np.complex128)


def ar1_fit(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex discrete AR(1)/OU fit, per row, assuming zero-mean reversion.

    Exactly `train.dmd.fit_mode_ou`'s closed-form estimator, factored out
    so it can also be applied to the (de-meaned) log-magnitude series of
    conjugate-pair representatives, not only to `fit_mode_ou`'s own
    per-mode complex residual.

    Parameters
    ----------
    series : (n_channels, T) complex128, already zero-mean.
    dt : float

    Returns
    -------
    theta, sigma : (n_channels,) float64
    phi : (n_channels,) complex128
    """
    s_t = series[:, :-1]
    s_t1 = series[:, 1:]
    numerator = np.sum(np.conj(s_t) * s_t1, axis=1)
    denominator = np.sum(np.abs(s_t) ** 2, axis=1)
    phi = numerator / denominator
    theta = -np.log(np.clip(phi.real, 1e-4, 0.9999)) / dt
    innovation = s_t1 - phi[:, None] * s_t
    innovation_variance = np.mean(np.abs(innovation) ** 2, axis=1)
    sigma = np.sqrt(innovation_variance * 2.0 * theta / (1.0 - np.abs(phi) ** 2))
    return (
        np.asarray(theta, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
        np.asarray(phi, dtype=np.complex128),
    )


def acf_fft(x_centered: np.ndarray, max_lag: int) -> np.ndarray:
    """Linear (non-circular) autocorrelation via zero-padded FFT.

    Parameters
    ----------
    x_centered : (T, n_state) float64, already mean-subtracted per column
        (mean-centering is required -- a nonzero small-sample mean
        inflates apparent correlation at every lag otherwise).
    max_lag : int

    Returns
    -------
    (max_lag + 1, n_state) float64, normalized so lag 0 = 1.
    """
    n_times = x_centered.shape[0]
    n_fft = 2 * n_times  # zero-pad to avoid circular wraparound
    spectrum = np.fft.rfft(x_centered, n=n_fft, axis=0)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=0)
    autocovariance = autocovariance[: max_lag + 1].real
    lag0 = autocovariance[0]
    safe_lag0 = np.where(lag0 == 0.0, 1.0, lag0)
    return np.asarray(autocovariance / safe_lag0, dtype=np.float64)


def welch_psd(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD along axis 0, batched across the remaining (state) axis."""
    nperseg = min(x.shape[0], WELCH_NPERSEG_CAP)
    freqs, psd = scipy.signal.welch(x, fs=1.0 / dt, axis=0, nperseg=nperseg)
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def subsample(
    values: np.ndarray, rng: np.random.Generator, size: int = KS_SUBSAMPLE_SIZE
) -> np.ndarray:
    """Random subsample so the KS test stays fast and its D-statistic
    meaningful, rather than pooling ~1e8+ values (which would make the
    p-value degenerate without making the D-statistic any more accurate).
    """
    if values.shape[0] <= size:
        return values
    idx = rng.choice(values.shape[0], size=size, replace=False)
    return np.asarray(values[idx], dtype=np.float64)


def generate_one_realization(
    fit: DMDFit,
    eta_train: np.ndarray,
    real_idx: np.ndarray,
    rep_idx: np.ndarray,
    partner_idx: np.ndarray,
    a_det_holdout: np.ndarray,
    n_holdout: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate one synthetic holdout realization with conjugate-pair-coherent OU noise.

    Real modes (`real_idx`): unchanged complex-residual OU, chained one
    step at a time (still required -- see the comment at the chaining
    call below).

    Conjugate pairs (`rep_idx`/`partner_idx`): OU is fit to the
    representative's de-meaned LOG-magnitude only; the phase is purely
    deterministic (the mode's own frequency, continued from the true
    measured phase at the end of training). The partner is set to the
    exact conjugate at every step -- automatic, not forced after the
    fact, since it's derived from the same shared magnitude/phase.

    Returns
    -------
    core_synth : (n_holdout, n_state) float64
    imag_real_ratio : float
    """
    n_train = eta_train.shape[1]
    dt = fit.dt

    # --- real modes: unchanged from the original design ---
    real_theta, real_sigma, real_phi = (
        ar1_fit(eta_train[real_idx], dt)
        if real_idx.size
        else (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=np.complex128),
        )
    )

    # --- conjugate pairs: log-magnitude OU + deterministic phase ---
    mag_train = np.abs(eta_train[rep_idx]) if rep_idx.size else np.zeros((0, n_train))
    log_mag_train = np.log(mag_train + LOG_MAGNITUDE_FLOOR)
    mean_log_mag = log_mag_train.mean(axis=1) if rep_idx.size else np.zeros(0)
    log_mag_centered = (log_mag_train - mean_log_mag[:, None]).astype(np.complex128)
    mag_theta, mag_sigma, mag_phi = (
        ar1_fit(log_mag_centered, dt)
        if rep_idx.size
        else (np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.complex128))
    )
    mode_freq = np.angle(fit.Lambda[rep_idx]) if rep_idx.size else np.zeros(0)
    true_phase_end = np.angle(eta_train[rep_idx, -1]) if rep_idx.size else np.zeros(0)

    # One combined OU state per stochastic channel: real modes' own
    # residual, plus one de-meaned log-magnitude channel per pair.
    combined_theta = np.concatenate([real_theta, mag_theta])
    combined_sigma = np.concatenate([real_sigma, mag_sigma])
    combined_phi = np.concatenate([real_phi, mag_phi])
    combined_ou = ModeOUFit(theta=combined_theta, sigma=combined_sigma, phi=combined_phi, dt=dt)

    n_real = real_idx.size
    seed_real = eta_train[real_idx, -1] if n_real else np.zeros(0, dtype=np.complex128)
    seed_mag = log_mag_centered[:, -1] if rep_idx.size else np.zeros(0, dtype=np.complex128)
    state = np.concatenate([seed_real, seed_mag])

    n_channels = state.shape[0]
    path = np.empty((n_channels, n_holdout), dtype=np.complex128)
    # MUST chain with n_steps=1.0 in a loop, not a single vectorized call
    # with an array of steps: `simulate_ou_forward`'s noise term is drawn
    # independently per element it's given, so a vectorized multi-step
    # call gives the correct marginal at each horizon but an
    # independently redrawn (wrong) joint law across horizons -- silently
    # decorrelating the path and corrupting the ACF comparison.
    for i in range(n_holdout):
        state = simulate_ou_forward(combined_ou, state, n_steps=1.0, rng=rng)
        path[:, i] = state

    real_path = path[
        :n_real
    ].real  # discard simulate_ou_forward's spurious imaginary part on a real channel
    log_mag_path = (
        path[n_real:].real + mean_log_mag[:, None] if rep_idx.size else np.zeros((0, n_holdout))
    )
    mag_path = np.exp(log_mag_path)

    r_dmd = fit.Lambda.shape[0]
    eta_holdout = np.zeros((r_dmd, n_holdout), dtype=np.complex128)
    if n_real:
        eta_holdout[real_idx] = real_path
    if rep_idx.size:
        steps = np.arange(1, n_holdout + 1, dtype=np.float64)
        phase_path = true_phase_end[:, None] + mode_freq[:, None] * steps[None, :]
        eta_rep = mag_path * np.exp(1j * phase_path)
        eta_holdout[rep_idx] = eta_rep
        eta_holdout[partner_idx] = np.conj(eta_rep)

    a_synth = a_det_holdout + eta_holdout
    core_synth_raw = fit.Phi @ a_synth  # (n_state, n_holdout)
    imag_real_ratio = float(
        np.linalg.norm(core_synth_raw.imag) / max(np.linalg.norm(core_synth_raw.real), 1e-300)
    )
    core_synth = core_synth_raw.real.T  # (n_holdout, n_state)
    return np.asarray(core_synth, dtype=np.float64), imag_real_ratio


def plot_diagnostics(
    out_dir: Path,
    tag: str,
    r_dmd: int,
    acf_real: np.ndarray,
    acf_synth: np.ndarray,
    freqs: np.ndarray,
    psd_real: np.ndarray,
    psd_synth: np.ndarray,
    n_state: int,
    rng: np.random.Generator,
) -> None:
    cell_idx = rng.choice(n_state, size=N_PLOT_CELLS, replace=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cell in cell_idx:
        ax.plot(acf_real[:, cell], color="C0", alpha=0.3)
        ax.plot(acf_synth[:, cell], color="C1", alpha=0.3, linestyle="--")
    ax.set_xlabel("lag (snapshots)")
    ax.set_ylabel("ACF")
    ax.set_title(f"r_dmd={r_dmd}: real (solid) vs. synthetic-ensemble-mean (dashed)")
    fig.savefig(out_dir / f"dmd_ou_acf_r_dmd{r_dmd}_{tag}.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cell in cell_idx:
        ax.loglog(freqs[1:], psd_real[1:, cell], color="C0", alpha=0.3)
        ax.loglog(freqs[1:], psd_synth[1:, cell], color="C1", alpha=0.3, linestyle="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"r_dmd={r_dmd}: real (solid) vs. synthetic-ensemble-mean (dashed)")
    fig.savefig(out_dir / f"dmd_ou_psd_r_dmd{r_dmd}_{tag}.png", dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--core_path", type=str, required=True)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "task07" / "gaussianize"),
    )
    p.add_argument("--tag", type=str, default="wsh_r124x288_202401-202410")
    p.add_argument("--r_dmd_candidates", type=int, nargs="+", default=R_DMD_CANDIDATES)
    args = p.parse_args()

    core_path = Path(args.core_path)
    npz = np.load(core_path)
    core = npz["core"].astype(np.float64)
    n_times, r_lat, r_lon = core.shape
    n_state = r_lat * r_lon
    n_train = int(round(n_times * TRAIN_FRACTION))
    n_holdout = n_times - n_train
    print(f"core shape {core.shape}, n_state={n_state}, n_train={n_train}, n_holdout={n_holdout}")

    rng = np.random.default_rng(SEED)

    copula = GaussianCopula(n_bins=N_BINS)
    copula.fit(_wrap(core[:n_train]))
    gauss_train = _unwrap(copula.transform(_wrap(core[:n_train])))  # 3D -- dmd_fit needs this shape
    gauss_holdout = _unwrap(copula.transform(_wrap(core[n_train:]))).reshape(n_holdout, n_state)
    del core, npz
    gc.collect()

    real_centered = gauss_holdout - gauss_holdout.mean(axis=0, keepdims=True)
    acf_real = acf_fft(real_centered, MAX_ACF_LAG)
    freqs, psd_real = welch_psd(gauss_holdout, SNAPSHOT_CADENCE_SECONDS)
    real_pool = subsample(gauss_holdout.reshape(-1), rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"dmd_ou_generative_check_{args.tag}.csv"

    fieldnames = [
        "r_dmd",
        "n_train",
        "n_holdout",
        "n_realizations",
        "core_path",
        "n_growing",
        "max_lambda",
        "n_conjugate_pairs",
        "n_real_modes",
        "skipped_growing_mode",
        "imag_real_ratio_mean",
        "acf_distance",
        "log_psd_distance",
        "ks_d_statistic",
        "ks_p_value",
        "elapsed_s",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r_dmd in args.r_dmd_candidates:
            t0 = time.time()
            fit = dmd_fit(
                gauss_train, r_dmd=r_dmd, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact"
            )
            n_growing = int(np.sum(np.abs(fit.Lambda) > 1.0 + 1e-6))
            max_lambda = float(np.abs(fit.Lambda).max())

            if max_lambda >= 1.0:
                # Guard rail: the deterministic term below is only
                # guaranteed to decay (not explode) while every |Lambda| <
                # 1. Today's data has max|Lambda| in [0.977, 0.982] for
                # every rank 5-200, but nothing enforces that structurally
                # -- an unguarded growing mode would silently fill every
                # downstream metric with inf/nan instead of raising.
                writer.writerow(
                    {
                        "r_dmd": r_dmd,
                        "n_train": n_train,
                        "n_holdout": n_holdout,
                        "n_realizations": 0,
                        "core_path": str(core_path),
                        "n_growing": n_growing,
                        "max_lambda": max_lambda,
                        "n_conjugate_pairs": "",
                        "n_real_modes": "",
                        "skipped_growing_mode": True,
                        "imag_real_ratio_mean": "",
                        "acf_distance": "",
                        "log_psd_distance": "",
                        "ks_d_statistic": "",
                        "ks_p_value": "",
                        "elapsed_s": f"{time.time() - t0:.2f}",
                    }
                )
                f.flush()
                print(
                    f"r_dmd={r_dmd}: SKIPPED, max|Lambda|={max_lambda:.6f} >= 1",
                    file=sys.stderr,
                )
                continue

            eta_train = residual_series(gauss_train, fit)
            real_idx, rep_idx, partner_idx = find_conjugate_pairs(fit.Lambda)

            # a_det_holdout[:, i] lands on absolute time index n_train+i,
            # matching gauss_holdout[i]'s own absolute time. Given
            # n_train ~= 1754 already underflows Lambda**t to ~1e-16 by
            # end of training (established in the DMD rank sweep), this
            # term is expected to be numerically negligible throughout
            # the holdout window -- meaning this experiment effectively
            # tests whether the OU-driven residual alone (real modes'
            # complex OU, pairs' log-magnitude OU + deterministic phase)
            # reproduces gauss_holdout's statistics, not a bug.
            steps = np.arange(1, n_holdout + 1, dtype=np.float64)
            a_det_holdout = fit.b0[:, None] * (
                fit.Lambda[:, None] ** (n_train - 1 + steps[None, :])
            )

            acf_accum = np.zeros_like(acf_real)
            psd_accum = np.zeros_like(psd_real)
            imag_ratios: list[float] = []
            synth_pool_parts: list[np.ndarray] = []
            for _realization in range(N_REALIZATIONS):
                core_synth, imag_real_ratio = generate_one_realization(
                    fit, eta_train, real_idx, rep_idx, partner_idx, a_det_holdout, n_holdout, rng
                )
                imag_ratios.append(imag_real_ratio)

                synth_centered = core_synth - core_synth.mean(axis=0, keepdims=True)
                acf_accum += acf_fft(synth_centered, MAX_ACF_LAG)
                _, psd_synth = welch_psd(core_synth, SNAPSHOT_CADENCE_SECONDS)
                psd_accum += psd_synth
                synth_pool_parts.append(subsample(core_synth.reshape(-1), rng))

                del core_synth, synth_centered
                gc.collect()

            acf_synth_mean = acf_accum / N_REALIZATIONS
            psd_synth_mean = psd_accum / N_REALIZATIONS
            # Lag 0 is trivially 1 for both by construction -- exclude it.
            acf_distance = float(np.mean(np.abs(acf_real[1:] - acf_synth_mean[1:])))
            log_psd_distance = float(
                np.mean((np.log(psd_real + 1e-300) - np.log(psd_synth_mean + 1e-300)) ** 2)
            )

            synth_pool = np.concatenate(synth_pool_parts)
            ks_result = scipy.stats.ks_2samp(real_pool, synth_pool)

            elapsed = time.time() - t0
            writer.writerow(
                {
                    "r_dmd": r_dmd,
                    "n_train": n_train,
                    "n_holdout": n_holdout,
                    "n_realizations": N_REALIZATIONS,
                    "core_path": str(core_path),
                    "n_growing": n_growing,
                    "max_lambda": max_lambda,
                    "n_conjugate_pairs": int(rep_idx.size),
                    "n_real_modes": int(real_idx.size),
                    "skipped_growing_mode": False,
                    "imag_real_ratio_mean": float(np.mean(imag_ratios)),
                    "acf_distance": acf_distance,
                    "log_psd_distance": log_psd_distance,
                    "ks_d_statistic": float(ks_result.statistic),
                    "ks_p_value": float(ks_result.pvalue),
                    "elapsed_s": f"{elapsed:.2f}",
                }
            )
            f.flush()
            print(
                f"r_dmd={r_dmd:>4d}  pairs={rep_idx.size:>3d}  real={real_idx.size:>3d}  "
                f"imag/real={np.mean(imag_ratios):.4e}  acf_dist={acf_distance:.4e}  "
                f"log_psd_dist={log_psd_distance:.4e}  "
                f"ks_D={ks_result.statistic:.4e} (p={ks_result.pvalue:.2e})  "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )

            if r_dmd in (args.r_dmd_candidates[0], args.r_dmd_candidates[-1]):
                plot_diagnostics(
                    out_dir,
                    args.tag,
                    r_dmd,
                    acf_real,
                    acf_synth_mean,
                    freqs,
                    psd_real,
                    psd_synth_mean,
                    n_state,
                    rng,
                )

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
