"""Temporal leakage simulations and figures.

This module separates the heavy computation from plotting so each can be
invoked independently from the command line:

    python temporal_leakage_plots.py compute    # run simulations, save HDF5
    python temporal_leakage_plots.py plot       # read HDF5, regenerate SVGs
    python temporal_leakage_plots.py all        # both (default)

Intermediate results are stored as pandas DataFrames in
``figs/temporal_leakage_data.h5`` (one HDF5 key per (panel, sub-table)).
"""

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib

# Force a headless backend only when running this file as a script. When
# imported (e.g. by the VS Code interactive cells below), let the user's
# environment pick its own backend so plt.show() works.
if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import cholesky, toeplitz
from scipy.optimize import brentq
from scipy.stats import norm

# ======================================================================
# Constants
# ======================================================================

# Covariance demo panels (illustrating what train↔test leakage looks like).
# Matches the 5 trials × 3 timepoints layout from correct_cv_plot.svg and
# cv_leakage_plot.svg so the two visualizations tell the same story:
# the CV-layout figure shows *which* points are held out, the heatmap shows
# *why* the leaky layout induces a non-zero train↔test cross block.
LEAKAGE_DEMO_N_OUTER_TRIALS = 5
LEAKAGE_DEMO_N_TIMEPOINTS = 3
LEAKAGE_DEMO_N_TRIALS = LEAKAGE_DEMO_N_OUTER_TRIALS * LEAKAGE_DEMO_N_TIMEPOINTS  # 15
LEAKAGE_DEMO_WITHIN_TRIAL_RHO = 0.8
LEAKAGE_DEMO_CROSS_TRIAL_RHO = 0.0
# Test-index sets mirror fold 1 of the corresponding CV-layout figures.
LEAKAGE_DEMO_CORRECT_TEST_INDICES = (0, 1, 2)        # whole trial 1
LEAKAGE_DEMO_LEAKY_TEST_INDICES = (8, 12, 13)        # scattered timepoints

PANEL_HDF_KEYS: dict[str, tuple[str, ...]] = {
    "cov_independent_train_test": ("covariance", "train_mask", "summary"),
    "cov_cross_rho_08": ("covariance", "train_mask", "summary"),
    "same_apparent_different_leakage": ("per_rho", "summary"),
    "complex_correction": ("per_block_size", "summary"),
}

PANEL_FILENAMES: dict[str, str] = {
    "cov_independent_train_test": "leakage_06_cov_independent_train_test.svg",
    "cov_cross_rho_08": "leakage_07_cov_cross_rho_08.svg",
    "same_apparent_different_leakage": "leakage_02_same_apparent_different_leakage.svg",
    "complex_correction": "leakage_05_complex_correction.svg",
}

# Order matters for RNG reproducibility — keep in sync with prior script.
COMPUTE_ORDER: tuple[str, ...] = (
    "cov_independent_train_test",
    "cov_cross_rho_08",
    "same_apparent_different_leakage",
    "complex_correction",
)

DEFAULT_OUT_DIR = Path("data")
DEFAULT_H5_PATH = DEFAULT_OUT_DIR / "temporal_leakage_data.h5"

@dataclass(frozen=True)
class SplitGroups:
    pos_train: np.ndarray
    neg_train: np.ndarray
    pos_test: np.ndarray
    neg_test: np.ndarray

    @property
    def n_train_per_class(self) -> int:
        return len(self.pos_train)

    @property
    def n_test_per_class(self) -> int:
        return len(self.pos_test)


# ======================================================================
# Math / simulation helpers
# ======================================================================


def exponential_temporal_covariance(
    n_trials: int,
    rho: float,
    decay: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    """Stationary exponential covariance with positive decaying autocorrelation."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be between 0 and 1.")
    times = np.arange(n_trials)
    lag = np.abs(times[:, None] - times[None, :])
    return variance * rho ** (decay * lag)


def alternating_block_mask(n_trials: int, block_size: int, train_first: bool = True) -> np.ndarray:
    block_id = np.arange(n_trials) // block_size
    train_mask = block_id % 2 == 0
    if not train_first:
        train_mask = ~train_mask
    return train_mask


def generate_split_balanced_labels(
    train_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random interleaved labels balanced within train and test masks."""
    labels = np.empty(len(train_mask), dtype=int)
    for mask_value in (True, False):
        indices = np.flatnonzero(train_mask == mask_value)
        if len(indices) % 2 != 0:
            raise ValueError("Train and test partitions must both have even sizes.")
        group_labels = np.r_[np.ones(len(indices) // 2, dtype=int), -np.ones(len(indices) // 2, dtype=int)]
        rng.shuffle(group_labels)
        labels[indices] = group_labels
    return labels


def balanced_split_groups(
    labels: np.ndarray,
    train_mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> SplitGroups:
    """Return equal positive/negative train/test index sets, subsampling if needed."""
    labels = np.asarray(labels)
    pos_train = np.flatnonzero(train_mask & (labels == 1))
    neg_train = np.flatnonzero(train_mask & (labels == -1))
    pos_test = np.flatnonzero((~train_mask) & (labels == 1))
    neg_test = np.flatnonzero((~train_mask) & (labels == -1))

    n_train = min(len(pos_train), len(neg_train))
    n_test = min(len(pos_test), len(neg_test))
    if n_train == 0 or n_test == 0:
        raise ValueError("Both train and test splits need samples from both classes.")

    def take(indices: np.ndarray, n_take: int) -> np.ndarray:
        if len(indices) == n_take:
            return np.sort(indices)
        if rng is None:
            return np.sort(indices[:n_take])
        return np.sort(rng.choice(indices, size=n_take, replace=False))

    return SplitGroups(
        pos_train=take(pos_train, n_train),
        neg_train=take(neg_train, n_train),
        pos_test=take(pos_test, n_test),
        neg_test=take(neg_test, n_test),
    )



def error_to_variance(error: np.ndarray | float, mu: float, eps: float = 1e-5) -> np.ndarray | float:
    error_array = np.asarray(error)
    clipped = np.clip(error_array, eps, 0.5 - eps)
    variance = (mu / norm.ppf(1.0 - clipped)) ** 2
    if np.isscalar(error):
        return float(variance)
    return variance


def variance_to_error(mu: float, variance: np.ndarray | float) -> np.ndarray | float:
    variance_array = np.maximum(np.asarray(variance), 1e-12)
    error = norm.cdf(-mu / np.sqrt(variance_array))
    if np.isscalar(variance):
        return float(error)
    return error


def scalar_corrected_error(observed_error: float, mu: float, leakage: float) -> float:
    corrected_variance = error_to_variance(observed_error, mu) + leakage
    return variance_to_error(mu, corrected_variance)


def full_corrected_error(mu: float, observed_error: float, leakages: np.ndarray) -> float:
    corrected_variance = solve_full_corrected_variance(mu, observed_error, leakages)
    return variance_to_error(mu, corrected_variance)


def estimate_exponential_covariance(
    residual_series: np.ndarray,
    max_lag: int = 48,
    variance_blend: float = 0.8,
) -> np.ndarray:
    """Fit a stationary exponential covariance from residual time series.

    A semivariogram fit is more stable than raw autocovariances for these
    short, highly autocorrelated traces because subtracting a trace mean biases
    long-timescale autocovariances downward.
    """
    residuals = np.atleast_2d(np.asarray(residual_series, dtype=float))
    residuals = residuals - np.mean(residuals, axis=1, keepdims=True)
    n_trials = residuals.shape[1]
    empirical_variance = float(np.mean(residuals**2))
    if empirical_variance <= 1e-10:
        return np.eye(n_trials) * 1e-10

    max_lag = min(max_lag, n_trials - 1)
    lags = np.arange(1, max_lag + 1)
    semivariances = np.array(
        [0.5 * np.mean((residuals[:, lag:] - residuals[:, :-lag]) ** 2) for lag in lags]
    )
    weights = (n_trials - lags).astype(float)

    alpha_grid = np.linspace(0.70, 0.992, 293)
    model_shape = 1.0 - alpha_grid[:, None] ** lags[None, :]
    denominators = np.sum(weights[None, :] * model_shape**2, axis=1)
    variances = np.sum(weights[None, :] * model_shape * semivariances[None, :], axis=1)
    variances = np.maximum(variances / np.maximum(denominators, 1e-12), 1e-8)
    losses = np.sum(
        weights[None, :] * (semivariances[None, :] - variances[:, None] * model_shape) ** 2,
        axis=1,
    )
    best = int(np.argmin(losses))
    alpha = float(alpha_grid[best])
    variogram_variance = float(variances[best])
    variance = variance_blend * variogram_variance + (1.0 - variance_blend) * empirical_variance

    first_column = variance * alpha ** np.arange(n_trials)
    return toeplitz(first_column)



def simple_leakage_terms(covariance: np.ndarray, train_mask: np.ndarray) -> dict[str, np.ndarray | float]:
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(~train_mask)
    n_train = len(train_indices)
    a_block = covariance[np.ix_(train_indices, train_indices)]
    c_block = covariance[np.ix_(train_indices, test_indices)]
    sigma0_sq = float(np.mean(np.diag(covariance)))
    boundary_variance = float(np.sum(a_block) / (2.0 * n_train**2))
    generation_variance = sigma0_sq + boundary_variance
    per_test_leakage = np.sum(c_block, axis=0) / n_train
    return {
        "A": a_block,
        "C": c_block,
        "boundary_variance": boundary_variance,
        "generation_variance": generation_variance,
        "per_test_leakage": per_test_leakage,
        "scalar_leakage": float(np.mean(per_test_leakage)),
    }


def one_process_leakage_terms(
    covariance: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | float | SplitGroups]:
    groups = balanced_split_groups(labels, train_mask, rng)
    p_tr = groups.pos_train
    n_tr = groups.neg_train
    p_te = groups.pos_test
    n_te = groups.neg_test
    n_train = groups.n_train_per_class
    n_test = groups.n_test_per_class

    a_pos = covariance[np.ix_(p_tr, p_tr)]
    a_neg = covariance[np.ix_(n_tr, n_tr)]
    b_cross = covariance[np.ix_(p_tr, n_tr)]
    c_pp = covariance[np.ix_(p_tr, p_te)]
    c_np = covariance[np.ix_(n_tr, p_te)]
    c_pn = covariance[np.ix_(p_tr, n_te)]
    c_nn = covariance[np.ix_(n_tr, n_te)]

    boundary_variance = float(np.sum(a_pos + a_neg + b_cross + b_cross.T) / (4.0 * n_train**2))
    sigma0_sq = float(np.mean(np.diag(covariance)))
    generation_variance = sigma0_sq + boundary_variance
    leakage_pos = (np.sum(c_pp, axis=0) + np.sum(c_np, axis=0)) / n_train
    leakage_neg = (np.sum(c_pn, axis=0) + np.sum(c_nn, axis=0)) / n_train
    c_eff = float((np.sum(leakage_pos) + np.sum(leakage_neg)) / (2.0 * n_test))

    return {
        "groups": groups,
        "A_pos": a_pos,
        "A_neg": a_neg,
        "B": b_cross,
        "C_pp": c_pp,
        "C_np": c_np,
        "C_pn": c_pn,
        "C_nn": c_nn,
        "boundary_variance": boundary_variance,
        "generation_variance": generation_variance,
        "leakage_pos": leakage_pos,
        "leakage_neg": leakage_neg,
        "c_eff": c_eff,
    }


def simple_test_error(mu: float, terms: dict[str, np.ndarray | float]) -> float:
    generation_variance = float(terms["generation_variance"])
    leakages = np.asarray(terms["per_test_leakage"])
    return float(np.mean(variance_to_error(mu, generation_variance - leakages)))


def one_process_test_error(mu: float, terms: dict[str, np.ndarray | float | SplitGroups], variance: float | None = None) -> float:
    generation_variance = float(terms["generation_variance"] if variance is None else variance)
    leakage_pos = np.asarray(terms["leakage_pos"])
    leakage_neg = np.asarray(terms["leakage_neg"])
    errors = np.r_[
        variance_to_error(mu, generation_variance - leakage_pos),
        variance_to_error(mu, generation_variance - leakage_neg),
    ]
    return float(np.mean(errors))


def solve_full_corrected_variance(mu: float, observed_error: float, leakages: np.ndarray) -> float:
    leakages = np.asarray(leakages, dtype=float)
    lower = float(np.max(leakages) + 1e-8)

    def mean_error(variance: float) -> float:
        return float(np.mean(variance_to_error(mu, variance - leakages)))

    low_value = mean_error(lower) - observed_error
    if low_value >= 0:
        return lower

    upper = max(float(error_to_variance(observed_error, mu) + np.mean(leakages) + 1.0), lower + 1.0)
    high_value = mean_error(upper) - observed_error
    while high_value <= 0:
        upper *= 2.0
        high_value = mean_error(upper) - observed_error
        if upper > 1e6:
            raise RuntimeError("Could not bracket full corrected variance.")

    return float(brentq(lambda value: mean_error(value) - observed_error, lower, upper, xtol=1e-10))


def tune_mu_for_simple_test_error(
    target_error: float,
    terms: dict[str, np.ndarray | float],
    lower: float = 0.05,
    upper: float = 6.0,
) -> float:
    def objective(mu: float) -> float:
        return simple_test_error(mu, terms) - target_error

    while objective(upper) > 0:
        upper *= 1.5
    return float(brentq(objective, lower, upper))



def compute_same_apparent_different_leakage() -> dict[str, pd.DataFrame]:
    n_trials = 96
    train_mask = alternating_block_mask(n_trials, block_size=1)
    rhos = np.array([0.25, 0.45, 0.62, 0.76, 0.86, 0.93])
    target_error = 0.18

    tuned_mu = []
    generalization_errors = []
    for rho in rhos:
        covariance = exponential_temporal_covariance(n_trials, rho=rho, decay=0.55)
        terms = simple_leakage_terms(covariance, train_mask)
        mu = tune_mu_for_simple_test_error(target_error, terms)
        tuned_mu.append(mu)
        generalization_errors.append(variance_to_error(mu, float(terms["generation_variance"])))

    per_rho = pd.DataFrame(
        {
            "rho": rhos,
            "tuned_mu": np.asarray(tuned_mu, dtype=float),
            "generalization_error": np.asarray(generalization_errors, dtype=float),
        }
    )
    summary = pd.DataFrame([{"target_error": float(target_error)}])
    return {"per_rho": per_rho, "summary": summary}



def compute_complex_correction(rng: np.random.Generator, quick: bool) -> dict[str, pd.DataFrame]:
    n_trials = 192
    block_sizes = np.array([1, 2, 4, 8, 16, 32, 96])
    covariance = exponential_temporal_covariance(n_trials, rho=0.94, decay=0.5)
    chol = cholesky(covariance, lower=True)
    mu = 1.25
    n_estimates = 75 if quick else 600
    n_calibration_traces = 8 if quick else 16

    rows: list[dict] = []
    for block_size in block_sizes:
        train_mask = alternating_block_mask(n_trials, int(block_size))
        labels = generate_split_balanced_labels(train_mask, rng)
        terms = one_process_leakage_terms(covariance, labels, train_mask)
        generalization_error = variance_to_error(mu, float(terms["generation_variance"]))
        leaky_error = one_process_test_error(mu, terms)
        leakages = np.r_[np.asarray(terms["leakage_pos"]), np.asarray(terms["leakage_neg"])]
        oracle_scalar_error = scalar_corrected_error(leaky_error, mu, float(terms["c_eff"]))
        oracle_full_error = full_corrected_error(mu, leaky_error, leakages)

        scalar_estimates = []
        full_estimates = []
        for _ in range(n_estimates):
            noise = rng.normal(size=(n_calibration_traces, n_trials)) @ chol.T
            x = mu * labels[None, :] + noise
            residuals = x.copy()
            for label in (-1, 1):
                mask = labels == label
                residuals[:, mask] -= np.mean(residuals[:, mask], axis=1, keepdims=True)
            estimated_covariance = estimate_exponential_covariance(residuals)
            estimated_terms = one_process_leakage_terms(estimated_covariance, labels, train_mask)
            estimated_leakages = np.r_[
                np.asarray(estimated_terms["leakage_pos"]),
                np.asarray(estimated_terms["leakage_neg"]),
            ]
            estimated_scalar_error = scalar_corrected_error(
                leaky_error,
                mu,
                float(estimated_terms["c_eff"]),
            )
            estimated_full_error = full_corrected_error(mu, leaky_error, estimated_leakages)
            scalar_estimates.append(estimated_scalar_error / generalization_error)
            full_estimates.append(estimated_full_error / generalization_error)

        rows.append({"block_size": int(block_size), "method": "leaky",
                     "ratio": float(leaky_error / generalization_error)})
        rows.append({"block_size": int(block_size), "method": "oracle_scalar",
                     "ratio": float(oracle_scalar_error / generalization_error)})
        rows.append({"block_size": int(block_size), "method": "estimated_scalar",
                     "ratio": float(np.mean(scalar_estimates))})
        rows.append({"block_size": int(block_size), "method": "oracle_full",
                     "ratio": float(oracle_full_error / generalization_error)})
        rows.append({"block_size": int(block_size), "method": "estimated_full",
                     "ratio": float(np.mean(full_estimates))})

    per_block_size = pd.DataFrame(rows)

    def mae_for(method: str) -> float:
        ratios = per_block_size.loc[per_block_size["method"] == method, "ratio"].to_numpy()
        return float(np.mean(np.abs(ratios - 1.0)))

    naive_mae = mae_for("leaky")
    oracle_scalar_mae = mae_for("oracle_scalar")
    oracle_full_mae = mae_for("oracle_full")
    estimated_scalar_mae = mae_for("estimated_scalar")
    estimated_full_mae = mae_for("estimated_full")
    summary = pd.DataFrame(
        [
            {
                "naive_mae": naive_mae,
                "oracle_scalar_mae": oracle_scalar_mae,
                "oracle_full_mae": oracle_full_mae,
                "estimated_scalar_mae": estimated_scalar_mae,
                "estimated_full_mae": estimated_full_mae,
                "passes": bool(estimated_scalar_mae < naive_mae and estimated_full_mae < naive_mae),
            }
        ]
    )
    return {"per_block_size": per_block_size, "summary": summary}


def _trial_block_covariance(
    n_outer_trials: int,
    n_timepoints: int,
    within_trial_rho: float,
    cross_trial_rho: float = 0.0,
) -> np.ndarray:
    """Block-structured covariance: exponential within each outer trial, flat across trials.

    Indices ``[0, n_timepoints)`` belong to outer trial 0,
    ``[n_timepoints, 2*n_timepoints)`` to outer trial 1, etc.
    """
    n_total = n_outer_trials * n_timepoints
    indices = np.arange(n_total)
    trial_ids = indices // n_timepoints
    same_trial = trial_ids[:, None] == trial_ids[None, :]
    lag = np.abs(indices[:, None] - indices[None, :])
    return np.where(same_trial, within_trial_rho ** lag, cross_trial_rho)


def _train_mask_from_test_indices(test_indices: tuple[int, ...]) -> np.ndarray:
    train_mask = np.ones(LEAKAGE_DEMO_N_TRIALS, dtype=bool)
    train_mask[list(test_indices)] = False
    return train_mask


def _compute_demo_covariance(*, train_mask: np.ndarray, title: str) -> dict[str, pd.DataFrame]:
    covariance = _trial_block_covariance(
        LEAKAGE_DEMO_N_OUTER_TRIALS,
        LEAKAGE_DEMO_N_TIMEPOINTS,
        LEAKAGE_DEMO_WITHIN_TRIAL_RHO,
        LEAKAGE_DEMO_CROSS_TRIAL_RHO,
    )
    return {
        "covariance": pd.DataFrame(covariance),
        "train_mask": pd.DataFrame({"train": train_mask}),
        "summary": pd.DataFrame(
            [
                {
                    "n_outer_trials": int(LEAKAGE_DEMO_N_OUTER_TRIALS),
                    "n_timepoints": int(LEAKAGE_DEMO_N_TIMEPOINTS),
                    "within_trial_rho": float(LEAKAGE_DEMO_WITHIN_TRIAL_RHO),
                    "cross_trial_rho": float(LEAKAGE_DEMO_CROSS_TRIAL_RHO),
                    "n_train": int(train_mask.sum()),
                    "n_test": int((~train_mask).sum()),
                    "title": title,
                }
            ]
        ),
    }


def compute_cov_independent_train_test() -> dict[str, pd.DataFrame]:
    """Whole-trial holdout — mirrors fold 1 of correct_cv_plot.svg.

    Test = trial 1 (indices 0..2); train = trials 2..5. The train↔test cross
    block is identically zero because the held-out trial shares no within-trial
    timepoints with the training trials.
    """
    return _compute_demo_covariance(
        train_mask=_train_mask_from_test_indices(LEAKAGE_DEMO_CORRECT_TEST_INDICES),
        title="Independent train and test",
    )


def compute_cov_cross_rho_08() -> dict[str, pd.DataFrame]:
    """Scattered-timepoint holdout — mirrors fold 1 of cv_leakage_plot.svg.

    Test = indices 8, 12, 13. Every test timepoint shares its trial with one
    or two train timepoints, so the cross block carries a sparse pattern of
    within-trial correlations rather than being zero.
    """
    return _compute_demo_covariance(
        train_mask=_train_mask_from_test_indices(LEAKAGE_DEMO_LEAKY_TEST_INDICES),
        title=r"Non-zero temporal correlation",
    )


COMPUTE_FUNCTIONS = {
    "cov_independent_train_test": lambda rng, quick: compute_cov_independent_train_test(),
    "cov_cross_rho_08": lambda rng, quick: compute_cov_cross_rho_08(),
    "same_apparent_different_leakage": lambda rng, quick: compute_same_apparent_different_leakage(),
    "complex_correction": lambda rng, quick: compute_complex_correction(rng, quick),
}


# ======================================================================
# HDF5 I/O (pandas DataFrames)
# ======================================================================


def save_panel(h5_path: Path, name: str, frames: dict[str, pd.DataFrame]) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    for sub_key, df in frames.items():
        df.to_hdf(h5_path, key=f"{name}/{sub_key}", mode="a")


def load_panel(h5_path: Path, name: str) -> dict[str, pd.DataFrame]:
    return {
        sub_key: pd.read_hdf(h5_path, key=f"{name}/{sub_key}")
        for sub_key in PANEL_HDF_KEYS[name]
    }


# ======================================================================
# Plot routines
#
# Each plot_* takes the ``dict[str, pd.DataFrame]`` returned by the
# matching compute_*/load_panel and returns a Figure. Saving and closing
# is the caller's job — that keeps the figure live for interactive use.
# ======================================================================


# Per-method visual specs for the multi-line panels. Keys must match the
# ``method`` column values produced by the corresponding compute_*.
COMPLEX_METHOD_SPECS: dict[str, dict] = {
    "leaky":            {"color": sns.color_palette("PiYG")[0], "linestyle": "-",  "label": "leaky test error"},
    "oracle_scalar":    {"color": "#2ca02c", "linestyle": "-",  "label": "oracle $c_\\mathrm{eff}$"},
    "estimated_scalar": {"color": "blue", "linestyle": "--", "label": "estimated $c_\\mathrm{eff}$"},
    "oracle_full":      {"color": "#7c3aed", "linestyle": "-",  "label": "oracle full"},
    "estimated_full":   {"color": "#ff7f0e", "linestyle": "--", "label": "estimated full"},
}


def _plot_block_size_methods(
    df: pd.DataFrame,
    specs: dict[str, dict],
    ax: plt.Axes,
) -> None:
    """Draw one line per ``method`` from a long-format (block_size, method, ratio) frame."""
    for method, spec in specs.items():
        if "oracle" in method:
            continue
        sub = df[df["method"] == method].sort_values("block_size")
        sns.lineplot(
            data=sub,
            x="block_size",
            y="ratio",
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=1.5,
            label=spec["label"],
            ax=ax,
        )


def plot_covariance_heatmap(
    frames: dict[str, pd.DataFrame],
    show_colorbar: bool = True,
) -> plt.Figure:
    """Heatmap of a covariance matrix reordered to put train trials before test trials.

    The reordering makes the train↔test cross block visually obvious: it sits in
    the upper-right and lower-left quadrants, separated from the within-class
    diagonal blocks by a white line.
    """
    covariance = frames["covariance"].to_numpy()
    train_mask = frames["train_mask"]["train"].to_numpy()
    title = str(frames["summary"].iloc[0]["title"])

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(~train_mask)
    order = np.r_[train_indices, test_indices]
    cov_reordered = covariance[np.ix_(order, order)]
    n_train = len(train_indices)
    n_test = len(test_indices)

    fig, ax = plt.subplots(figsize=(3.0, 2))
    cmap = sns.color_palette("gray_r", as_cmap=True)
    im = ax.imshow(
        cov_reordered,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.axvline(n_train - 0.5, color="white", linewidth=1.0)
    ax.axhline(n_train - 0.5, color="white", linewidth=1.0)

    ax.set_xticks([n_train / 2 - 0.5, n_train + n_test / 2 - 0.5])
    ax.set_xticklabels(["train", "test"])
    ax.set_yticks([n_train / 2 - 0.5, n_train + n_test / 2 - 0.5])
    ax.set_yticklabels(["train", "test"])
    ax.tick_params(length=0)

    ax.set_title(title, fontsize=10)
    if show_colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation", ticks=[0.0, 0.5, 1.0])
    [x.patch.set_visible(False) for x in (fig, ax)]
    return fig


def plot_same_apparent_different_leakage(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    per_rho = frames["per_rho"].sort_values("rho")
    target_error = float(frames["summary"].iloc[0]["target_error"])

    fig, ax = plt.subplots(figsize=(2.2, 2))
    sns.lineplot(
        x=per_rho["rho"],
        y=np.full(len(per_rho), target_error),
        color=sns.color_palette("PiYG")[0],
        linewidth=1.5,
        label="leaky test error",
        ax=ax,
    )
    sns.lineplot(
        data=per_rho,
        x="rho",
        y="generalization_error",
        color="black",
        linewidth=1.5,
        label="gen error",
        ax=ax,
    )
    ax.set_xlabel("Averaged temporal autocorrelation")
    # move x-axis to the left to give more space
    ax.xaxis.set_label_coords(0.34, -0.2)
    ax.set_ylabel("Error")
    ax.set_ylim(0.14, max(0.30, float(per_rho["generalization_error"].max()) + 0.035))
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    [x.patch.set_visible(False) for x in (fig, ax)]
    return fig



def plot_complex_correction(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    df = frames["per_block_size"]
    block_sizes = sorted(int(v) for v in df["block_size"].unique())
    # Only plot every other block size to avoid overcrowding the plot.
    block_sizes = block_sizes[::2]

    fig, ax = plt.subplots(figsize=(2.1, 2))
    _plot_block_size_methods(df, COMPLEX_METHOD_SPECS, ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels([str(v) for v in block_sizes])
    ax.set_xlabel("Train/test block size")
    ax.set_ylabel("Error / gen error")
    ax.set_ylim(0.45, 1.08)
    ax.legend(
        frameon=False,
        fontsize=10,
        columnspacing=0.8,
        handlelength=1.4
    )
    [x.patch.set_visible(False) for x in (fig, ax)]
    return fig


PLOT_FUNCTIONS = {
    "cov_independent_train_test": partial(plot_covariance_heatmap, show_colorbar=False),
    "cov_cross_rho_08": plot_covariance_heatmap,
    "same_apparent_different_leakage": plot_same_apparent_different_leakage,
    "complex_correction": plot_complex_correction,
}


# ======================================================================
# CLI orchestration
# ======================================================================


def run_compute(h5_path: Path, seed: int, quick: bool) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        h5_path.unlink()
    rng = np.random.default_rng(seed)

    print(f"Computing panels (seed={seed}, quick={quick})...")
    for name in COMPUTE_ORDER:
        print(f"  {name}...")
        frames = COMPUTE_FUNCTIONS[name](rng, quick)
        save_panel(h5_path, name, frames)
    print(f"Saved results to {h5_path}")


def run_plot(h5_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(out_dir / "main.mplstyle")
    print(f"Plotting panels from {h5_path}...")
    for name in COMPUTE_ORDER:
        frames = load_panel(h5_path, name)
        fig = PLOT_FUNCTIONS[name](frames)
        output_path = out_dir / PANEL_FILENAMES[name]
        fig.savefig(output_path)
        plt.close(fig)
        print(f"  {output_path} ({output_path.stat().st_size} bytes)")


def print_summaries(h5_path: Path) -> bool:
    cov_indep = load_panel(h5_path, "cov_independent_train_test")["summary"].iloc[0]
    same_apparent_frames = load_panel(h5_path, "same_apparent_different_leakage")
    same_apparent = same_apparent_frames["summary"].iloc[0]
    same_apparent_per_rho = same_apparent_frames["per_rho"]
    complex_result = load_panel(h5_path, "complex_correction")["summary"].iloc[0]

    print("\nNumerical summaries:")
    print(
        "  Covariance demos: "
        f"{int(cov_indep['n_outer_trials'])} trials × "
        f"{int(cov_indep['n_timepoints'])} timepoints, "
        f"within-trial rho = {float(cov_indep['within_trial_rho']):.2f} "
        "(panels differ by train/test mask, not by covariance)"
    )
    gen_min = float(same_apparent_per_rho["generalization_error"].min())
    gen_max = float(same_apparent_per_rho["generalization_error"].max())
    print(
        "  Same apparent error: "
        f"target = {float(same_apparent['target_error']):.3f}, "
        f"gen error range = {gen_min:.3f}-{gen_max:.3f}"
    )
    print(
        "  Complex correction: "
        f"leaky ratio MAE = {float(complex_result['naive_mae']):.4f}, "
        f"oracle scalar MAE = {float(complex_result['oracle_scalar_mae']):.4f}, "
        f"estimated scalar MAE = {float(complex_result['estimated_scalar_mae']):.4f}, "
        f"oracle full MAE = {float(complex_result['oracle_full_mae']):.4f}, "
        f"estimated full MAE = {float(complex_result['estimated_full_mae']):.4f}, "
        f"passes = {bool(complex_result['passes'])}"
    )

    return bool(complex_result["passes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporal leakage simulation plots.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("compute", "plot", "all"),
        help="What to do: run simulations, regenerate plots, or both (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        type=Path,
        help=f"Directory for SVG outputs (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--h5-path",
        default=DEFAULT_H5_PATH,
        type=Path,
        help=f"HDF5 file holding intermediate DataFrames (default: {DEFAULT_H5_PATH}).",
    )
    parser.add_argument("--seed", default=143, type=int, help="Random seed (used for compute).")
    parser.add_argument("--quick", action="store_true", help="Use fewer repetitions for a smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in ("compute", "all"):
        run_compute(args.h5_path, args.seed, args.quick)
    if args.mode in ("plot", "all"):
        run_plot(args.h5_path, args.out_dir)

    all_passed = print_summaries(args.h5_path)
    if not all_passed:
        raise SystemExit("At least one numerical acceptance check failed.")


if __name__ == "__main__":
    main()
    sys.exit(0)
