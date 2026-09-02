# %% this script is used to analytically compute the bias and variance of the PCR estimator
# it supports the following covariance matrices:
# - identity
# - spiked
# - random power law

# %% imports
from typing import Union, Tuple

from loguru import logger as logging
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

# NOTE: import the module, not `ortho_group` itself.  `scipy.stats.ortho_group`
# is a singleton whose `_random_state` *is* numpy's global RandomState.  When a
# function defined in `__main__` refers to that singleton directly, joblib's
# cloudpickle serialises it by value into each worker, which detaches its RNG
# from the global one -- `np.random.seed(run_id)` then no longer controls it and
# the simulation stops being reproducible.  Going through the module keeps the
# reference lazy, so workers use their own correctly seeded singleton.
from scipy import stats as scipy_stats
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys
from joblib import Parallel, delayed


# %% functions
def generate_covariance(
    d: int, sigma_type: str, k: int, decay_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a d x d covariance matrix, its eigenvectors, eigenvalues, and a regression vector beta.
    
    Parameters
    ----------
    d : int
        The dimensionality of the covariance matrix (number of features).
    sigma_type : str
        The type of covariance matrix to generate. Supported options:
            - "identity": Identity matrix (all eigenvalues are 1).
            - "spiked": Two large leading eigenvalues (spikes), rest are 1, normalized to trace d.
            - "random_power_law": Eigenvalues decay as a power law, normalized to trace d.
    k : int
        The number of nonzero directions in beta; beta will be a random linear combination of the first k eigenvectors.
    decay_rate : float, optional
        Used for "random_power_law". Lower values give slower (flatter) decay. Default is 1.0.
    
    Returns
    -------
    Sigma : np.ndarray
        The generated covariance matrix of shape (d, d).
    orth : np.ndarray
        The matrix of eigenvectors for Sigma (random orthogonal matrix, shape (d, d)).
    eigs : np.ndarray
        The eigenvalues of Sigma (length d).
    beta : np.ndarray
        Regression vector in the span of the top k eigenvectors (length d).

    Notes
    -----
    The returned Sigma always has trace d, like the identity, regardless of its eigenspectrum.
    For "spiked", two leading eigenvalues are made large; for "random_power_law", eigenvalues decay, all normalized.
    The returned beta is always a k-sparse (in the eigenbasis) random linear combination of the first k eigenvectors.
    """
    orth = scipy_stats.ortho_group.rvs(dim=d)
    Sigma = np.eye(d)
    eigs = np.ones(d)
    if sigma_type == "identity":
        pass
    elif sigma_type == "spiked":
        # generate a spiked covariance matrix
        eigs[:2] = 3
        # normalize to have `d` trace, same as identity
        eigs *= d / eigs.sum()
        Sigma = orth @ np.diag(eigs) @ orth.T
    elif sigma_type == "random_power_law":
        eigs = np.arange(1, d + 1) ** (-decay_rate)
        # assert it's strictly decreasing
        assert np.all(
            np.diff(eigs) < 0
        ), "Eigenspectrum in this case must be strictly decreasing"
        eigs *= d / eigs.sum()  # Normalize to have `d` trace, same as identity
        Sigma = orth @ np.diag(eigs) @ orth.T
    else:
        raise ValueError(f"Unknown sigma_type: {sigma_type}")

    # generate the beta vector
    coefficients = np.random.randn(k)
    beta = orth[:, :k] @ coefficients

    return Sigma, orth, eigs, beta


def generate_data(
    Sigma: np.ndarray,
    eigenvectors: np.ndarray,
    eigs: np.ndarray,
    beta: np.ndarray,
    n: int,
    target_signal_var: float = 1.0,
    noise_to_signal_ratio: float = 1.0,
) -> Tuple:
    """
    Prepare a scaled regression vector beta, generate a synthetic data matrix X with covariance Sigma,
    and determine the output noise variance sigma2, set for a specific signal-to-noise ratio.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix of the features (shape: d x d).
    eigenvectors : np.ndarray
        Eigenvectors of Sigma (shape: d x d). Columns are directions in eigenspace.
    eigs : np.ndarray
        Eigenvalues of Sigma (shape: d,). Entries correspond to feature variances along the PCs.
    beta : np.ndarray
        Regression vector (shape: d,). Will be rescaled so that Var[x' beta] equals target_signal_var.
    n : int
        Number of training data points/samples to generate.
    target_signal_var : float, optional
        Desired signal variance of x' beta (default: 1.0).
    noise_to_signal_ratio : float, optional
        Sets noise variance as sigma2 = noise_to_signal_ratio x signal variance (default: 1.0).

    Returns
    -------
    beta : np.ndarray
        Regression vector, rescaled so that betaᵀ Sigma beta == target_signal_var. Shape: (d,).
    X : np.ndarray
        Training data matrix with n samples and d features. Each row is drawn from N(0, Sigma).
        Shape: (n, d).
    sigma2 : float
        Output noise variance corresponding to the specified SNR, i.e., sigma2 = noise_to_signal_ratio × target_signal_var.

    Notes
    -----
    - The input beta should be a vector (often in the span of the top-k eigenvectors of Sigma). It is rescaled
      so that its variance under Sigma is target_signal_var.
    - If Sigma is the identity, X is i.i.d. standard normal. Otherwise, X is generated using the eigendecomposition:
      X = Z @ eigenvectors.T, where Z is standard normal with per-direction std sqrt(eigs).
    """
    d = Sigma.shape[0]
    # compute the expected signal variance
    signal_var_raw = float(beta @ Sigma @ beta)
    # scale the beta to the target signal variance
    beta = beta / np.sqrt(signal_var_raw) * np.sqrt(target_signal_var)
    # compute the expected noise variance
    sigma2 = target_signal_var * noise_to_signal_ratio

    # Generate training data X based on the covariance matrix.
    if np.array_equal(Sigma, np.eye(d)):
        X = np.random.normal(size=(n, d))
    else:
        # Instead of np.random.multivariate_normal, use uncorrelated normals and apply transformation
        Z = np.random.normal(scale=np.sqrt(eigs), size=(n, d))
        X = Z @ eigenvectors.T  # project the uncorrelated normals onto the eigenvectors

    return beta, X, sigma2


def compute_proj_components(
    data_matrix: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Computes the top k eigenvectors (Vk) and projection matrix Pk from the raw data matrix.
    """
    n_samples, d = data_matrix.shape
    assert k < min(n_samples, d), "k must be less than both n_samples and d"

    # "Computing PCA components"
    # Setting n_oversamples for accuracy
    n_oversamples: int = 2 * d - k

    # Use randomized_svd for efficiency.
    _, _, Vt = randomized_svd(
        data_matrix, n_components=k, n_oversamples=n_oversamples, random_state=None
    )

    Vk = Vt.T
    Pk = Vk @ Vk.T
    return Vk, Pk, k


def compute_bias_variance_terms_for_point(
    X: np.ndarray,
    x0: np.ndarray,
    k: int,
    Sigma: np.ndarray,
    beta: np.ndarray,
    sigma2: float,
    preprocessor=None,
) -> Tuple[float, float, float, float]:
    """
    Compute Bias^2 and Variance terms for a single test point x0 under:
    - Leaky case: projection derived from [X; x0]
    - Generalization (proper) expectation: using Sigma for bias and trace form for variance

    Returns (leaky_bias_sq, leaky_variance, proper_bias_sq, proper_variance).
    """
    d: int = X.shape[1]
    I_d = np.eye(d)

    if preprocessor is None:
        Vk, Pk, _ = compute_proj_components(np.vstack([X, x0]), k)
    else:
        Vk = preprocessor.components_.T
        Pk = Vk @ Vk.T
        assert Vk.shape == (
            d,
            k,
        ), f"Vk should be a matrix of shape {d}x{k}, but got shape: {Vk.shape}"
        assert Pk.shape == (
            d,
            d,
        ), f"Pk should be a matrix of shape {d}x{d}, but got shape: {Pk.shape}"

    # Bias terms
    tmp_term = Vk @ np.linalg.inv(Vk.T @ X.T @ X @ Vk) @ Vk.T @ X.T @ X
    residual = (tmp_term - I_d) @ beta
    leaky_bias_sq = float((x0.T @ residual) ** 2)
    proper_bias_sq = float(residual.T @ Sigma @ residual)

    # Variance terms
    X_proj_coords = X @ Vk
    if X_proj_coords.shape[1] == 0:
        # No projected dimensions; return zero variances
        return leaky_bias_sq, 0.0, proper_bias_sq, 0.0

    A = np.linalg.inv(X_proj_coords.T @ X_proj_coords)
    x0_proj_coords = Vk.T @ x0
    if x0_proj_coords.ndim == 1:
        x0_proj_coords = x0_proj_coords[..., None]

    tmp = x0_proj_coords.T @ A @ x0_proj_coords
    assert tmp.shape == (1, 1), f"tmp should be a scalar, but got shape: {tmp.shape}"
    leaky_variance = float(sigma2 * tmp[0, 0])
    B_trace = Vk @ A @ Vk.T
    proper_variance = sigma2 * (B_trace @ Sigma).trace()

    return leaky_bias_sq, leaky_variance, proper_bias_sq, proper_variance


def run_single_simulation_run(
    run_id: int,
    d: int,
    k: int,
    num_samples_values: np.ndarray,
    sigma_type: str,
    iters_per_run: int,
    **kwargs,
) -> list:
    """
    Worker function for a single independent run. Generates one master dataset
    and calculates results across different subsample sizes.
    """
    np.random.seed(run_id)  # Ensure each parallel run is different but reproducible
    max_n = num_samples_values.max()
    results_for_this_run = []

    Sigma, eigenvectors, eigs, beta = generate_covariance(d, sigma_type, k, kwargs["decay_rate"])
    # Generate one master dataset for this run
    beta, X_full, sigma2 = generate_data(Sigma, eigenvectors, eigs, beta, max_n, noise_to_signal_ratio=kwargs["noise_to_signal_ratio"])
    assert X_full.shape == (
        max_n,
        d,
    ), f"X_full should be a matrix of shape {max_n}x{d}, but got shape: {X_full.shape}"
    assert beta.shape == (
        d,
    ), f"beta should be a vector of shape {d}, but got shape: {beta.shape}"

    if sigma_type == "identity":
        x0_samples = np.random.normal(size=(iters_per_run, d))
    else:
        Z = np.random.normal(scale=np.sqrt(eigs), size=(iters_per_run, d))
        x0_samples = Z @ eigenvectors.T

    for n in num_samples_values:
        X = X_full[:n]

        run_leaky_biases, run_proper_biases = [], []
        run_leaky_vars, run_proper_vars = [], []

        for x0 in x0_samples:
            leaky_b, leaky_v, proper_b, proper_v = (
                compute_bias_variance_terms_for_point(
                    X=X,
                    x0=x0,
                    k=k,
                    Sigma=Sigma,
                    beta=beta,
                    sigma2=sigma2,
                )
            )
            run_leaky_biases.append(leaky_b)
            run_proper_biases.append(proper_b)
            run_leaky_vars.append(leaky_v)
            run_proper_vars.append(proper_v)

        results_for_this_run.extend(
            [
                {
                    "run_id": run_id,
                    "n_samples": n,
                    "method": "Leaky",
                    "term": "Bias²",
                    "value": np.mean(run_leaky_biases),
                },
                {
                    "run_id": run_id,
                    "n_samples": n,
                    "method": "Proper (Gen. Err.)",
                    "term": "Bias²",
                    "value": np.mean(run_proper_biases),
                },
                {
                    "run_id": run_id,
                    "n_samples": n,
                    "method": "Leaky",
                    "term": "Variance",
                    "value": np.mean(run_leaky_vars),
                },
                {
                    "run_id": run_id,
                    "n_samples": n,
                    "method": "Proper (Gen. Err.)",
                    "term": "Variance",
                    "value": np.mean(run_proper_vars),
                },
            ]
        )
    return results_for_this_run


# %% Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PCR bias-variance simulation and plotting."
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Run the computation and save results to h5.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Load results from h5 and generate plot."
    )
    parser.add_argument(
        "--sigma-types",
        nargs="+",
        default=["spiked", "random_power_law"],
        help="Covariance structures to compute/plot. One h5 file is written per type.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing results store. Off by default: the stores back "
            "published panels, and simulation parameters have changed over time, so "
            "an accidental recompute would silently alter a figure."
        ),
    )
    args = parser.parse_args()

    # Shared parameters
    D = 100
    K = 2
    ITERS_PER_RUN = 2
    NUM_RUNS = 10_000

    num_samples_values = np.arange(20, 61, 5, dtype=int)
    sigma_types_to_test = args.sigma_types
    COVARIANCE_PARAMS = {
        "method": "pcr",
        "decay_rate": 1.0,
        "noise_to_signal_ratio": 3.0,
    }

    def results_file(sigma_type: str) -> Path:
        """One store per covariance structure, so runs never overwrite each other."""
        return Path(f"pcr_simulation_results_{sigma_type}.h5")

    if not args.compute and not args.plot:
        logging.info(
            "No action specified. Use --compute to run simulations or --plot to generate plots."
        )
        sys.exit(0)

    if args.compute:
        logging.info("--- Starting Computation ---")

        for sigma_type in sigma_types_to_test:
            if results_file(sigma_type).is_file() and not args.force:
                logging.warning(
                    f"{results_file(sigma_type)} already exists; skipping. "
                    "Pass --force to recompute."
                )
                continue
            logging.info(
                f"\n===== Running Parallel Simulation Set for sigma='{sigma_type}' ====="
            )
            sigma_results = []

            with Parallel(n_jobs=-1) as parallel:
                jobs = [
                    delayed(run_single_simulation_run)(
                        i,
                        D,
                        K,
                        num_samples_values,
                        sigma_type,
                        ITERS_PER_RUN,
                        **COVARIANCE_PARAMS,
                    )
                    for i in range(NUM_RUNS)
                ]
                results_from_runs = parallel(
                    tqdm(jobs, desc=f"Parallel Runs ({sigma_type})")
                )

                # Flatten the list of lists into a single list
                for run_result_list in results_from_runs:
                    for result in run_result_list:
                        result["sigma_type"] = sigma_type
                        sigma_results.append(result)

            output_path = results_file(sigma_type)
            pd.DataFrame(sigma_results).to_hdf(
                output_path, key="results", mode="w", complevel=1
            )
            logging.info(f"Computation complete. Results saved to {output_path}")

    if args.plot:
        logging.info("--- Starting Plotting ---")
        fname = Path("main.mplstyle")
        if fname.is_file():
            logging.info(f"Using custom style from {fname}")
            plt.style.use(fname)

        frames = []
        for sigma_type in sigma_types_to_test:
            input_path = results_file(sigma_type)
            if not input_path.is_file():
                logging.error(
                    f"Results file not found: {input_path}. Please run with --compute first."
                )
                sys.exit(1)
            frames.append(pd.read_hdf(input_path))
        all_results_df = pd.concat(frames, ignore_index=True)

        # change some entries in the `method` column to be more descriptive
        all_results_df["method"] = all_results_df["method"].map(
            {
                "Leaky": "Leaky CV",
                "Proper (Gen. Err.)": "Gen err",
            }
        )
        cmap = {
            "Leaky CV": sns.color_palette("PiYG")[0],
            "CV": sns.color_palette("PiYG")[-1],
            "Gen err": "black",
        }

        sigma_types = all_results_df["sigma_type"].unique()
        for sigma in sigma_types:
            data_sigma = all_results_df.query("sigma_type == @sigma")
            terms = data_sigma["term"].unique()
            assert len(terms) == 2, "Expected exactly two terms: Bias² and Variance"
            fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharex=True)
            for i, term in enumerate(terms):
                ax = axes[i]
                sns.lineplot(
                    ax=ax,
                    data=data_sigma.query("term == @term"),
                    x="n_samples",
                    y="value",
                    hue="method",
                    palette=cmap,
                    marker="o",
                    legend=bool(i),
                    markerfacecolor="auto",
                    markeredgewidth=0,
                )
                if i:
                    ax.legend(columnspacing=0.3, handletextpad=0.3, frameon=False)
                ax.set(xlabel="", ylabel=terms[i])
                ax.patch.set_visible(False)
            # plot the sum of bias and variance
            ax = axes[2]
            # group the rows with same `run_id`, `n_samples`, `method`, and sum the `value`
            df_tmp = (
                data_sigma.groupby(["run_id", "n_samples", "method"])
                .agg({"value": "sum"})
                .reset_index()
            )
            sns.lineplot(
                ax=ax,
                data=df_tmp,
                x="n_samples",
                y="value",
                hue="method",
                palette=cmap,
                marker="o",
                legend=False,
                markerfacecolor="auto",
                markeredgewidth=0,
            )
            ax.patch.set_visible(False)
            ax.set(xlabel="", ylabel="Bias² + Variance")
            fig.supxlabel("Number of samples")
            fig.patch.set_visible(False)
            plot_filename = f"bias_variance_vs_samples_plot_{sigma}_pcr.svg"
            fig.savefig(plot_filename)
            logging.info(f"Plot saved to {plot_filename}")
