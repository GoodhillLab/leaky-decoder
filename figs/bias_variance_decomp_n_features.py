# %% this script looks at the bias and variance of the PCR estimator as a function of the number of features

# %% imports
from loguru import logger as logging
import numpy as np
import pandas as pd
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys
from sklearn.utils.extmath import randomized_svd
from joblib import Parallel, delayed
from bias_variance_decomp import (
    generate_data,
    compute_bias_variance_terms_for_point,
    generate_covariance,
)


def run_single_simulation_run(
    run_id: int,
    k: int,
    num_features_values: np.ndarray,
    sigma_type: str,
    iters_per_run: int,
    **kwargs,
) -> list:
    """
    Worker function for a single independent run. Generates one master dataset
    and calculates results across different subsample sizes.
    """
    assert (
        k <= num_features_values.min()
    ), "k must be less than or equal to the minimum number of features"
    np.random.seed(run_id)  # Ensure each parallel run is different but reproducible
    max_n = 50
    results_for_this_run = []

    Sigma_full, _, _, beta_full = generate_covariance(
        num_features_values.max(), sigma_type, k, kwargs["decay_rate"]
    )
    for d in num_features_values:
        # Generate one master dataset for this run
        Sigma = Sigma_full[:d, :d]
        # perform eigendecomposition on Sigma
        eigs, eigenvectors = np.linalg.eigh(Sigma)
        # sort it descendingly
        eigs = eigs[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        beta, X_full, sigma2 = generate_data(
            Sigma,
            eigenvectors,
            eigs,
            beta_full[:d],
            max_n,
            noise_to_signal_ratio=kwargs["noise_to_signal_ratio"],
        )
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

        X = X_full.copy()

        run_leaky_vars, run_proper_vars = [], []
        run_approx_vars1, run_approx_vars2 = [], []

        # compute the analytical out-of-sample variance for the case where the test point is not included in the training data
        # for each eigenvector and eigenvalue pair, compute the variance
        # $V(\bm{x}) = \sigma^2_\epsilon \sum_{i=1}^k \frac{\hat{v}_j^\top \Sigma \hat{v}_j}{\hat{\lambda}_i}$
        _, S, eigenvectors = randomized_svd(X, n_components=k)
        out_of_sample_var = 0.0
        for eig, eigenvec in zip(S**2, eigenvectors):
            out_of_sample_var += sigma2 * eigenvec.T @ Sigma @ eigenvec / eig
        run_approx_vars1.append(out_of_sample_var)

        for x0 in x0_samples:
            _, leaky_v, _, proper_v = compute_bias_variance_terms_for_point(
                X=X,
                x0=x0,
                k=k,
                Sigma=Sigma,
                beta=beta,
                sigma2=sigma2,
            )
            run_leaky_vars.append(leaky_v)
            run_proper_vars.append(proper_v)

            # $V(\bm{x}) = \frac{\sigma^2_{\epsilon}}{n} \sum_{i=1}^k \frac{z_i^2}{\lambda_i}$
            # use SVD directly on X_full which is already centered
            _, S, eigenvectors = randomized_svd(np.vstack([X, x0]), n_components=k)
            Z = x0 @ eigenvectors.T
            approx_v = sigma2 * np.sum(Z**2 / S**2)
            run_approx_vars2.append(approx_v)

        results_for_this_run.extend(
            [
                {
                    "run_id": run_id,
                    "n_features": d,
                    "method": "Leaky",
                    "term": "Variance",
                    "value": np.mean(run_leaky_vars),
                },
                {
                    "run_id": run_id,
                    "n_features": d,
                    "method": "Proper (Gen. Err.)",
                    "term": "Variance",
                    "value": np.mean(run_proper_vars),
                },
                {
                    "run_id": run_id,
                    "n_features": d,
                    "method": "Approx1",
                    "term": "Variance",
                    "value": np.mean(run_approx_vars1),
                },
                {
                    "run_id": run_id,
                    "n_features": d,
                    "method": "Approx2",
                    "term": "Variance",
                    "value": np.mean(run_approx_vars2),
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
    args = parser.parse_args()

    # Shared parameters
    D = None
    K = 20
    ITERS_PER_RUN = 2
    NUM_RUNS = 3_000

    num_features_values = np.array([100, 300, 500, 1000, 2000])
    sigma_types_to_test = ["random_power_law"]
    COVARIANCE_PARAMS = {
        "method": "pcr",
        "decay_rate": 1.0,
        "noise_to_signal_ratio": 3.0,
    }
    RESULTS_FILE = f"{Path(__file__).stem}_results.h5"

    if not args.compute and not args.plot:
        logging.info(
            "No action specified. Use --compute to run simulations or --plot to generate plots."
        )
        sys.exit(0)

    if args.compute:
        logging.info("--- Starting Computation ---")
        all_results = []

        for sigma_type in sigma_types_to_test:
            logging.info(
                f"\n===== Running Parallel Simulation Set for sigma='{sigma_type}' ====="
            )

            with Parallel(n_jobs=-1) as parallel:
                jobs = [
                    delayed(run_single_simulation_run)(
                        i,
                        K,
                        num_features_values,
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
                        all_results.append(result)

        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_hdf(RESULTS_FILE, key="results", mode="w", complevel=1)
        logging.info(f"Computation complete. Results saved to {RESULTS_FILE}")

    if args.plot:
        logging.info("--- Starting Plotting ---")
        fname = Path("main.mplstyle")
        if fname.is_file():
            logging.info(f"Using custom style from {fname}")
            plt.style.use(fname)

        if not Path(RESULTS_FILE).is_file():
            logging.error(
                f"Results file not found: {RESULTS_FILE}. Please run with --compute first."
            )
            sys.exit(1)
        all_results_df = pd.read_hdf(RESULTS_FILE)

        # change some entries in the `method` column to be more descriptive
        all_results_df["method"] = all_results_df["method"].replace(
            {
                "Leaky": "Leaky CV",
                "Proper (Gen. Err.)": "Gen err",
                "Approx1": "Analytical out-of-sample",
                "Approx2": "Empirical in-sample",
            },
        )
        cmap = {
            "Leaky CV": sns.color_palette("PiYG")[0],
            "CV": sns.color_palette("PiYG")[-1],
            "Gen err": "black",
            "Analytical out-of-sample": "gray",
            "Empirical in-sample": "pink",
        }
        marker_map = {
            "Leaky CV": "o",
            "CV": "o",
            "Gen err": "s",
            "Analytical out-of-sample": "o",
            "Empirical in-sample": "o",
        }

        sigma_types = all_results_df["sigma_type"].unique()
        for sigma in sigma_types:
            term = "Variance"
            df_tmp = all_results_df.query("sigma_type == @sigma and term == @term")
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6, 2))
            sns.lineplot(
                ax=ax,
                data=df_tmp,
                x="n_features",
                y="value",
                hue="method",
                style="method",
                palette=cmap,
                markers=marker_map,
                legend=True,
                markerfacecolor="auto",
                markeredgewidth=0,
                alpha=0.6,
            )

            ax.set(xlabel="Number of neurons", ylabel=term, xscale="log")
            # line plot of the analytical in-sample variance (sigma^2 * k / n)
            ax.plot(
                num_features_values,
                3.0 * K / 51 * np.ones(len(num_features_values)),
                color="red",
                linestyle="--",
                label="Analytical in-sample",
            )

            # set legend to the left of the figure panel
            lgd = ax.legend(
                columnspacing=0.3,
                handletextpad=0.3,
                frameon=False,
                bbox_to_anchor=(-0.25, 0.5),
                loc="center right",
                borderaxespad=0.0,
                fontsize=10,
            )
            # set handles in legend to non-opaque
            for handle in lgd.legend_handles:
                handle.set_alpha(1)

            # plot the ratio between the in-sample variance and the analytical out-of-sample variance
            df_tmp2 = (
                df_tmp.query("method == 'Analytical out-of-sample'")
                .copy()
                .reset_index(drop=True)
            )
            df_tmp2["Ratio"] = (3.0 * K / 51) / df_tmp2["value"]
            sns.lineplot(
                ax=ax2,
                data=df_tmp2,
                x="n_features",
                y="Ratio",
                color="red",
                marker="o",
                markerfacecolor="auto",
                markeredgewidth=0,
            )

            # divide the Leaky CV variance by the analytical out-of-sample variance
            df_tmp3 = df_tmp.query("method == 'Leaky CV' or method == 'Gen err'").copy().reset_index(drop=True)
            df_tmp3 = df_tmp3.pivot(
                index=["run_id", "n_features", "term", "sigma_type"],
                columns="method",
                values="value",
            ).reset_index()
            df_tmp3["Ratio_Leaky"] = df_tmp3["Leaky CV"] / df_tmp3["Gen err"]
            sns.lineplot(
                ax=ax2,
                data=df_tmp3,
                x="n_features",
                y="Ratio_Leaky",
                color=cmap["Leaky CV"],
                marker="o",
                markerfacecolor="auto",
                markeredgewidth=0,
            )
            ax2.set(xlabel="Number of neurons", ylabel="Variance ratio\ncompare to gen err", xscale="log")

            [x.patch.set_visible(False) for x in (fig, ax, ax2)]
            plot_filename = f"tmp_{sigma}_pcr.svg"
            fig.savefig(plot_filename)
            logging.info(f"Plot saved to {plot_filename}")
