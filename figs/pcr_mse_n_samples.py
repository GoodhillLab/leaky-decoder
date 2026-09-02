# %% imports
from pathlib import Path
from joblib import Parallel, delayed
import platform
import os
import sys

from loguru import logger as logging
import numpy as np
import pandas as pd
from tqdm.auto import trange
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from bias_variance_decomp import generate_data, generate_covariance

# %% Shared parameters
d: int = 100
K: int = 2
max_n: int = 60
sigma_type: str = "random_power_law"
pre_type: str = "PCA"
N_CPUS: int = int(os.environ.get("SLURM_NTASKS_PER_NODE", -1))
logging.info(f"Using {N_CPUS} CPUs")

# %% parameters

# Simulation hyper‑parameters
N = max_n  # sample size
N_test = 100
p = d  # raw feature dimension

# we assume oracle knows "2 components" should be used
preprocessor = TruncatedSVD(n_components=K)
lr = LinearRegression(fit_intercept=False)
pipeline = make_pipeline(preprocessor, lr)
iters_per_run: int = 100
N_sim: int = 10_000


def calc_stats(y0_samples: np.ndarray, y0_true: np.ndarray, y0_pred: np.ndarray) -> pd.DataFrame:
    assert len(y0_pred) > 1, "y0_pred must have more than 1 element"
    # calculate the mean squared error
    score = mean_squared_error(y0_samples, y0_pred)
    score2 = mean_squared_error(y0_true, y0_pred)

    return pd.DataFrame({"Type": ["from sample", "from true signal"], "MSE": [score, score2]})


# %% main loop

def run_single_iteration(
    seed: int,
    d: int = d,
    K: int = K,
    max_n: int = max_n,
    sigma_type: str = sigma_type,
    noise_to_signal_ratio: float = 1.0,
    iters_per_run: int = iters_per_run,
    N_test: int = N_test,
    p: int = p,
    preprocessor=None,
    lr=None,
) -> list:
    """
    Run a single simulation for a single `seed` (one full iteration).
    Returns a list of results for that seed.
    """

    if preprocessor is None:
        preprocessor = TruncatedSVD(n_components=K)
    if lr is None:
        lr = LinearRegression(fit_intercept=False)
    res = []

    rng = np.random.default_rng(seed)
    Sigma, eigenvectors, eigs, beta = generate_covariance(d, sigma_type, k=K, decay_rate=1.0)
    beta, X_full, sigma2 = generate_data(
        Sigma,
        eigenvectors,
        eigs,
        beta,
        n=max_n,
        noise_to_signal_ratio=noise_to_signal_ratio,
    )
    y_signal = X_full @ beta
    y_full = y_signal + np.random.normal(size=(max_n), scale=np.sqrt(sigma2))

    if sigma_type == "identity":
        x0_samples = np.random.normal(size=(iters_per_run, d))
    else:
        x0_samples = np.random.multivariate_normal(
            mean=np.zeros(d), cov=Sigma, size=iters_per_run
        )

    y0_true = x0_samples @ beta
    y0_samples = y0_true + np.random.normal(size=iters_per_run, scale=np.sqrt(sigma2))

    # create a pipeline
    pipeline = make_pipeline(preprocessor, lr)


    for n_samples in np.arange(20, max_n + 1, 5):
        X_train = X_full[:n_samples]
        y_train = y_full[:n_samples]
        # no leakage case
        pipeline.fit(X_train, y_train)
        y0_pred = pipeline.predict(x0_samples)
        df_tmp = calc_stats(y0_samples, y0_true, y0_pred)
        df_tmp["Method"] = "CV"
        df_tmp["n_samples"] = n_samples
        res.append(df_tmp)

        beta_hat = preprocessor.components_.T @ lr.coef_
        beta_hat_mse = ((beta_hat - beta) ** 2).mean()
        df_tmp = pd.DataFrame(
            {"Type": ["Parameter"], "MSE": [beta_hat_mse], "Method": ["CV"]}
        )
        df_tmp["n_samples"] = n_samples
        res.append(df_tmp)

        # pre-allocate memory for the leaky predictions
        y0_leaky_pred = np.zeros_like(y0_pred)
        y0_leaky_pred[:] = np.nan
        for i, x0 in enumerate(x0_samples):
            # leaky case: fit global
            preprocessor.fit(np.vstack([X_train, x0]))
            Z_train = preprocessor.transform(X_train)
            lr.fit(Z_train, y_train)
            # evaluate on the same sample used in the dimension reduction step
            y_pred = pipeline.predict([x0])
            assert y_pred.shape == (1,)
            y0_leaky_pred[i] = y_pred[0]

            # calculate the MSE of the parameter estimate (between beta hat and beta)
            beta_hat = preprocessor.components_.T @ lr.coef_
            beta_hat_mse = mean_squared_error(beta, beta_hat)
            df_tmp = pd.DataFrame(
                {"Type": ["Parameter"], "MSE": [beta_hat_mse], "Method": ["Leaky CV"]}
            )
            df_tmp["n_samples"] = n_samples
            res.append(df_tmp)

            # generate the held-out test data
            X_new = rng.multivariate_normal(np.zeros(p), Sigma, size=N_test)
            # evaluate the out-of-sample MSE using new data
            y_new_signal = X_new @ beta
            y_new = y_new_signal + rng.normal(scale=np.sqrt(sigma2), size=N_test)
            y_new_pred = pipeline.predict(X_new)
            df_tmp = calc_stats(y_new, y_new_signal, y_new_pred)
            df_tmp["Method"] = "Gen err"
            df_tmp["n_samples"] = n_samples
            res.append(df_tmp)


        # calculate MSE
        df_tmp = calc_stats(y0_samples, y0_true, y0_leaky_pred)
        df_tmp["Method"] = "Leaky CV"
        df_tmp["n_samples"] = n_samples
        res.append(df_tmp)

    res = pd.concat(res)
    res["Seed"] = seed
    res["noise_to_signal_ratio"] = noise_to_signal_ratio

    return res


# %% run the simulations

noise_to_signal_ratio = 3

res = Parallel(n_jobs=N_CPUS)(
    delayed(run_single_iteration)(
        seed,
        d=d,
        K=K,
        max_n=max_n,
        sigma_type=sigma_type,
        noise_to_signal_ratio=noise_to_signal_ratio,
        iters_per_run=iters_per_run,
        preprocessor=preprocessor,
    )
    for seed in trange(N_sim, desc="Running simulations")
)

df = pd.concat(res)

# %% construct the dataframe

for col, dtype in [
    ("noise_to_signal_ratio", float),
    ("MSE", float),
    ("Method", str),
    ("Type", str),
    ("Seed", int),
]:
    df[col] = df[col].astype(dtype)

fname = f"{pre_type}_empirical_results_{sigma_type}_n_samples.h5"

# df.to_hdf(fname, key="data", mode="w", complevel=1)

# %% load the dataframe

# df = pd.read_hdf(fname, key="data")
df_tmp = df.groupby(["Seed", "n_samples", "Method", "Type"]).mean()

# %% set color palette

cmap = {
    "Leaky CV": sns.color_palette("PiYG")[0],
    "CV": sns.color_palette("PiYG")[-1],
    "Gen err": "black",
}

plt.style.use("main.mplstyle")

# %% plots

for value_to_plot in df["Type"].unique():
    fig, ax = plt.subplots()
    legend_flag = True if value_to_plot == "from sample" else False

    sns.lineplot(
        data=df_tmp.query("Type == @value_to_plot"),
        x="n_samples",
        y="MSE",
        hue="Method",
        marker="o",
        ax=ax,
        palette=cmap,
        markerfacecolor="auto",  # ensures marker and edge color match for no white circle
        markeredgewidth=0,  # removes possible visible white border
        legend=legend_flag,
    )

    ylabel = value_to_plot
    if value_to_plot == "Bias":
        ylabel = "Bias²"
    elif value_to_plot == "from true signal":
        ylabel = "Reducible error in MSE"
    elif value_to_plot == "from sample":
        ylabel = "MSE"
    ax.set(
        ylabel=ylabel,
        xlabel="Number of samples",
        # yscale="log",
        # ylim=(1, None),
    )
    ax.legend(columnspacing=0.3, handletextpad=0.3, frameon=False)
    [x.patch.set_visible(False) for x in (fig, ax)]

    fig.savefig(f"{pre_type}_{value_to_plot}_{sigma_type}_n_samples.svg", transparent=True)
