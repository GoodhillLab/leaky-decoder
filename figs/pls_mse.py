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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from bias_variance_decomp import generate_data, generate_covariance
sys.path.append(str(Path(__file__).parent.parent))
from utils import PLS

# %% Shared parameters
d: int = 100
K: int = 1
max_n: int = 60
sigma_type: str = "random_power_law"
pre_type: str = "PLS"
# Simulation hyper‑parameters
N_test: int = 100
N_CPUS: int = int(os.environ.get("SLURM_NTASKS_PER_NODE", -1))
logging.info(f"Using {N_CPUS} CPUs")


def calc_stats(y0_samples: np.ndarray, y0_pred: np.ndarray) -> pd.DataFrame:
    assert len(y0_pred) > 1, "y0_pred must have more than 1 element"
    # calculate the mean squared error
    score = root_mean_squared_error(y0_samples, y0_pred)
    return pd.DataFrame({"Type": ["from sample"], "RMSE": [score]})


# %% main loop

def run_single_iteration(
    pipeline: Pipeline,
    seed: int,
    d: int = d,
    K: int = K,
    max_n: int = max_n,
    sigma_type: str = sigma_type,
    noise_to_signal_ratio: float = 1.0,
    iters_per_run: int = 100,
    N_test: int = N_test,
    p: int = 100,
) -> list:
    """
    Run a single simulation for a single `seed` (one full iteration).
    Returns a list of results for that seed.
    """

    preprocessor = pipeline.named_steps["pls"]
    lr = pipeline.named_steps["linearregression"]

    rng = np.random.default_rng(seed)
    Sigma, eigenvectors, eigs, beta = generate_covariance(d, sigma_type, k=1, decay_rate=1.0)
    beta, X_full, sigma2 = generate_data(
        Sigma,
        eigenvectors,
        eigs,
        beta,
        n=max_n,
        noise_to_signal_ratio=noise_to_signal_ratio,
    )
    y_full = X_full @ beta + np.random.normal(size=(max_n), scale=np.sqrt(sigma2))

    if sigma_type == "identity":
        x0_samples = np.random.normal(size=(iters_per_run, d))
    else:
        x0_samples = np.random.multivariate_normal(
            mean=np.zeros(d), cov=Sigma, size=iters_per_run
        )

    y0_samples = x0_samples @ beta + np.random.normal(size=iters_per_run, scale=np.sqrt(sigma2))
    res = []
    for n_samples in np.arange(20, max_n + 1, 5):
        X_train = X_full[:n_samples]
        y_train = y_full[:n_samples]

        # no leakage case
        # RMSE from held-out test data
        pipeline.fit(X_train, y_train)
        y0_pred = pipeline.predict(x0_samples)
        df_tmp = calc_stats(y0_samples, y0_pred)
        df_tmp["Method"] = "CV"
        df_tmp["n_samples"] = n_samples
        res.append(df_tmp)

        # RMSE of the parameter estimate
        beta_hat = preprocessor.components_.T @ lr.coef_
        beta_hat_rmse = root_mean_squared_error(beta, beta_hat)
        df_tmp = pd.DataFrame(
            {"Type": ["Parameter"], "RMSE": [beta_hat_rmse], "Method": ["CV"]}
        )
        df_tmp["n_samples"] = n_samples
        res.append(df_tmp)

        # leaky case
        # allocate memory for the leaky predictions
        y0_leaky_pred = np.zeros_like(y0_pred)
        y0_leaky_pred[:] = np.nan
        for i, (x0, y0) in enumerate(zip(x0_samples, y0_samples, strict=True)):
            # leaky case: fit global
            preprocessor.fit(np.vstack([X_train, x0]), np.hstack([y_train, y0]))
            lr.fit(preprocessor.transform(X_train), y_train)
            # evaluate on the same sample used in the dimension reduction step
            y_pred = pipeline.predict([x0])
            assert y_pred.shape == (1,)
            y0_leaky_pred[i] = y_pred[0]

            # calculate the MSE of the parameter estimate (between beta hat and beta)
            beta_hat = preprocessor.components_.T @ lr.coef_
            beta_hat_rmse = root_mean_squared_error(beta, beta_hat)
            df_tmp = pd.DataFrame(
                {"Type": ["Parameter"], "RMSE": [beta_hat_rmse], "Method": ["Leaky CV"]}
            )
            df_tmp["n_samples"] = n_samples
            res.append(df_tmp)

            # generate the held-out test data
            X_new = rng.multivariate_normal(np.zeros(p), Sigma, size=N_test)
            y_new = X_new @ beta + rng.normal(scale=np.sqrt(sigma2), size=N_test)
            # evaluate the out-of-sample MSE using new data
            y_new_pred = pipeline.predict(X_new)
            df_tmp = calc_stats(y_new, y_new_pred)
            df_tmp["Method"] = "Gen err"
            df_tmp["n_samples"] = n_samples
            res.append(df_tmp)


        df_tmp = calc_stats(y0_samples, y0_leaky_pred)
        df_tmp["Method"] = "Leaky CV"
        df_tmp["n_samples"] = n_samples
        res.append(df_tmp)

    res = pd.concat(res)
    res["Seed"] = seed
    res["noise_to_signal_ratio"] = noise_to_signal_ratio

    return res


# %% run the simulations

# below we use the PLS implementation from the bias_variance_decomp package
# instead of the one from sklearn, since we assume centered data
# similarly, OLS is done without intercept as well
preprocessor = PLS()
lr = LinearRegression(fit_intercept=False)
pipeline = make_pipeline(preprocessor, lr)
iters_per_run: int = 2_00
N_sim: int = 10_000
noise_to_signal_ratio = 1.0

res = Parallel(n_jobs=N_CPUS)(
    delayed(run_single_iteration)(
        pipeline,
        seed,
        d=d,
        K=K,
        max_n=max_n,
        sigma_type=sigma_type,
        noise_to_signal_ratio=noise_to_signal_ratio,
        iters_per_run=iters_per_run,
    )
    for seed in trange(N_sim, desc="Running simulations")
)

df = pd.concat(res)

# %% construct the dataframe

for col, dtype in [
    ("noise_to_signal_ratio", float),
    ("RMSE", float),
    ("Method", str),
    ("Type", str),
    ("Seed", int),
]:
    df[col] = df[col].astype(dtype)

# df.to_hdf(f"{pre_type}_empirical_results.h5", key="data", mode="w", complevel=1)

# %% average the results

# df = pd.read_hdf(f"{pre_type}_empirical_results.h5", key="data")
df_tmp = df.groupby(["Seed", "noise_to_signal_ratio", "n_samples", "Method", "Type"]).mean()

# %% set color palette

cmap = {
    "Leaky CV": sns.color_palette("PiYG")[0],
    "CV": sns.color_palette("PiYG")[-1],
    "Gen err": "black",
}

plt.style.use("main.mplstyle")

# %% plot the results

for value_to_plot in df["Type"].unique():
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_tmp.query("Type == @value_to_plot"),
        x="n_samples",
        y="RMSE",
        hue="Method",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
        palette=cmap,
        markerfacecolor="auto",  # ensures marker and edge color match for no white circle
        markeredgewidth=0,  # removes possible visible white border
    )
    ax.set(xlabel="Number of samples")
    ax.legend(fontsize=8, columnspacing=0.3, handletextpad=0.3, frameon=False)
    [x.patch.set_visible(False) for x in (fig, ax)]
    fig.savefig(f"{pre_type}_{value_to_plot}.svg", transparent=True)
