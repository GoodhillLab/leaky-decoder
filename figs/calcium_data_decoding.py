# %% imports and functions
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from joblib import Parallel, delayed
from loguru import logger as logging
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    LeaveOneGroupOut,
    cross_val_score,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from tqdm.auto import trange
from scipy.stats import wilcoxon


def gen_sinusoidal_data(g=[0.98], sn=0.3, T=3000, framerate=30, b=0, N=20, beta=None):
    """
    Generate data from inhomogenous Poisson Process with sinusoidal instantaneous activity
    Adapted from [github.com/j-friedrich/OASIS](https://github.com/j-friedrich/OASIS/blob/master/oasis/functions.py#L63)

    Parameters
    ----------
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    b : float, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    beta : array, shape (2,), optional, default None
        Projection matrix from stimulus to neural firing rate.

    Returns
    -------
    y : array, shape (T,)
        Noisy fluorescence data.
    """
    # in radians
    angle = np.linspace(0, np.pi, T)
    theta = np.array([np.cos(angle), np.sin(angle)])
    firing_rate = np.exp(beta @ theta)  # Instantaneous firing rate
    firing_rate /= framerate  # Convert to Hz

    trueSpikes = np.random.rand(N, T) < firing_rate
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    Y = b + truth + sn * np.random.randn(N, T)
    return Y


def generate_and_process_sinusoidal_data(run_id: int, params: dict):
    """
    Generates calcium data based on a sinusoidal signal and processes it for decoding.

    This function creates a single continuous block of data and splits it into
    CV and test sets. It also creates groups for time-series cross-validation.

    Returns:
        A tuple containing (X_cv, y_cv, groups_cv, X_test, y_test).
    """
    T: int = params["num_frames"]
    framerate = params["calcium_imaging_rate"]

    # 1. Generate target variable (y) based on the same underlying signal
    # This matches the signal used internally by gen_sinusoidal_data
    y = np.linspace(0, np.pi, T)

    # 2. Generate neural activity data for cross-validation and test sets
    N_neuron = params["num_neurons"]
    np.random.seed(run_id)  # Ensure reproducibility for each run
    beta = np.random.randn(N_neuron, 2)
    X_cv = [
        gen_sinusoidal_data(T=T, framerate=framerate, N=N_neuron, b=1, beta=beta).T
        for _ in range(params["num_cv_trials"])
    ]
    X_test = [
        gen_sinusoidal_data(T=T, framerate=framerate, N=N_neuron, b=1, beta=beta).T
        for _ in range(params["num_test_trials"])
    ]

    return (
        np.vstack(X_cv),
        np.tile(y, params["num_cv_trials"]),
        np.repeat(np.arange(params["num_cv_trials"]), T),
        np.vstack(X_test),
        np.tile(y, params["num_test_trials"]),
    )


def evaluate_decoding_performance(
    X_cv, y_cv, groups_cv, X_test, y_test, run_id: int, params: dict
):
    """
    Evaluates decoding performance using different CV strategies.
    This function is now independent of the data generation process.
    """
    # 1. Define Model and Hyperparameter Grid
    regressor = make_pipeline(PCA(n_components=0), LinearRegression())
    param_grid = {"pca__n_components": np.arange(1, params["num_neurons"])}

    # --- Scenario 1: Leaky CV (KFold with shuffle=True) ---
    cv_leaky = KFold(n_splits=5, shuffle=True, random_state=run_id)
    grid_search = GridSearchCV(
        regressor, param_grid, scoring="neg_mean_squared_error", cv=cv_leaky
    )
    # we evaluate the RMSE of the model on the cross-validation set
    # average first then take the square root instead of taking the square root first then averaging
    # this is because the RMSE is a non-linear function and we want to avoid the bias introduced by the non-linearity
    leaky_rmse = np.sqrt(-cross_val_score(grid_search, X_cv, y_cv, cv=cv_leaky).mean())

    # --- Scenario 2: Proper Time-Series CV (Leave-One-Group-Out) ---
    logo_cv = LeaveOneGroupOut()
    # The inner CV for GridSearchCV should also respect the grouped structure
    grid_search.cv = logo_cv
    scores = cross_validate(
        grid_search,
        X_cv,
        y_cv,
        cv=logo_cv,
        params={"groups": groups_cv},  # Pass group info
        return_estimator=True,
    )
    proper_cv_rmse = np.sqrt(-scores["test_score"].mean())

    # --- Generalization Error Calculation on Held-Out Test Set ---
    gen_error = [-estimator.score(X_test, y_test) for estimator in scores["estimator"]]
    generalization_error_rmse = np.sqrt(np.mean(gen_error))

    logging.info(
        f"Finished run #{run_id}. Leaky: {leaky_rmse:.2f}, Proper: {proper_cv_rmse:.2f}, Gen. Error: {generalization_error_rmse:.2f}"
    )

    return {
        "run_id": run_id,
        "leaky_rmse": leaky_rmse,
        "proper_cv_rmse": proper_cv_rmse,
        "generalization_error_rmse": generalization_error_rmse,
    }


def run_single_simulation(run_id: int, params: dict) -> dict:
    """
    Main worker function for a single simulation run. This is the target for joblib.
    Orchestrates data generation and performance evaluation.
    """
    # Set the config inside the worker function to ensure it's active in this process.
    sklearn.set_config(enable_metadata_routing=True)

    # Step 1: Generate and process data using the simplified method
    X_cv, y_cv, groups_cv, X_test, y_test = generate_and_process_sinusoidal_data(
        run_id, params
    )

    # Step 2: Evaluate decoding performance on the generated data
    return evaluate_decoding_performance(
        X_cv, y_cv, groups_cv, X_test, y_test, run_id, params
    )


# %% main execution
if __name__ == "__main__":
    # --- Simulation Parameters ---
    params = {
        "num_runs": 100,  # Total number of independent simulation runs
        "num_neurons": 50,
        "num_frames": 300,  # Number of frames in each trial
        "num_cv_trials": 20,  # Number of trials for cross-validation
        "num_test_trials": 10,  # Large number of trials for stable generalization error
        "calcium_imaging_rate": 40,  # Hz
    }

    # --- Run Simulations in Parallel ---
    logging.info(f"Starting {params['num_runs']} simulation runs...")
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, params) for i in trange(params["num_runs"])
    )
    # --- Save Results to Database ---
    results_df = pd.DataFrame(results)
    db_file = "calcium_results.db"
    with sqlite3.connect(db_file) as conn:
        results_df.to_sql("decoding_errors", conn, if_exists="replace", index=False)
    conn.close()
    logging.info(f"Successfully saved results to {db_file}")
    sys.exit()


# %% Visualize the first neuron of an example trial

# Set plot style (ensure this file exists or comment it out)
plt.style.use("main.mplstyle")

np.random.seed(42)  # For reproducibility
beta = np.random.randn(50, 2)  # Random projection matrix
T = 300  # Number of frames
framerate = 40  # Hz

X = gen_sinusoidal_data(T=T, framerate=framerate, N=50, b=1, beta=beta)

fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(np.arange(T) / framerate, X[10], color="black")
ax.set(xlabel="Time (seconds)", ylabel="Calcium signal (arb. units)")
# move y-axis label down
ax.yaxis.set_label_coords(-0.12, 0.4)
[x.patch.set_visible(False) for x in (fig, ax)]
fig.savefig("calcium_trace_example.svg", transparent=True)

# %% example decoding results

params = {
    "num_neurons": 50,
    "num_frames": 300,  # Number of frames in each trial
    "num_cv_trials": 10,  # Number of trials for cross-validation
    "num_test_trials": 1,
    "calcium_imaging_rate": 40,  # Hz
}

X_cv, y_cv, groups_cv, _, _ = generate_and_process_sinusoidal_data(
    run_id=42, params=params
)

sklearn.set_config(enable_metadata_routing=True)

clf = make_pipeline(PCA(n_components=-1), LinearRegression())


for cv_params in (
    {"cv": KFold(n_splits=5, shuffle=True, random_state=42)},
    {"cv": LeaveOneGroupOut(), "params": {"groups": groups_cv}},
):
    search_cv = GridSearchCV(
        clf,
        param_grid={"pca__n_components": np.arange(1, params["num_neurons"])},
        scoring="neg_root_mean_squared_error",
        cv=cv_params["cv"],
    )
    y_hat = cross_val_predict(search_cv, X_cv, y_cv, **cv_params)
    # plot results
    fig, ax = plt.subplots()
    ax.scatter(np.rad2deg(y_hat), np.rad2deg(y_cv), s=1, color="black", alpha=0.2)
    ax.set(xlabel="Predicted angle (deg)", ylabel="True angle (deg)")
    ax.set_aspect("equal")
    # add diagonal line
    ax.plot([0, 180], [0, 180], color="green", lw=1.5, ls="--")
    # add R^2 to the plot
    r2 = r2_score(y_cv, y_hat)
    ax.text(
        0.75, 0.05, f"$R^2$={r2:.2f}", transform=ax.transAxes, ha="right", va="bottom"
    )

    [x.patch.set_visible(False) for x in (fig, ax)]
    fig.savefig(
        f"calcium_data_decoding_example_{type(cv_params['cv']).__name__}.svg",
        transparent=True,
    )

# %% Visualize Results
logging.info("Plotting results...")
plt.style.use("main.mplstyle")

with sqlite3.connect("calcium_results.db") as conn:
    results_df = pd.read_sql("SELECT * FROM decoding_errors", conn)
conn.close()

results_df.rename(
    columns={
        "leaky_rmse": "Rand CV",
        "proper_cv_rmse": "LOGO CV",
        "generalization_error_rmse": "Gen err",
    },
    inplace=True,
)

# Melt the DataFrame for easier plotting with seaborn
results_melted = results_df.melt(
    id_vars=["run_id"],
    value_vars=[
        "Rand CV",
        "LOGO CV",
        "Gen err",
    ],
    var_name="RMSE_Type",
    value_name="RMSE_Value",
)
# convert radians to degrees
results_melted["RMSE_Value"] = np.rad2deg(results_melted["RMSE_Value"])

fig, ax = plt.subplots()
sns.boxplot(
    data=results_melted,
    x="RMSE_Type",
    y="RMSE_Value",
    ax=ax,
    width=0.5,
    fill=None,
    color="black",
)

ax.set(ylabel="RMSE (deg)", xlabel="Method")
# set x-ticks font size
ax.tick_params(axis="x", labelsize=8)

# Perform Wilcoxon signed-rank test
leaky_rmse = np.rad2deg(results_df["Rand CV"])
non_leaky_rmse = np.rad2deg(results_df["LOGO CV"])
gen_error_rmse = np.rad2deg(results_df["Gen err"])
s1, p1 = wilcoxon(leaky_rmse, non_leaky_rmse)
s2, p2 = wilcoxon(leaky_rmse, gen_error_rmse)
s3, p3 = wilcoxon(non_leaky_rmse, gen_error_rmse)
logging.info(f"Wilcoxon test results:")
logging.info(f"Rand CV vs LOGO CV: statistic={s1}, p-value={p1}")
logging.info(f"Rand CV vs Gen error: statistic={s2}, p-value={p2}")
logging.info(f"LOGO CV vs Gen error: statistic={s3}, p-value={p3}")

x1, x2 = 0.1, 0.9
y = 16.5
h = 0.5
plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
plt.text((x1 + x2) * 0.5, y + h * 1.5, f"p=1e-17", ha="center", va="bottom")

x1, x2 = 0, 2
y = 19
plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
plt.text((x1 + x2) * 0.5, y + h * 1.5, f"p=1e-17", ha="center", va="bottom")

x1, x2 = 1.1, 1.9
y = 17
plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
plt.text((x1 + x2) * 0.5, y + h * 1.5, f"p={p3:.1f}", ha="center", va="bottom")


[x.patch.set_visible(False) for x in (fig, ax)]
fig.savefig("calcium_data_decoding_results.svg", transparent=True)
