# %% imports
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt

from SimulationHelper import (
    SimulationRandomMean,
    SimulationGaussianTuningCurve,
    EyeCovarianceEstimator,
)

# set style
plt.style.use("main.mplstyle")


def add_lineplots_and_legend(
    fig, axes, results_df, N_neuron_list: List[int], chance_acc: float, cmap_dict: Dict
):
    for ax, method, ls, leg in zip(
        axes[1:], ["pre", "proper_pre"], [None, "dashed"], ["full", None]
    ):
        sns.lineplot(
            ax=ax,
            data=results_df.query(f"method=='{method}'"),
            x="N_neurons",
            y="accuracy",
            hue="preprocessor",
            marker="o",
            markersize=4,
            palette=cmap_dict,
            linestyle=ls,
            legend=leg,
        )
        ax.set(xlabel="Number of neurons", ylabel="Accuracy")
        # add horizontal line
        ax.hlines(
            chance_acc,
            N_neuron_list.min(),
            N_neuron_list.max(),
            color="red",
            label="chance level",
        )
        ax.hlines(
            0.6915,
            N_neuron_list.min(),
            N_neuron_list.max(),
            color="green",
            label="upper bound",
        )
        ax.set_ylim(0.48, 1)

    # set legend on the second axis
    ax = axes[1]
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "decoding accuracy"
    ax.legend(
        handles=handles,
        frameon=False,
        labels=labels,
        loc=(0.05, 0.6),
        title="",
        fontsize=8,
    )

    # hide patches for fig and all axes
    for x in [fig] + list(axes.flatten()):
        x.patch.set_visible(False)


# %% set up parameters for simulations
preprocessor = SelectKBest(f_classif, k=100)

angles = [0, 1]
chance_acc = 1 / len(angles)
n_simulations = 100
cv = StratifiedKFold(n_splits=5)
# Define the number of trials per stimulus type
M = 300
# Define the visual stimuli
y = np.repeat(angles, M)
cmap_dict = {"SelectKBest": "black"}

# %% run simulations
N_neuron_list = 10 * (np.array([2, 4, 8, 12, 16, 24, 32]) + 10)
res = []
np.random.seed(42)

sim = SimulationRandomMean(
    y,
    preprocessor,
    max_n_neurons=N_neuron_list.max(),
    cv=cv,
)
# Run all simulations across all specified neuron counts in a single call
results_df = sim.run_simulations(
    n_simulations=n_simulations, n_neuron_list=N_neuron_list, n_jobs=8
)

# %% plot example firing rate with two neurons

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(6, 2))

ax = axes[0]
ax.scatter(
    *np.random.normal(loc=(10, 10), size=(M, 2)).T,
    color="black",
    alpha=0.5,
    edgecolor="none",
    label="0",
)

ax.scatter(
    *np.random.normal(loc=(11, 10), size=(M, 2)).T,
    color="gray",
    alpha=0.5,
    edgecolor="none",
    label="1",
)

ax.legend(title="Stimulus", facecolor="white", fontsize=8, title_fontsize=8)
# make markers in the legend non transparent
for handle in ax.legend_.legend_handles:
    handle.set_alpha(1.0)
ax.set(xlabel="Neuron 1 firing rate", ylabel="Neuron 2 firing rate", aspect="equal")
# move y label down
ax.yaxis.set_label_coords(-0.2, 0.4)

add_lineplots_and_legend(fig, axes, results_df, N_neuron_list, chance_acc, cmap_dict)
fig.savefig("fig2a_S1_ab.svg", transparent=True)


# %% run simulations with Gaussian tuning curves

# run simulations with larger number of neurons
np.random.seed(42)

sim = SimulationGaussianTuningCurve(
    y,
    preprocessor,
    cov=np.eye(N_neuron_list.max()),
    max_n_neurons=N_neuron_list.max(),
    cv=cv,
)
# Run all simulations across all specified neuron counts in a single call
results_df = sim.run_simulations(
    n_simulations=n_simulations, n_neuron_list=N_neuron_list, n_jobs=8
)

# %% plots

N_curves = 15

fig, axes = plt.subplots(1, 3, figsize=(6, 2))

ax = axes[0]
x = np.linspace(-np.pi, np.pi, 100)
mu_list = np.linspace(-np.pi, np.pi, N_curves)
cmap_tmp = sns.color_palette("gist_rainbow", N_curves)

for i, mu in enumerate(mu_list):
    y_tmp = SimulationGaussianTuningCurve.gaussian(x, mu)
    sns.lineplot(ax=ax, x=x, y=y_tmp, color=cmap_tmp[i], linewidth=0.8)
ax.set(xlabel="Stimulus", ylabel="Response")

add_lineplots_and_legend(fig, axes, results_df, N_neuron_list, chance_acc, cmap_dict)

fig.savefig("fig2_cd.svg", transparent=True)


# %% ---------------------------------------------------------------------------
# Mechanism of F-statistic leakage
#
# For each CV fold of each simulated dataset, we rank neurons two ways:
#   - leaky: top-k by F-statistic computed on ALL trials (train + test)
#   - clean: top-k by F-statistic computed on TRAIN trials only
# For each selection, we record the mean per-neuron "leaked F increment"
#   Delta F_i = F_i^all - F_i^train
# across the k selected neurons (x-axis), together with the LDA test-fold
# accuracy (y-axis). Delta F measures how much a neuron's ranking score was
# pushed up by including the held-out trials in the F computation. If the
# inflated accuracy is driven by test-fold leakage into selection, the leaky
# cloud should sit at clearly positive Delta F and elevated accuracy, while
# the clean cloud should hug Delta F approximately 0 at chance accuracy.
# -----------------------------------------------------------------------------

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def _mechanism_one_sim(sim_idx: int, N: int, k: int, y_arr: np.ndarray, cv_splits):
    """Simulate one dataset and, for each fold, record per-method (x, y)."""
    np.random.seed(sim_idx)
    sim_local = SimulationGaussianTuningCurve(
        y_arr,
        SelectKBest(f_classif, k=k),
        cov=np.eye(N),
        max_n_neurons=N,
    )
    X = sim_local.generate_data()

    F_all, _ = f_classif(X, y_arr)
    # neurons selected by global F
    sel_all = np.argsort(-F_all)[:k]
    clf = LDA(solver="lsqr", covariance_estimator=EyeCovarianceEstimator())
    records = []
    for fold_idx, (tr, te) in enumerate(cv_splits):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_arr[tr], y_arr[te]

        F_train, _ = f_classif(X_tr, y_tr)
        # per-neuron "leaked F increment": how much the held-out trials
        # pushed this neuron's F score above its train-only value.
        dF = F_all - F_train
        # neurons selected by train-only F
        sel_train = np.argsort(-F_train)[:k]

        for method, sel in (("leakage", sel_all), ("no leakage", sel_train)):
            mean_dF = dF[sel].mean()
            clf.fit(X_tr[:, sel], y_tr)
            acc = clf.score(X_te[:, sel], y_te)

            records.append(
                dict(
                    sim_idx=sim_idx,
                    fold_idx=fold_idx,
                    method=method,
                    mean_dF=mean_dF,
                    accuracy=acc,
                )
            )
    return records


# Fix the CV fold boundaries so every simulation uses identical splits;
# differences between clouds then reflect the selection step alone.
_cv_mech = StratifiedKFold(n_splits=5)
cv_splits = list(_cv_mech.split(np.zeros_like(y), y))

N_neurons_mech = int(N_neuron_list.max())  # regime with clearest inflation in Fig. 2d
k_select_mech = 100
n_sims_mech = 50

records_nested = Parallel(n_jobs=8)(
    delayed(_mechanism_one_sim)(i, N_neurons_mech, k_select_mech, y, cv_splits)
    for i in trange(n_sims_mech, desc="Mechanism simulations")
)
mech_df = pd.DataFrame([r for rec in records_nested for r in rec])

# %% plot mechanism scatter

cmap = sns.color_palette("PiYG")
palette_mech = {"leakage": cmap[1], "no leakage": cmap[-2]}

fig, ax = plt.subplots(figsize=(3, 2))
sns.scatterplot(
    ax=ax,
    data=mech_df,
    x="mean_dF",
    y="accuracy",
    hue="method",
    palette=palette_mech,
    s=12,
    alpha=0.7,
    edgecolor="none",
)

ax.axhline(chance_acc, color="red", linewidth=1, label="chance")
ax.axhline(0.6915, color="green", linewidth=1, label="upper bound")
# ax.axvline(0, color="gray", linestyle=":", linewidth=1)
ax.set(
    xlabel="Mean F increment\n"
    r"$\overline{\Delta F} = \overline{F^{\mathrm{all}} - F^{\mathrm{train}}}$"
    "\nover selected neurons",
    ylabel="Test-fold\ndecoder accuracy",
    # ylim=(0.48, 1.0),
)
# move legend outside of the plot
ax.legend(bbox_to_anchor=(1.05, 1))
[x.patch.set_visible(False) for x in (fig, ax)]
for ext in ["svg", "pdf"]:
    fig.savefig(f"fig2_S_mechanism.{ext}", transparent=True)
