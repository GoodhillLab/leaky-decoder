# %% imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from gamma_poisson import NeuralDataGenerator

plt.style.use("main.mplstyle")

# %% plot

np.random.seed(42)  # For reproducibility

generator = NeuralDataGenerator(100, 61, 1, "gaussian_tuning_curve", {"xi_scale": 1.0})
y_spikes = generator.neural_responses_full
theta_deg = np.rad2deg(generator.stimulus_angles_rad_full)

fig, ax = plt.subplots(figsize=(3, 2))
df = pd.DataFrame(y_spikes.T, columns=theta_deg.round().astype(int))
sns.heatmap(
    df,
    cmap="gray_r",
    cbar_kws={"label": "Spike Count"},
    xticklabels=30,
    yticklabels=20,
    rasterized=True,
    ax=ax,
)
ax.set(xlabel="Stimulus Angle (degrees)", ylabel="Neuron Index")
[x.patch.set_visible(False) for x in (fig, ax)]
# fig.savefig("gaussian_tuning_curve_heatmap.svg")


# %% regression results

theta_hat = cross_val_predict(
    LinearRegression(), y_spikes, theta_deg, cv=LeaveOneOut()
)

# scatter plot
fig, ax = plt.subplots()
sns.scatterplot(
    x=theta_deg,
    y=theta_hat,
    s=15,
    color="black",
    edgecolor="none",
    ax=ax,
)
ax.plot([0, 180], [0, 180], color="gray", linestyle="--")
ax.set(xlabel="True angle", ylabel="Predicted angle")
ax.set_aspect("equal")
[x.patch.set_visible(False) for x in (fig, ax)]

fig.savefig(f"{Path(__file__).stem}.svg", transparent=True)
