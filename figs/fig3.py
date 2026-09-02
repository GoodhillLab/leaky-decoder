# %%
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import make_pipeline

LIB_PATH = Path.cwd().parent
LIB_PATH = str(LIB_PATH)
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)
from utils import TrialAveragePCA

plt.style.use("main.mplstyle")

# %% trial averaged PCA
n_neurons: int = 2000
n_stim: int = 11
n_trials: int = 20
n_components: int = 2


def figure_postprocess(fig,ax):
    """
    Postprocess the figure
    """
    [x.patch.set_visible(False) for x in (fig, ax)]
    ax.set_xlabel("PC1", labelpad=0)
    ax.set_ylabel("PC2", labelpad=0)


np.random.seed(0)
# we generate random data
X = np.random.randn(n_trials, n_stim, n_neurons)
# we average over trials and run PCA
# here we use SVD instead of PCA because the data is inherently centered
pca = TruncatedSVD(n_components=n_components)
X2 = pca.fit_transform(X.mean(0))

fig, ax = plt.subplots()
sns.scatterplot(
    ax=ax,
    x=X2[:, 0],
    y=X2[:, 1],
    hue=np.arange(1, n_stim + 1) * 15,
    legend=False,
    palette="tab20",
    edgecolor="none",
)
figure_postprocess(fig,ax)

X_new = pca.transform(X.reshape(-1, n_neurons))

fig, ax = plt.subplots(figsize=(3, 2))
sns.scatterplot(
    ax=ax,
    x=X_new[:, 0],
    y=X_new[:, 1],
    hue=np.tile(np.arange(1, n_stim + 1) * 15, n_trials),
    legend="full",
    palette="tab20",
    edgecolor="none",
)

# set legend
lgnd = ax.legend(
    ncol=2,
    title="Stimulus angle",
    columnspacing=0,
    fontsize=9,
    title_fontsize=9,
    handletextpad=0,
    loc="upper left",
    borderpad=0.5,
    borderaxespad=0.5,
    frameon=True,
    edgecolor="black",
    bbox_to_anchor=(1.02, 1.0),  # Place legend outside the axes
)

lgnd.get_frame().set_alpha(0.5)
lgnd.get_frame().set_facecolor((1, 1, 1))

# change the marker size manually for the markers in the legend
for count, legend_handle in enumerate(ax.get_legend().legend_handles):
    legend_handle.set(markersize=5)

figure_postprocess(fig,ax)
fig.savefig(f"{Path(__file__).stem}a.svg", transparent=True)

# %% plot fish data in PC space

# use fish data
fname = "data/tectum.npz"

# load the data
with np.load(fname) as f:
    X_pvl = f["X_pvl"]
    X_pvr = f["X_pvr"]
    y = pd.Series(f["y"])

X_dict = {"contra": X_pvl, "ipsi": X_pvr}

# define the model
# Normalizer
normalizer = Normalizer(norm="l1")
scaler = StandardScaler()

for region, X in X_dict.items():
    for pca in [
        PCA(n_components=n_components),
        TrialAveragePCA(n_components=n_components),
    ]:
        model = make_pipeline(normalizer, scaler, pca)
        model.fit(X, y)
        X_new = model.transform(X)

        fig, ax = plt.subplots()
        sns.scatterplot(
            ax=ax,
            x=X_new[:, 0],
            y=X_new[:, 1],
            hue=y,
            legend=False,
            palette="tab20",
            edgecolor="none",
        )
        figure_postprocess(fig,ax)
        if "TrialAverage" in pca.__str__():
            fig.savefig(f"trial_averaged_PCA_{region}.svg", transparent=True)
        else:
            fig.savefig(f"PCA_{region}.svg", transparent=True)
