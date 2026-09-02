# %% imports
from pathlib import Path
from typing import Dict
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, LeaveOneOut
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.auto import tqdm, trange
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LIB_PATH = Path.cwd().parent
LIB_PATH = str(LIB_PATH)
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)
from utils import TrialAveragePCA
from io_utils import save_shuffle_control_results

# set style
plt.style.use("main.mplstyle")

# %% use fish data
fname = "data/tectum.npz"

# load the data
with np.load(fname) as f:
    X_pvl = f["X_pvl"]
    X_pvr = f["X_pvr"]
    y = pd.Series(f["y"])

# %% we illustrate that data leakage thru feature selection

n_stim = len(y.unique())
scaler = StandardScaler()
lda = LDA()
pca = TrialAveragePCA(n_components=3, random_state=0)

# define the model
model = make_pipeline(scaler, pca, lda)
# stratified is used to ensure the same class distribution in each split
# this is better suited in the settings of trial-average PCA because
# it averages the trials within each class
cv = StratifiedShuffleSplit(n_splits=20, random_state=42)

# %% Run decoding with and without data leakage
df_tmp = []
for X in [X_pvl, X_pvr]:
    if X is X_pvl:
        region = "contra"
    else:
        region = "ipsi"

    # feature selection: data leakage
    X_new = model[:-1].fit_transform(X, y)
    # run decoding
    scores = cross_val_score(model[-1], X_new, y, cv=cv)
    df_tmp.append(
        pd.DataFrame({"accuracy": scores, "method": "leakage", "region": region})
    )

    # feature selection and decoding: no data leakage
    scores = cross_val_score(model, X, y, cv=cv)
    df_tmp.append(
        pd.DataFrame({"accuracy": scores, "method": "no leakage", "region": region})
    )

df_tmp = pd.concat(df_tmp)

# %% shuffle control at the first level

shuffle_res: Dict[str, np.ndarray] = {}
for X in [X_pvl, X_pvr]:
    if X is X_pvl:
        region = "contra"
    else:
        region = "ipsi"

    # Fixed permutation seeds keep the shuffle control reproducible.
    scores = [
        cross_val_score(model, X, y.sample(frac=1, random_state=seed), cv=cv, n_jobs=-1)
        for seed in trange(100, desc=f"Shuffle control for {region}")
    ]

    shuffle_res[region] = np.array(scores)

# %% persist what the panel shows
# The source-data export reads this file rather than re-running the simulation,
# so the workbook and the figure can never drift apart.
save_shuffle_control_results(__file__, df_tmp, shuffle_res)

# %% Plot the results

cmap = sns.color_palette("PiYG")
palette = {"leakage": cmap[1], "no leakage": cmap[-2]}
regions = ["contra", "ipsi"]
fig, ax = plt.subplots(figsize=(3, 2), layout="constrained")

np.random.seed(42)

sns.barplot(
    data=df_tmp,
    x="region",
    y="accuracy",
    hue="method",
    ax=ax,
    width=0.4,
    order=regions,
    color="black",
    errorbar=None,
    fill=False,
    legend=False,
    linewidth=1,
)

# plot the shuffle control for contra as a shaded rectangle

for i, region in enumerate(regions):
    ax.fill_between(
        [-0.3 + i, 0.3 + i],
        *np.percentile(shuffle_res[region].mean(1), [2.5, 97.5]),
        color="gray",
        alpha=0.5,
    )

    for j, method in enumerate(["leakage", "no leakage"]):
        df_tmp2 = df_tmp.query(f"region == '{region}' and method == '{method}'")
        offset = -0.1 if j == 0 else 0.1
        label = method if i == 0 else None

        ax.scatter(
            x=(i + offset) * (np.ones(len(df_tmp2)))
            + np.random.uniform(-0.05, 0.05, size=len(df_tmp2)),
            y=df_tmp2["accuracy"],
            color=palette[method],
            s=10,
            edgecolor=None,
            label=label,
        )

sns.despine()

# set legend outside
ax.legend(
    ncol=2,
    columnspacing=0.2,
    fontsize=8,
    handletextpad=0.2,
    loc=(0.1, 0.97),
)

ax.set(xlabel="Tectum", ylabel="Decoding accuracy")
# move y-label down
ax.yaxis.set_label_coords(-0.2, 0.4)
[x.patch.set_visible(False) for x in (fig, ax)]
fig.savefig(f"{Path(__file__).stem}.svg", transparent=True)
