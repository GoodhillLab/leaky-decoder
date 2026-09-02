# %%
"""
Null-signal Fisher-ratio sweeps for trial-averaged PCA leakage.

Three demos:
  a. Fixed d = 2000, sweep T: J decreases roughly as kd / [C(T-1) - k - 1].
  b. (d, T) heatmap of J: iso-J bands run along lines of constant d/T.
  c. Sweep T at fixed d/T ratios: J plateaus at k * (d/T) / C.
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm.auto import trange

LIB_PATH = Path.cwd().parent
LIB_PATH = str(LIB_PATH)
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

plt.style.use("main.mplstyle")


# %% helpers
def fisher_ratio(Z, y):
    """Multiclass Fisher ratio J = tr(S_W^{-1} S_B) on projected samples."""
    classes = np.unique(y)
    k = Z.shape[1]
    SB = np.zeros((k, k))
    SW = np.zeros((k, k))
    # below we assume the population is centered
    for c in classes:
        Zc = Z[y == c]
        mu_c = Zc.mean(axis=0)
        SB += len(Zc) * np.outer(mu_c, mu_c)
        SW += (Zc - mu_c).T @ (Zc - mu_c)
    return np.linalg.solve(SW, SB).trace()


def trial_avg_pca_project(X, n_components=2):
    """
    Trial-averaged PCA fit on X of shape (T, C, d); project all T*C trials.
    Returns Z of shape (T*C, n_components) and labels y of shape (T*C,).
    """
    T, C, d = X.shape
    pca = TruncatedSVD(n_components=n_components)
    pca.fit(X.mean(axis=0))
    Z = pca.transform(X.reshape(-1, d))
    y = np.tile(np.arange(C), T)
    return Z, y


def nonleaky_fisher_ratio(X, n_components=2, n_splits=5, random_state=0):
    """
    Per-fold Fisher ratio under no-leakage: PCA is fit on trial-averaged
    training trials, then held-out test trials are projected and scored.
    Each fold computes J in its own PC basis; the scalars are averaged.
    """
    T, C, d = X.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    J_folds = []
    for train_idx, test_idx in kf.split(np.arange(T)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        pca = TruncatedSVD(n_components=n_components)
        pca.fit(X_train.mean(axis=0))
        Z_test = pca.transform(X_test.reshape(-1, d))
        y_test = np.tile(np.arange(C), len(test_idx))
        J_folds.append(fisher_ratio(Z_test, y_test))
    return np.mean(J_folds)


# %% shared parameters
n_stim = 11          # number of conditions C
n_components = 2     # k = PC-projection dimension
n_seeds = 100         # repetitions for mean/std bands
k = n_components
C = n_stim


def J_theory(d, T, C=C, k=k):
    """Expected leaky Fisher ratio under the null (Wishart-inverse + MP limit)."""
    denom = C * (T - 1) - k - 1
    return k * d / denom


def J_noleak_theory(T, C=C, k=k, n_splits=5):
    """
    Expected non-leaky Fisher ratio per fold under the null.
    Independent of d. T is trials-per-condition before splitting.
    """
    T_test = T / n_splits
    denom = C * (T_test - 1) - k - 1
    return k * (C - 1) / denom


# %% Demo a: sweep T at fixed d = 2000 (single master per seed, slice to T)

d_a = 2000
T_values_a = np.array([5, 10, 20, 50, 100, 200, 500], dtype=int)
T_max_a = T_values_a.max()

n_splits_a = 5
J_a = np.full((n_seeds, len(T_values_a)), np.nan)
J_a_noleak = np.full((n_seeds, len(T_values_a)), np.nan)
for si in trange(n_seeds):
    rng = np.random.default_rng(si)
    master = rng.standard_normal((T_max_a, n_stim, d_a))
    for ti, T in enumerate(T_values_a):
        X_sub = master[:T]
        # leaky: global trial-averaged PCA on all trials
        Z, y = trial_avg_pca_project(X_sub, n_components=n_components)
        J_a[si, ti] = fisher_ratio(Z, y)
        # non-leaky: K-fold CV, PCA fit on training trial-averages only
        # K = min(n_splits_a, T // 2)  # ensure T_test >= 2 per fold
        # J_a_noleak[si, ti] = nonleaky_fisher_ratio(
        #     X_sub, n_components=n_components, n_splits=K, random_state=si
        # )
    del master

# %% plot demo a
fig, ax = plt.subplots(figsize=(2, 2))
mean_a = J_a.mean(axis=0)
std_a = J_a.std(axis=0)
mean_a_nl = J_a_noleak.mean(axis=0)
std_a_nl = J_a_noleak.std(axis=0)

# leaky
ax.plot(T_values_a, J_theory(d_a, T_values_a), "--", color="black",
        label="leaky theory")
ax.plot(T_values_a, mean_a, "o-", color="black", label="leaky sim.")
ax.fill_between(T_values_a, mean_a - std_a, mean_a + std_a,
                alpha=0.25, color="black", linewidth=0)
# non-leaky
# K_eff = np.minimum(n_splits_a, T_values_a // 2)
# ax.plot(T_values_a,
#         [J_noleak_theory(T, n_splits=K) for T, K in zip(T_values_a, K_eff)],
#         "--", color="tab:red", label="non-leaky theory")
# ax.plot(T_values_a, mean_a_nl, "s-", color="tab:red", label="non-leaky sim.")
# ax.fill_between(T_values_a, mean_a_nl - std_a_nl, mean_a_nl + std_a_nl,
#                 alpha=0.25, color="tab:red", linewidth=0)

ax.set(xscale="log", yscale="log",
       xlabel="Trials per condition, T",
       ylabel=r"Fisher ratio J")
ax.legend(frameon=False, fontsize=8)
[x.patch.set_visible(False) for x in (fig, ax)]
for ext in ["svg", "pdf"]:
    fig.savefig(f"trial_averaged_pca_T_sweep.{ext}", transparent=True)

# %% Demo b: (d, T) heatmap
d_grid = np.array([100, 200, 500, 1000, 2000, 5000], dtype=int)
T_grid = T_values_a.copy()
J_heat = np.full((n_seeds, len(d_grid), len(T_grid)), np.nan)

d_max_b, T_max_b = d_grid.max(), T_grid.max()
for si in trange(n_seeds):
    rng = np.random.default_rng(si)
    master = rng.standard_normal((T_max_b, n_stim, d_max_b))
    for di, d in enumerate(d_grid):
        for ti, T in enumerate(T_grid):
            Xsub = master[:T, :, :d]
            Z, y = trial_avg_pca_project(Xsub, n_components=n_components)
            J_heat[si, di, ti] = fisher_ratio(Z, y)
    del master

J_heat_mean = J_heat.mean(axis=0)

# %% plot heatmap

fig, ax = plt.subplots(figsize=(3.3, 2))
im = ax.imshow(
    np.log10(J_heat_mean),
    origin="lower",
    aspect="auto",
    cmap="gray_r",
    extent=[np.log10(T_grid[0]), np.log10(T_grid[-1]),
            np.log10(d_grid[0]), np.log10(d_grid[-1])],
)
cb = fig.colorbar(im, ax=ax)
cb.set_label(r"$\log_{10}$(Fisher ratio)")
# iso-J contours at d/T = const (dashed white)
for rho in [1, 5, 25, 100]:
    T_line = np.geomspace(T_grid[0], T_grid[-1], 50)
    d_line = rho * T_line
    mask = (d_line >= d_grid[0]) & (d_line <= d_grid[-1])
    if mask.any():
        ax.plot(np.log10(T_line[mask]), np.log10(d_line[mask]),
                "w--", lw=0.6, alpha=0.7)
        # label at right edge
        if d_line[mask][-1] < d_grid[-1] * 0.95:
            ax.text(np.log10(T_line[mask][-1]) + 0.03,
                    np.log10(d_line[mask][-1]),
                    rf"$d/T={rho}$", color="white", fontsize=7, va="center")

# set heatmap to have square aspect ratio
ax.set(aspect="equal", xlabel=r"$\log_{10}$(#trials per condition)", ylabel=r"$\log_{10}$(#neurons)")
[x.patch.set_visible(False) for x in (fig, ax)]
for ext in ["svg", "pdf"]:
    fig.savefig(f"trial_averaged_pca_heatmap.{ext}", transparent=True)
