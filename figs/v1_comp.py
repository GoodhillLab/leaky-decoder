# %% imports
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
plt.style.use("main.mplstyle")

data_dir = Path("data") / "v1"
f_type = {"figs": "Leaky CV", "figs_corrected": "CV"}
cmap = {
    "Leaky CV": sns.color_palette("PiYG")[0],
    "CV": sns.color_palette("PiYG")[-1],
}

# %% plot the decoding error for different number of PCs

fig, ax = plt.subplots()

for dirname in ("figs_corrected", "figs"):
    color = cmap[f_type[dirname]]
    d = np.load(data_dir / dirname / f"pcdecode.npy", allow_pickle=True).item()
    errors, nPC = d["errors"], d["nPC"]
    for k in range(len(nPC)):
        ax.errorbar(
            nPC[k],
            errors[:, k].mean(axis=0),
            errors[:, k].std(axis=0) / np.sqrt(6),
            markersize=20,
            color=color,
            zorder=5,
            alpha=0.5,
        )
        label = None
        if k == 0:
            label = f_type[dirname]
        ax.scatter(nPC[k], errors[:, k].mean(axis=0), s=3, color=color, zorder=10, label=label)

ax.legend(loc=(0.4, 0.2), frameon=False, fontsize=10, handletextpad=0.1, handlelength=1)
ax.set(xlabel="Number of PCs", ylabel="Decoding error (deg)", xscale="log")
[x.patch.set_visible(False) for x in (fig, ax)]
fig.savefig("compare_stringer_pcr.svg")


# %% plot the median error for different animals

res = []
for dirname in f_type:
    fname = data_dir / dirname / "linear_decoder_asymp.npy"
    d = np.load(fname, allow_pickle=True).item()
    Elin = d['E']
    median_error = Elin[0,0,:]

    # Load the data
    df_tmp = pd.DataFrame({
        'median_error': median_error,
        'mouse': np.arange(len(median_error))
    })
    df_tmp["type"] = f_type[dirname]
    res.append(df_tmp)

# Concatenate the dataframes
df = pd.concat(res, ignore_index=True)

fig, ax = plt.subplots(figsize=(1.7, 2))
sns.barplot(data=df, x='type', y='median_error', width=0.4, ax=ax, fill=None, color="black", errorbar=None)
sns.stripplot(data=df, x='type', y='median_error', alpha=0.5, jitter=False, color='black', palette=cmap)
# add lines connecting the points
for i in range(len(df)//2):
    x = [0, 1]
    y = [df['median_error'][i], df['median_error'][i+len(df)//2]]
    ax.plot(x, y, color='black', alpha=0.5)

ax.set(xlabel="", ylabel="Median error", ylim=(0.88, None))

# paired Wilcoxon test
_, p = wilcoxon(*df.groupby("type")['median_error'].apply(list))
ax.text(0.5, 1.25, f"p={p:.2f}", fontsize=10, ha="center", va="center")

[x.patch.set_visible(False) for x in (fig, ax)]
fig.savefig("linear_decoder_asymp.svg", transparent=True)


# %% calculate and plot the percent change in error for different downsample percentages

res = []
for dirname in f_type:
    for percent in np.logspace(-2, 0, 10):
        if percent == 1:
            fname = 'linear_decoder_asymp.npy'
        else:
            fname = f'linear_decoder_asymp_percent_{percent:.2f}.npy'
        d = np.load(data_dir / dirname / fname, allow_pickle=True).item()
        Elin = d['E']
        median_error = Elin[0,0,:]

        # Load the data
        df_tmp = pd.DataFrame({
            'median_error': median_error,
            'mouse': np.arange(len(median_error)),
            'percent': percent,
            'type': f_type[dirname]
        })
        res.append(df_tmp)

df = pd.concat(res, ignore_index=True)

# %% Calculate the ratio of error between leakage and no leakage for each percent value using query

ratio_list = []
for p in df['percent'].unique():
    df_leakage = df.query("percent == @p and type == 'Leaky CV'").set_index('mouse')
    df_no_leakage = df.query("percent == @p and type == 'CV'").set_index('mouse')
    # Only keep mouse IDs present in both conditions
    common_mice = df_leakage.index.intersection(df_no_leakage.index)
    err_leakage = df_leakage.loc[common_mice]['median_error'].values
    err_no_leakage = df_no_leakage.loc[common_mice]['median_error'].values
    # Ratio: leakage / no_leakage
    ratio = err_leakage / err_no_leakage
    ratio_list.append(pd.DataFrame({'percent': p, 'error_ratio': ratio}))

ratio_df = pd.concat(ratio_list, ignore_index=True)

fig, ax = plt.subplots(figsize=(2.5, 2))
sns.lineplot(data=ratio_df, x='percent', y='error_ratio', errorbar="sd", ax=ax, color="black")
ax.set(xlabel="Downsample ratio", ylabel="Error ratio\n(leakage / no leakage)", xscale="log")
[x.patch.set_visible(False) for x in (fig, ax)]
for ext in ("svg", "pdf"):
    fig.savefig(f"stringer_error_ratio.{ext}", transparent=True)
