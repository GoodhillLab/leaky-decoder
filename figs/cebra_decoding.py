# %% import libraries
from pathlib import Path
import argparse
from loguru import logger as logging

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.pipeline import make_pipeline

from cebra import CEBRA, KNNDecoder

plt.style.use("main.mplstyle")

# %% Define a custom splitter to avoid temporal leakage


class CustomSplitter(BaseCrossValidator):
    def __init__(self, trial_times: np.ndarray[int]):
        self.trial_times = trial_times

    def _iter_test_masks(self, X, y=None, groups=None):
        T = X.shape[0]
        for i, t1 in enumerate(self.trial_times):
            if i == 0:
                continue
            test_idx = np.zeros(T, dtype=bool)
            t0 = self.trial_times[i - 1]
            test_idx[t0:t1] = True
            yield test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.trial_times) - 1


# %% Make the CEBRA estimators usable inside an sklearn Pipeline


class SklearnKNNDecoder(KNNDecoder):
    """KNNDecoder that reports its fitted state to scikit-learn.

    ``KNNDecoder.fit`` stores the fitted model in ``self.knn``, an attribute
    without a trailing underscore, so ``check_is_fitted`` treats the estimator
    as unfitted. Inside a ``Pipeline`` that surfaces as a ``NotFittedError`` on
    ``predict`` (a ``FutureWarning`` before scikit-learn 1.8).
    """

    def fit(self, X, y):
        super().fit(X, y)
        self.knn_ = self.knn  # trailing underscore marks the estimator as fitted
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "knn")


class UnsupervisedCEBRA(CEBRA):
    """CEBRA that discards the labels ``Pipeline.fit`` forwards to every step.

    With ``conditional="time"`` the labels are unused for sampling anyway;
    dropping them keeps the unsupervised arm verifiably label-free.
    """

    def fit(self, X, y=None, **kwargs):
        return super().fit(X, **kwargs)


if __name__ == "__main__":
    # %% parse arguments
    parser = argparse.ArgumentParser(
        description="CEBRA decoding with conditional parameter"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="supervised",
        help='Conditional parameter: "supervised" or "unsupervised" (default: supervised)',
    )
    args = parser.parse_args()

    # Set conditional parameter
    if args.condition == "supervised":
        conditional_param = "time_delta"
        cebra_cls = CEBRA
    elif args.condition == "unsupervised":
        conditional_param = "time"
        cebra_cls = UnsupervisedCEBRA
    else:
        raise ValueError(f"Invalid condition: {args.condition}")

    # Determine CSV filename based on condition; results live alongside the
    # other saved inputs in data/, which is also where export_source_data.py
    # reads them from.
    csv_file = (
        Path(__file__).parent / "data" / f"{Path(__file__).stem}_{args.condition}.csv"
    )

    # Check if CSV file exists
    if Path(csv_file).exists():
        logging.info(f"Loading results from {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        # %% load data
        rat_data = sio.loadmat("data/Achilles_data.mat")
        X, y = rat_data["spikes"], rat_data["loc"].squeeze()
        X.shape, y.shape

        # %% fit with cebra

        max_iterations = 10000  # default is 5000.
        output_dimension = (
            32  # here, we set as a variable for hypothesis testing below.
        )
        X = rat_data["spikes"].copy()

        cebra_pos_model = cebra_cls(
            model_architecture="offset10-model",
            batch_size=512,
            learning_rate=3e-4,
            temperature=1,
            output_dimension=output_dimension,
            max_iterations=max_iterations,
            distance="cosine",
            conditional=conditional_param,
            device="cuda_if_available",
            verbose=True,
            time_offsets=10,
        )

        # UnsupervisedCEBRA ignores y, so the same call covers both conditions
        cebra_pos_model.fit(X, y)
        embedding = cebra_pos_model.transform(X)

        embedding.shape

        # %% run cross validation on the embedding, leakage

        trial_t = rat_data["trial"].squeeze()
        T = y.shape[0]

        splitter = CustomSplitter(trial_t)

        knn = SklearnKNNDecoder(n_neighbors=36, metric="cosine")
        leakage_score = cross_val_score(
            knn,
            embedding,
            y,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=6,
        )

        # %% no leakage, long runtime!

        pipeline = make_pipeline(cebra_pos_model, knn)
        no_leakage_score = cross_val_score(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            verbose=10,
        )

        # %% compute the p-value for the difference in performance

        df1 = pd.DataFrame(no_leakage_score, columns=["RMSE (cm)"])
        df1["method"] = "CV"
        df2 = pd.DataFrame(leakage_score, columns=["RMSE (cm)"])
        df2["method"] = "Leaky CV"

        df = pd.concat([df1, df2])
        # convert to positive and meters to cm
        df["RMSE (cm)"] *= -100

        # Save results to CSV
        print(f"Saving results to {csv_file}")
        df.to_csv(csv_file, index=False)

    # %% generate the figure

    stats_res = wilcoxon(*df.groupby("method")["RMSE (cm)"].apply(list))

    fig, ax = plt.subplots(figsize=(2, 2))
    sns.boxplot(
        ax=ax,
        data=df,
        x="method",
        y="RMSE (cm)",
        color="k",
        fill=None,
        width=0.3,
        linewidth=1,
    )

    ax.set(xlabel="")

    # manually add the p-value; keep the bracket clear of the highest outlier
    # (the unsupervised arm reaches ~44.9 cm, which the fixed height of 45 would
    # have crossed). The floor of 45 leaves the supervised panel unchanged.
    bar_y = max(45, df["RMSE (cm)"].max() * 1.1)
    ax.plot([0, 1], [bar_y, bar_y], "k", lw=1)
    ax.plot([0, 0], [bar_y - 2, bar_y], "k", lw=1)
    ax.plot([1, 1], [bar_y - 2, bar_y], "k", lw=1)
    exp = f"{stats_res.pvalue:.0e}"[1:]
    ax.text(0.5, bar_y + 0.5, f"p={exp}", ha="center", va="bottom")

    [x.patch.set_visible(False) for x in (fig, ax)]
    fig.savefig(f"{Path(__file__).stem}_{args.condition}.svg", transparent=True)
