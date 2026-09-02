import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import _fit_context
from sklearn.base import BaseEstimator, TransformerMixin


class TrialAveragePCA(PCA):
    """
    A modified PCA class which applies PCA to the trial-averaged data.
    """

    def __init__(self, n_components=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # for each y, average over the trials
        X_mean = [X[y == y_unique].mean(axis=0) for y_unique in np.unique(y)]
        X_mean = np.array(X_mean)

        # Use the parent class fit method directly
        super().fit(X_mean)

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class PLS(BaseEstimator, TransformerMixin):
    """
    PLS with 1D components. Assume zero-mean X and y.
    """

    def __init__(self):
        self.n_components: int = 1
        self.w: np.ndarray = None
        self.components_: np.ndarray = None

    def fit(self, X, y):
        self.w = X.T @ y
        self.w = self.w / np.linalg.norm(self.w)
        self.w = self.w[..., None]
        self.components_ = self.w.T
        return self

    def transform(self, X, y=None):
        return X @ self.components_.T

    def __repr__(self):
        return "PLS"

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def plot_accuracy(df, score_shuffle, palette="Set1"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Plot the accuracy of the model with and without leakage.
    Confidence intervals from permutation tests plotted as shaded rectangle.
    """

    fig, ax = plt.subplots(figsize=(2.5, 2), layout="tight")
    sns.barplot(
        data=df,
        x="method",
        y="accuracy",
        hue="method",
        order=["leakage", "no leakage"],
        ax=ax,
        width=0.4,
        palette=palette,
        errorbar=("ci", 68),
    )

    # plot the shuffle control as a shaded rectangle
    ax.fill_between(
        [-0.5, 1.5],
        np.percentile(score_shuffle, 2.5),
        np.percentile(score_shuffle, 97.5),
        color="gray",
        alpha=0.5,
    )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    sns.despine()

    return fig, ax
