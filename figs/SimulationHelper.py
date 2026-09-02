from typing import Dict, Union

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
import pandas as pd
from tqdm.auto import trange
from joblib import Parallel, delayed


# custom covariance estimator
class EyeCovarianceEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.covariance_ = np.eye(X.shape[1])
        return self


class Simulation:
    """
    Simulate a decoding experiment.
    """

    def __init__(
        self,
        y: np.ndarray,
        preprocessor,
        max_n_neurons: int = 500,
        clf=LDA(solver="lsqr", covariance_estimator=EyeCovarianceEstimator()),
        cv=LeaveOneOut(),
    ):
        """
        Parameters
        ----------
        y : np.ndarray
            The visual stimuli.
        max_n_neurons : int
            The maximum number of neurons to simulate. Defaults to 500.
        """
        self.y = y
        self.max_n_neurons = max_n_neurons
        self.cv = cv
        self.preprocessor = preprocessor
        self.clf = make_pipeline(self.preprocessor, clf)
        self.methods = ["pre", "proper_pre"]

    def generate_data(self) -> np.ndarray:
        """Generate random data for the maximum number of neurons."""
        X = np.random.rand(len(self.y), self.max_n_neurons) < 0.1
        return X

    def run_simulation(self, X: np.ndarray, method: str = "full") -> float:
        """Run a single decoding analysis with a given data matrix X."""

        assert method in self.methods, f"Unknown method: {method}"

        if method == "full":
            accuracy = cross_val_score(self.clf[-1], X, self.y, cv=self.cv)
        elif method == "pre":
            X_new = self.preprocessor.fit_transform(X, self.y)
            accuracy = cross_val_score(self.clf[-1], X_new, self.y, cv=self.cv)
        elif method == "proper_pre":
            accuracy = cross_val_score(self.clf, X, self.y)
        else:
            raise ValueError(f"Unknown method: {method}")

        return accuracy.mean()

    def _run_one_simulation(
        self, sim_idx: int, n_neuron_list: list, base_seed: int = 0
    ) -> list:
        """
        Helper function: runs the logic for a single simulation index.
        This is designed to be called in parallel by run_simulations.
        """
        # Seed per simulation index so the result depends only on sim_idx.
        # Without this the joblib workers each draw from an independently
        # OS-seeded global stream, which makes the run irreproducible and
        # dependent on n_jobs.
        np.random.seed(base_seed + sim_idx)
        # Generate data for the maximum number of neurons once per simulation
        X_full = self.generate_data()

        results_for_this_sim = []
        # Inner loop for subsampling neurons
        for n_neurons in n_neuron_list:
            # Subsample the first n_neurons columns
            X_sub = X_full[:, :n_neurons]

            for method in self.methods:
                accuracy = self.run_simulation(X_sub, method=method)
                results_for_this_sim.append((accuracy, method, n_neurons, sim_idx))

        return results_for_this_sim

    def run_simulations(
        self,
        n_simulations: int,
        n_neuron_list: list,
        n_jobs: int = 1,
        base_seed: int = 0,
    ) -> pd.DataFrame:
        """
        Run n_simulations in parallel. In each simulation, generate one full dataset and
        subsample neurons to test decoding accuracy across different neuron counts.

        Results are reproducible and independent of ``n_jobs``: simulation ``i``
        is seeded with ``base_seed + i``.
        """
        # Use joblib to parallelize the outer loop over simulations.
        # Each call to _run_one_simulation is a single job.
        results_list_of_lists = Parallel(n_jobs=n_jobs)(
            delayed(self._run_one_simulation)(i, n_neuron_list, base_seed)
            for i in trange(n_simulations, desc="Simulations")
        )

        # The parallel call returns a list of lists; this flattens it.
        accuracies = [item for sublist in results_list_of_lists for item in sublist]

        df = pd.DataFrame(
            accuracies, columns=["accuracy", "method", "N_neurons", "simulation_id"]
        )

        name = self.preprocessor.__class__.__name__
        if "Custom" in name:
            name = str(self.preprocessor)
        df["preprocessor"] = name
        return df


class SimulationRandomMean(Simulation):
    @staticmethod
    def get_class_means(n_neurons: int) -> Dict[int, np.ndarray]:
        """Get the class means."""

        # generate a random vector with unit l2 norm
        random_vector = np.random.randn(n_neurons)
        random_vector /= np.linalg.norm(random_vector)

        assert np.isclose(np.linalg.norm(random_vector), 1), "Not unit l2 norm"

        # create different class means
        class_means: Dict[int, np.ndarray] = {}
        # make the base of the class means as [10, 10, 10, ...]
        class_means[0] = np.ones(n_neurons) * 10
        class_means[1] = class_means[0] + random_vector

        return class_means

    def generate_data(self) -> np.ndarray:
        """Generate random data."""
        assert np.array_equal(np.unique(self.y), (0, 1)), "Only support two stimuli"
        # create different class means
        class_means = self.get_class_means(self.max_n_neurons)

        # Generate random data
        X = np.zeros((len(self.y), self.max_n_neurons))
        # we sample from the normal distribution of each class
        for y, mu in class_means.items():
            idx = self.y == y
            samples = np.random.normal(loc=mu, size=(idx.sum(), self.max_n_neurons))
            assert samples.min() > 0, "Negative firing rate"
            X[idx] = samples

        return X


class SimulationStructuredResponse(Simulation):
    """
    Simulate a decoding experiment with X with N(mu, cov) where mu is different for different stimuli.
    """

    def __init__(self, y, preprocessor, cov: np.ndarray = None, **kwargs):
        super().__init__(y, preprocessor, **kwargs)
        self.cov: np.ndarray = cov

    @classmethod
    def get_class_means(self) -> Dict[int, np.ndarray]:
        """
        Get the class means. To be implemented by the subclass.
        """
        return {}

    def generate_data(self) -> np.ndarray:
        """Generate random data."""
        # create different class means
        class_means = self.get_class_means()

        # Generate random data
        X = np.zeros((len(self.y), self.max_n_neurons))
        isDiag = self.isDiag(self.cov)
        if isDiag:
            scale = np.sqrt(self.cov.diagonal())

        # we sample from the normal distribution of each class
        for y, mu in class_means.items():
            idx = self.y == y
            if isDiag:
                assert scale is not None
                samples = np.random.normal(
                    loc=mu, scale=scale, size=(idx.sum(), self.max_n_neurons)
                )
            else:
                samples = np.random.multivariate_normal(mu, self.cov, size=idx.sum())

            assert samples.min() > 0, "Negative firing rate"
            X[idx] = samples

        return X

    @staticmethod
    def isDiag(M):
        """
        Check if a matrix is diagonal.
        https://stackoverflow.com/a/43885215
        """
        i, j = np.nonzero(M)
        return np.all(i == j)


class SimulationGaussianTuningCurve(SimulationStructuredResponse):
    def __init__(self, y, preprocessor, **kwargs):
        super().__init__(y, preprocessor, **kwargs)

    @staticmethod
    def gaussian(x, mu: Union[float, np.ndarray], sigma: float = 1):
        """Gaussian function."""
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def get_class_means(self) -> Dict[int, np.ndarray]:
        """Get the class means from Gaussian tuning curve."""
        y = set(np.unique(self.y))
        assert len(y) == 2, "Only support two stimuli"

        class_means: Dict[int, np.ndarray] = {}
        mu_list = np.linspace(-np.pi, np.pi, self.max_n_neurons)

        for y_ in y:
            mean_tmp = y_
            if y_ == 0:
                mean_tmp = -1
            class_means[y_] = self.gaussian(mean_tmp, mu_list)

        normalization = np.linalg.norm(class_means[0] - class_means[1])
        class_means[0] /= normalization
        class_means[1] /= normalization

        class_means[0] += 10
        class_means[1] += 10

        assert np.isclose(
            np.linalg.norm(class_means[0] - class_means[1]), 1
        ), "Not unit l2 norm difference"

        return class_means
