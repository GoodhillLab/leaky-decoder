import unittest
import numpy as np
from sklearn.decomposition import PCA
from utils import TrialAveragePCA


class TestTrialAveragePCA(unittest.TestCase):
    """
    Test the TrialAveragePCA class.
    """

    def setUp(self):
        # Create a sample dataset
        np.random.seed(0)
        n_trials = 10  # number of trials for each class
        n_neurons = 10  # number of neurons
        n_stim = 4  # number of unique stimuli
        self.X = np.random.randn(n_trials * n_stim, n_neurons)
        self.y = np.repeat(np.arange(n_stim), n_trials)

    @staticmethod
    def trial_average(X, y):
        """
        Average the data over trials for each class.
        """
        X_mean = [X[y == y_unique].mean(axis=0) for y_unique in np.unique(y)]

        return np.array(X_mean)

    def test_fit_transform(self):
        """
        Test the fit_transform method.
        """
        # Trial-average the data
        X_mean = self.trial_average(self.X, self.y)

        # Apply standard PCA on trial-averaged data
        pca = PCA(n_components=2)
        pca.fit(X_mean)
        pca_result = pca.transform(self.X)

        # Apply TrialAveragePCA on the original data
        trial_avg_pca = TrialAveragePCA(n_components=2)
        trial_avg_pca_result = trial_avg_pca.fit_transform(self.X, self.y)

        # Compare the results
        np.testing.assert_almost_equal(pca_result, trial_avg_pca_result, decimal=3)

    def test_attributes(self):
        """
        Test the attributes of the TrialAveragePCA class.
        """
        # Trial-average the data
        X_mean = self.trial_average(self.X, self.y)

        # Apply standard PCA on trial-averaged data
        pca = PCA(n_components=2)
        pca.fit(X_mean)

        # Apply TrialAveragePCA on the original data
        trial_avg_pca = TrialAveragePCA(n_components=2)
        trial_avg_pca.fit(self.X, self.y)

        # Compare the attributes
        np.testing.assert_almost_equal(pca.mean_, trial_avg_pca.mean_, decimal=3)
        np.testing.assert_almost_equal(
            pca.components_, trial_avg_pca.components_, decimal=3
        )
        np.testing.assert_almost_equal(
            pca.explained_variance_, trial_avg_pca.explained_variance_, decimal=3
        )
        np.testing.assert_almost_equal(
            pca.explained_variance_ratio_,
            trial_avg_pca.explained_variance_ratio_,
            decimal=3,
        )


if __name__ == "__main__":
    unittest.main()
