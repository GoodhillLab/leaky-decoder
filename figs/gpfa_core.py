import numpy as np
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut, KFold


def circular_rmse(y_pred, y_test):
    """Calculates the circular Root Mean Squared Error in degrees."""
    # Convert predictions and true values to angles
    angle_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    angle_true = np.arctan2(y_test[:, 1], y_test[:, 0])

    # Calculate the difference and wrap it to the [-pi, pi] interval
    angle_diff = angle_pred - angle_true
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    # Calculate RMSE and convert to degrees
    rmse_rad = np.sqrt((angle_diff**2).mean())
    return np.rad2deg(rmse_rad)


class NeuralDataGenerator:
    """
    Generates and holds a master record of neural data for a simulation run.
    """

    def __init__(
        self,
        num_spiketrains=50,
        trial_duration_s=2,
        timestep_ms=1,
        max_rate_hz=150,
        baseline_hz=10.0,
        angular_frequency: float = 2 * np.pi * 1e-3,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.num_spiketrains = num_spiketrains
        self.timestep = timestep_ms * pq.ms
        self.trial_duration = trial_duration_s * pq.s
        self.num_steps = int(
            (self.trial_duration.rescale("ms") / self.timestep).magnitude
        )
        self.max_rate = max_rate_hz * pq.Hz
        self.baseline_hz = baseline_hz

        # Generate fixed properties for this instance
        _, self.oscillator_trajectory_2dim = self._integrated_oscillator(angular_frequency=angular_frequency)
        self.projection_matrix, self.oscillator_trajectory_Ndim = (
            self._random_projection()
        )
        self.instantaneous_rates = self._convert_to_rates()

    def _integrated_oscillator(self, x0=0, y0=1, angular_frequency=2 * np.pi * 1e-3):
        t = self.timestep.magnitude * np.arange(self.num_steps)
        x = x0 * np.cos(angular_frequency * t) + y0 * np.sin(angular_frequency * t)
        y = -x0 * np.sin(angular_frequency * t) + y0 * np.cos(angular_frequency * t)
        return t, np.array((x, y))

    def _random_projection(self, loc=0, scale=None):
        if scale is None:
            scale = 1 / np.sqrt(self.oscillator_trajectory_2dim.shape[0])
        proj_matrix = np.random.normal(
            loc, scale, (self.num_spiketrains, self.oscillator_trajectory_2dim.shape[0])
        )
        projected_data = proj_matrix @ self.oscillator_trajectory_2dim
        return proj_matrix, projected_data

    def _convert_to_rates(self):
        """
        Converts the projected oscillator trajectory to instantaneous firing rates.
        """
        # use a per-neuron scaling factor to ensure that the maximum firing rate is self.max_rate
        scale = np.max(
            np.abs(self.oscillator_trajectory_Ndim),
            axis=1,
            keepdims=True,
        )
        normed_traj = self.oscillator_trajectory_Ndim / scale
        return (
            np.power(self.max_rate.magnitude, normed_traj) + self.baseline_hz
        )  # baseline_hz baseline

    def generate_trials(self, num_trials):
        """Generates a specified number of spike train trials."""
        spiketrains = []
        for _ in range(num_trials):
            spiketrains_per_trial = []
            for inst_rate in self.instantaneous_rates:
                anasig_inst_rate = neo.AnalogSignal(
                    inst_rate, sampling_rate=1 / self.timestep, units=pq.Hz
                )
                spiketrains_per_trial.append(
                    inhomogeneous_poisson_process(anasig_inst_rate)
                )
            spiketrains.append(spiketrains_per_trial)
        return spiketrains


class GPFADecoder:
    """
    Encapsulates the GPFA fitting and decoding logic.
    """

    def __init__(
        self,
        bin_size_ms: int = 20,
        latent_dimensionality: int = 2,
        n_neighbors: int = 5,
    ):
        self.bin_size = bin_size_ms * pq.ms
        self.latent_dimensionality = latent_dimensionality
        self.gpfa = GPFA(bin_size=self.bin_size, x_dim=self.latent_dimensionality)
        self.regressor = MultiOutputRegressor(
            KNeighborsRegressor(n_neighbors=n_neighbors)
        )

    def run_decoding_comparison(
        self, spiketrains_cv, spiketrains_test, true_trajectory
    ):
        """
        Performs decoding for both leaky and non-leaky scenarios.
        Note: True non-leaky would re-run GPFA inside each fold, which is very slow.
        Here, we compare the leaky CV error vs. the generalization error of that same leaky model.
        """

        # --- Leaky Analysis ---
        # 1. Fit GPFA on the entire CV dataset (this is the leakage step)
        trajectories_all_leaky = self.gpfa.fit_transform(spiketrains_cv)
        # 2. Transform the held-out test data using the leaky model
        trajectories_test = self.gpfa.transform(spiketrains_test)

        # here we assume the array shapes are consistent across trials
        trajectories_all_leaky = np.stack(trajectories_all_leaky)
        X_test_gen = np.stack(trajectories_test)
        y_test_gen = np.tile(true_trajectory, len(spiketrains_test)).T

        # 3. Perform Leave-One-Out CV on the leaky latent variables
        cv = KFold(n_splits=5, shuffle=False)
        fold_rmses_leaky = []
        fold_rmses_gen_err = []
        no_leakage_score = []
        fold_rmses_gen_err2 = []

        for train_idx, test_idx in cv.split(trajectories_all_leaky):
            # Prepare train/test data for this fold
            X_train = np.hstack(trajectories_all_leaky[train_idx]).T
            y_train = np.tile(true_trajectory, len(train_idx)).T
            X_test = np.hstack(trajectories_all_leaky[test_idx]).T
            y_test = np.tile(true_trajectory, len(test_idx)).T

            # Fit regressor on the training part of this fold
            self.regressor.fit(X_train, y_train)

            # --- Calculate Leaky CV Error for this fold ---
            y_pred = self.regressor.predict(X_test)
            rmse_leaky = circular_rmse(y_pred, y_test)
            fold_rmses_leaky.append(rmse_leaky)

            # --- Calculate Generalization Error of this fold's model ---
            y_pred_gen = self.regressor.predict(np.hstack(X_test_gen).T)
            rmse_gen_err = circular_rmse(y_pred_gen, y_test_gen)
            fold_rmses_gen_err.append(rmse_gen_err)

            # no leakage scenario
            X_train = [xx for i, xx in enumerate(spiketrains_cv) if i in train_idx]
            X_test = [xx for i, xx in enumerate(spiketrains_cv) if i in test_idx]
            train_reduced = self.gpfa.fit_transform(X_train)
            self.regressor.fit(np.hstack(train_reduced).T, y_train)
            test_reduced = self.gpfa.transform(X_test)
            y_pred = self.regressor.predict(np.hstack(test_reduced).T)
            rmse_no_leak = circular_rmse(y_pred, y_test)
            no_leakage_score.append(rmse_no_leak)

            # 2. Transform the held-out test data using the non-leak model
            test_reduced = self.gpfa.transform(spiketrains_test)
            y_pred_gen = self.regressor.predict(np.hstack(test_reduced).T)
            rmse_gen_err = circular_rmse(y_pred_gen, y_test_gen)
            fold_rmses_gen_err2.append(rmse_gen_err)

        # Average the results across all folds
        return (
            np.mean(fold_rmses_leaky),
            np.mean(fold_rmses_gen_err),
            np.mean(no_leakage_score),
            np.mean(fold_rmses_gen_err2),
        )
