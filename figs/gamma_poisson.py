# %%
import argparse

from loguru import logger as logging
import numpy as np
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


# %% Database setup
DB_NAME = "data/neural_decoding_simulations.db"
RESULT_COLUMN_ORDER = [
    'timestamp', 'tuning_type', 'num_samples', 'num_neurons', 'k_best_neurons',
    'xi_scale', 'num_high_rate_neurons', 'rate_increase_factor',
    'selection_score', 'selection_activity', 'baseline_firing_rate',
    'spontaneous_sample_multiplier', 'shape_multiplier', 'cv_type',
    'run_id_within_params', 'leaky_mse', 'non_leaky_mse', 'leaky_gen_err_mse',
]


EXTRA_RESULT_COLUMNS = {
    "selection_score": "TEXT",
    "selection_activity": "TEXT",
    "baseline_firing_rate": "REAL",
    "spontaneous_sample_multiplier": "INTEGER",
    "shape_multiplier": "REAL",
}

# Default used by the original semi-parametric Gamma-Poisson simulation:
# shape_param = exp(x @ beta).
DEFAULT_GAMMA_POISSON_SHAPE_MULTIPLIER = 1.0


def _ensure_result_columns(conn: sqlite3.Connection):
    """Adds result columns when reusing an older simulation database."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(simulation_results)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    for column_name, column_type in EXTRA_RESULT_COLUMNS.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE simulation_results ADD COLUMN {column_name} {column_type}"
            )


def init_db(db_name=DB_NAME, overwrite: bool = False):
    """Initializes the SQLite database and creates the results table if it doesn't exist."""
    Path(db_name).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if overwrite:
        cursor.execute("DROP TABLE IF EXISTS simulation_results")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS simulation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            tuning_type TEXT,
            num_samples INTEGER,
            num_neurons INTEGER,
            k_best_neurons INTEGER,
            xi_scale REAL,
            num_high_rate_neurons INTEGER,
            rate_increase_factor REAL,
            selection_score TEXT,
            selection_activity TEXT,
            baseline_firing_rate REAL,
            spontaneous_sample_multiplier INTEGER,
            shape_multiplier REAL,
            cv_type TEXT,
            run_id_within_params INTEGER,
            leaky_mse REAL,
            non_leaky_mse REAL,
            leaky_gen_err_mse REAL
        )
    """
    )
    _ensure_result_columns(conn)
    conn.commit()
    conn.close()
    logging.info(f"Database '{db_name}' initialized.")


# %% Feature-selection helpers
SELECTION_SCORES = {"variance", "mean_firing_rate"}
SELECTION_ACTIVITIES = {"all", "evoked", "spontaneous"}


def score_neurons(X: np.ndarray, selection_score: str) -> np.ndarray:
    """Scores neurons for unsupervised feature selection."""
    if selection_score == "variance":
        return np.var(X, axis=0, ddof=1)
    if selection_score == "mean_firing_rate":
        return np.mean(X, axis=0)
    raise ValueError(f"Unknown selection score: {selection_score}")


def select_neuron_indices(
    X_selection: np.ndarray,
    k_best_neurons_setting: Any,
    selection_score: str,
) -> np.ndarray:
    """Returns the indices of the top-scoring neurons."""
    num_features = X_selection.shape[1]
    k_to_select = (
        int(k_best_neurons_setting)
        if isinstance(k_best_neurons_setting, int)
        else num_features
    )
    if k_to_select < 1 or k_to_select > num_features:
        raise ValueError(
            f"k_best_neurons must be between 1 and {num_features}; got {k_to_select}"
        )

    scores = score_neurons(X_selection, selection_score)
    return np.argsort(scores, kind="mergesort")[-k_to_select:]


def make_selection_activity(
    evoked_activity: np.ndarray,
    spontaneous_activity: Optional[np.ndarray],
    selection_activity: str,
) -> np.ndarray:
    """Builds the activity matrix used to score neurons for selection."""
    if selection_activity not in SELECTION_ACTIVITIES:
        raise ValueError(f"Unknown activity source for selection: {selection_activity}")
    if selection_activity == "evoked":
        return evoked_activity
    if spontaneous_activity is None:
        raise ValueError(
            f"spontaneous_activity is required for {selection_activity} selection"
        )
    if spontaneous_activity.shape[1] != evoked_activity.shape[1]:
        raise ValueError("spontaneous_activity must have the same number of neurons")
    if selection_activity == "spontaneous":
        return spontaneous_activity
    return np.concatenate((spontaneous_activity, evoked_activity), axis=0)


# %% Data Generation Class
class NeuralDataGenerator:
    """
    Generates and holds master records of neural data for both CV and a held-out test set.
    """
    def __init__(self, num_neurons: int, max_n_samples: int, held_out_samples: int, tuning_type: str, generation_params: dict):
        self.num_neurons = num_neurons
        self.tuning_type = tuning_type
        self.params = generation_params
        self.tuning_params = generation_params.get('tuning_params', {})
        self.spontaneous_sample_multiplier = int(
            self.tuning_params.get('spontaneous_sample_multiplier', 1)
        )
        if self.spontaneous_sample_multiplier < 1:
            raise ValueError("spontaneous_sample_multiplier must be at least 1")
        
        self.beta = self._generate_projection_matrix(num_neurons)
        X_stimulus, stimulus_angles = self._generate_stimuli(2 * max_n_samples + held_out_samples)
        neural_responses_evoked = self._generate_evoked_activity(X_stimulus, stimulus_angles)
        self.neural_responses_spontaneous_full = self._generate_spontaneous_activity(
            max_n_samples * self.spontaneous_sample_multiplier
        )

        # Split off the held-out test set (last held_out_samples) without shuffling
        (
            evoked_remaining,
            self.neural_responses_test,
            stimulus_remaining,
            self.stimulus_angles_rad_test,
        ) = train_test_split(
            neural_responses_evoked,
            stimulus_angles,
            test_size=held_out_samples,
            shuffle=False,
        )
        # Split the remaining data (2 * max_n_samples samples) equally into CV set and dummy set
        (
            self.neural_responses_full,
            self._neural_responses_dummy,
            self.stimulus_angles_rad_full,
            _,
        ) = train_test_split(
            evoked_remaining,
            stimulus_remaining,
            test_size=max_n_samples,
            shuffle=False,
        )

    @staticmethod
    def _generate_stimuli(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        theta = np.random.uniform(0, np.pi, n_samples)
        X_stimulus = np.column_stack((np.cos(theta), np.sin(theta)))
        return X_stimulus, theta

    @staticmethod
    def _generate_projection_matrix(num_neurons: int) -> np.ndarray:
        return np.random.randn(num_neurons, 2)

    def _generate_evoked_activity(self, X_stimulus, stimulus_angles_rad) -> np.ndarray:
        """Generates stimulus-evoked spike counts."""
        num_neurons = self.num_neurons
        lambda_rates: np.ndarray
        if self.tuning_type == 'gamma_poisson_mixed':
            tuning_params = self.params.get('tuning_params', {})
            shape_multiplier = tuning_params.get(
                'shape_multiplier',
                DEFAULT_GAMMA_POISSON_SHAPE_MULTIPLIER,
            )
            if shape_multiplier <= 0:
                raise ValueError("shape_multiplier must be positive")
            psi = X_stimulus @ self.beta.T
            shape_param = shape_multiplier * np.exp(psi) + 1e-9
            lambda_rates = np.random.gamma(shape=shape_param, scale=self.params['xi_scale'])
        elif self.tuning_type == 'gaussian_tuning_curve':
            tuning_params = self.params.get('tuning_params', {})
            preferred_angles = np.random.uniform(0, np.pi, num_neurons)
            tuning_std_rad = np.deg2rad(tuning_params.get('tuning_curve_std_deg', 30.0))
            max_fr = tuning_params.get('max_firing_rate', 20.0)
            baseline_fr = tuning_params.get('baseline_firing_rate', 2.0)
            angle_diff = stimulus_angles_rad[:, np.newaxis] - preferred_angles[np.newaxis, :]
            exponent = -0.5 * (angle_diff / tuning_std_rad)**2
            tuned_part = max_fr * np.exp(exponent)
            shape_param = baseline_fr + tuned_part
            lambda_rates = np.random.gamma(shape=shape_param, scale=self.params['xi_scale'])
        else:
            raise ValueError(f"Unknown tuning_type: {self.tuning_type}")
        
        high_rate_params = self.params.get('tuning_params', {})
        num_high_rate = high_rate_params.get('num_high_rate_neurons', 0)
        increase_factor = high_rate_params.get('rate_increase_factor', 1.0)
        if num_high_rate > 0 and increase_factor > 1.0:
            actual_high_rate = min(num_high_rate, num_neurons)
            lambda_rates[:, :actual_high_rate] *= increase_factor

        return np.random.poisson(lam=lambda_rates)

    def _generate_spontaneous_activity(self, n_samples: int) -> np.ndarray:
        """Generates pre-stimulus activity independent of the stimulus."""
        tuning_params = self.params.get('tuning_params', {})
        baseline_fr = tuning_params.get('baseline_firing_rate', 0.0)
        if baseline_fr <= 0:
            return np.zeros((n_samples, self.num_neurons), dtype=int)

        return np.random.poisson(
            lam=baseline_fr,
            size=(n_samples, self.num_neurons),
        )
    
    def get_cv_subsample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        return self.neural_responses_full[:n_samples, :], self.stimulus_angles_rad_full[:n_samples]

    def get_heldout_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self.neural_responses_test, self.stimulus_angles_rad_test

    def get_activity_subsample(self, n_samples: int, activity: str = "all") -> np.ndarray:
        return make_selection_activity(
            self.neural_responses_full[:n_samples, :],
            self.neural_responses_spontaneous_full[:n_samples, :],
            activity,
        )

    def get_dummy_activity(self, activity: str = "all") -> np.ndarray:
        return make_selection_activity(
            self._neural_responses_dummy,
            self.neural_responses_spontaneous_full,
            activity,
        )

    def get_spontaneous_subsample(self, n_samples: int) -> np.ndarray:
        n_spontaneous_samples = n_samples * self.spontaneous_sample_multiplier
        return self.neural_responses_spontaneous_full[:n_spontaneous_samples, :]

    @property
    def neural_responses_dummy(self) -> np.ndarray:
        """Returns dummy neural responses for controling the number of samples used in preprocessing."""
        return self._neural_responses_dummy


# %% Decoding Function
def _calculate_mse(y_pred_rad, y_true_rad) -> float:
    """Helper to calculate MSE in degrees."""
    diff_deg = np.rad2deg(y_pred_rad - y_true_rad)
    return (diff_deg**2).mean()

def decode_angles_cv_and_gen_err(
    cv_neural_responses: np.ndarray, cv_stimulus_angles: np.ndarray,
    X_dummy: np.ndarray,  # Dummy responses for preprocessing
    test_neural_responses: np.ndarray, test_stimulus_angles: np.ndarray,
    k_best_neurons_setting: Any, cv_splitter: object,
    selection_score: str = "variance",
    selection_activity: str = "evoked",
    cv_spontaneous_responses: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """
    Computes CV MSEs and the leaky generalization error estimate.
    Returns (mse_leaky_cv, mse_non_leaky_cv, mse_leaky_gen_err).
    """
    lm = LinearRegression()
    if selection_score not in SELECTION_SCORES:
        raise ValueError(f"Unknown selection score: {selection_score}")
    if selection_activity not in SELECTION_ACTIVITIES:
        raise ValueError(f"Unknown selection activity: {selection_activity}")
    if cv_spontaneous_responses is not None and cv_spontaneous_responses.shape[1] != cv_neural_responses.shape[1]:
        raise ValueError("cv_spontaneous_responses must have the same number of neurons")

    # --- 1. Non-Leaky Path (CV only) ---
    mse_non_leaky_cv_list = []
    for (train_idx, test_idx) in cv_splitter.split(cv_neural_responses):
        X_train, X_test = cv_neural_responses[train_idx], cv_neural_responses[test_idx]
        y_train, y_test = cv_stimulus_angles[train_idx], cv_stimulus_angles[test_idx]
        X_train2 = np.concatenate((X_train, X_dummy[test_idx]), axis=0)
        X_selection_train = make_selection_activity(
            X_train2,
            cv_spontaneous_responses,
            selection_activity,
        )
        selected_neurons = select_neuron_indices(
            X_selection_train,
            k_best_neurons_setting,
            selection_score,
        )
        lm.fit(X_train[:, selected_neurons], y_train)
        y_test_predicted = lm.predict(X_test[:, selected_neurons])
        mse_non_leaky_cv = _calculate_mse(y_test_predicted, y_test)
        mse_non_leaky_cv_list.append(mse_non_leaky_cv)
    
    # --- 2. Leaky Path (CV and Generalization) ---
    X_selection_leaky = make_selection_activity(
        cv_neural_responses,
        cv_spontaneous_responses,
        selection_activity,
    )
    selected_neurons = select_neuron_indices(
        X_selection_leaky,
        k_best_neurons_setting,
        selection_score,
    )
    X_cv_leaky = cv_neural_responses[:, selected_neurons]
    # Leaky CV MSE and Generalization Error
    mse_gen_err_list = []
    mse_leaky_cv_list = []
    X_gen_test = test_neural_responses[:, selected_neurons]
    for (train_idx, test_idx) in cv_splitter.split(X_cv_leaky):
        X_train, X_test = X_cv_leaky[train_idx], X_cv_leaky[test_idx]
        y_train, y_test = cv_stimulus_angles[train_idx], cv_stimulus_angles[test_idx]
        lm.fit(X_train, y_train)
        y_gen_predicted = lm.predict(X_gen_test)
        y_test_predicted = lm.predict(X_test)
        mse_gen_err = _calculate_mse(y_gen_predicted, test_stimulus_angles)
        mse_leaky_cv = _calculate_mse(y_test_predicted, y_test)
        mse_gen_err_list.append(mse_gen_err)
        mse_leaky_cv_list.append(mse_leaky_cv)

    return np.mean(mse_leaky_cv_list), np.mean(mse_non_leaky_cv_list), np.mean(mse_gen_err_list)

# %% Joblib Worker Function
def run_single_simulation(run_id: int, config: dict, max_n_samples: int, num_samples_values: np.ndarray, cv_methods: list, held_out_samples: int) -> list:
    """Worker function for a single simulation run. Generates data once and subsamples."""
    np.random.seed(run_id) 

    data_generator = NeuralDataGenerator(
        num_neurons=config['num_neurons'],
        max_n_samples=max_n_samples,
        held_out_samples=held_out_samples,
        tuning_type=config['tuning_type'],
        generation_params=config
    )
    X_test, y_test = data_generator.get_heldout_set()
    X_dummy = data_generator.neural_responses_dummy
    selection_score = config.get('selection_score', 'variance')
    selection_activity = config.get('selection_activity', 'evoked')
    
    results_for_this_run = []
    
    for n_samples in num_samples_values:
        responses_subset, angles_subset = data_generator.get_cv_subsample(n_samples)
        spontaneous_subset = data_generator.get_spontaneous_subsample(n_samples)

        for cv_method_name, _ in cv_methods:
            cv_splitter: object
            if cv_method_name[-5:] == "-fold":
                num_folds = int(cv_method_name[:-5])
                if n_samples < num_folds: continue
                cv_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=run_id)
            elif cv_method_name == "loocv":
                if n_samples < 2: continue
                cv_splitter = LeaveOneOut()
            else:
                continue
            
            # MSE of leaky and non-leaky decoders, and generalization error
            mse_l_cv, mse_nl_cv, mse_l_gen = decode_angles_cv_and_gen_err(
                responses_subset,
                angles_subset,
                X_dummy,
                X_test,
                y_test,
                config['k_best_neurons'],
                cv_splitter,
                selection_score=selection_score,
                selection_activity=selection_activity,
                cv_spontaneous_responses=spontaneous_subset,
            )

            tuning_params = config.get('tuning_params', {})
            result_dict = {
                'timestamp': int(time.time()), 'tuning_type': config['tuning_type'],
                'num_samples': n_samples, 'num_neurons': config['num_neurons'],
                'k_best_neurons': config['k_best_neurons'], 'xi_scale': config['xi_scale'],
                'num_high_rate_neurons': tuning_params.get('num_high_rate_neurons', 0),
                'rate_increase_factor': tuning_params.get('rate_increase_factor', 1.0),
                'selection_score': selection_score,
                'selection_activity': selection_activity,
                'baseline_firing_rate': tuning_params.get('baseline_firing_rate', 0.0),
                'spontaneous_sample_multiplier': tuning_params.get('spontaneous_sample_multiplier', 1),
                'shape_multiplier': tuning_params.get(
                    'shape_multiplier',
                    DEFAULT_GAMMA_POISSON_SHAPE_MULTIPLIER
                    if config['tuning_type'] == 'gamma_poisson_mixed'
                    else 1.0,
                ),
                'cv_type': cv_method_name, 'run_id_within_params': run_id,
                'leaky_mse': mse_l_cv, 'non_leaky_mse': mse_nl_cv,
                'leaky_gen_err_mse': mse_l_gen,
            }
            results_for_this_run.append(result_dict)
                
    return results_for_this_run


def parse_int_values(values: list[str]) -> list[int]:
    """Parses whitespace- or comma-separated integer arguments."""
    parsed_values = []
    for value in values:
        parsed_values.extend(int(item) for item in value.split(",") if item)
    return parsed_values


def parse_str_values(values: list[str]) -> list[str]:
    """Parses whitespace- or comma-separated string arguments."""
    parsed_values = []
    for value in values:
        parsed_values.extend(item for item in value.split(",") if item)
    return parsed_values


def build_generation_config(
    tuning_type: str,
    num_neurons: int,
    k_best_neurons: int,
    num_simulation_runs: int,
    shape_multiplier: float,
    spontaneous_sample_multiplier: int,
    mixed_num_high_rate_neurons: int = 5,
    mixed_rate_increase_factor: float = 5.0,
) -> dict:
    base_params = {
        'num_neurons': num_neurons,
        'k_best_neurons': k_best_neurons,
        'num_simulation_runs': num_simulation_runs,
    }
    if tuning_type == 'gamma_poisson_mixed':
        return {
            **base_params,
            'tuning_type': 'gamma_poisson_mixed',
            'xi_scale': 1.0,
            'tuning_params': {
                'num_high_rate_neurons': mixed_num_high_rate_neurons,
                'rate_increase_factor': mixed_rate_increase_factor,
                'shape_multiplier': shape_multiplier,
                'baseline_firing_rate': 2.0,
                'spontaneous_sample_multiplier': spontaneous_sample_multiplier,
            },
        }
    if tuning_type == 'gaussian_tuning_curve':
        return {
            **base_params,
            'tuning_type': 'gaussian_tuning_curve',
            'xi_scale': 1.0,
            'tuning_params': {
                'num_high_rate_neurons': 0,
                'rate_increase_factor': 0.0,
                'tuning_curve_std_deg': 20.0,
                'max_firing_rate': 50.0,
                'baseline_firing_rate': 1.0,
                'spontaneous_sample_multiplier': spontaneous_sample_multiplier,
            },
        }
    raise ValueError(f"Unknown tuning_type: {tuning_type}")


def insert_results(db_connection: sqlite3.Connection, all_results: list[dict]):
    data_to_insert = [[res[col] for col in RESULT_COLUMN_ORDER] for res in all_results]
    placeholders = ", ".join(["?"] * len(RESULT_COLUMN_ORDER))
    cursor = db_connection.cursor()
    cursor.executemany(
        f"INSERT INTO simulation_results ({', '.join(RESULT_COLUMN_ORDER)}) VALUES ({placeholders})",
        data_to_insert,
    )
    db_connection.commit()


# %% Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gamma-Poisson unsupervised feature-selection leakage simulations."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate the simulation_results table before running.",
    )
    parser.add_argument("--db", default=DB_NAME, help="SQLite database output path.")
    parser.add_argument("--num-runs", type=int, default=10_000)
    parser.add_argument("--num-neurons", type=int, default=100)
    parser.add_argument("--k-best-neurons", type=int, default=10)
    parser.add_argument("--held-out-samples", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--num-samples",
        nargs="+",
        default=["20", "25", "30", "35", "40", "45", "50", "55", "60"],
        help="Sample counts, as space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--cv",
        nargs="+",
        default=["5-fold"],
        help="Cross-validation methods to run.",
    )
    parser.add_argument(
        "--tuning-types",
        nargs="+",
        default=["gaussian_tuning_curve", "gamma_poisson_mixed"],
    )
    parser.add_argument(
        "--selection-scores",
        nargs="+",
        default=["mean_firing_rate", "variance"],
    )
    parser.add_argument(
        "--selection-activities",
        nargs="+",
        default=["evoked"],
    )
    parser.add_argument(
        "--shape-multiplier",
        type=float,
        default=DEFAULT_GAMMA_POISSON_SHAPE_MULTIPLIER,
        help="Gamma shape multiplier used only for gamma_poisson_mixed.",
    )
    parser.add_argument(
        "--mixed-num-high-rate-neurons",
        type=int,
        default=5,
        help=(
            "Number of gamma_poisson_mixed neurons with an extra multiplicative "
            "rate increase."
        ),
    )
    parser.add_argument(
        "--mixed-rate-increase-factor",
        type=float,
        default=5.0,
        help="Multiplicative rate increase for the high-rate gamma_poisson_mixed neurons.",
    )
    parser.add_argument(
        "--spontaneous-sample-multiplier",
        type=int,
        default=5,
        help="Number of spontaneous samples per evoked sample.",
    )
    args = parser.parse_args()

    num_samples_values = np.array(parse_int_values(args.num_samples), dtype=int)
    cv_methods = [(name, None) for name in parse_str_values(args.cv)]
    tuning_types = parse_str_values(args.tuning_types)
    selection_scores = parse_str_values(args.selection_scores)
    selection_activities = parse_str_values(args.selection_activities)
    unknown_cv_methods = {name for name, _ in cv_methods} - {"5-fold", "loocv"}
    unknown_tuning_types = set(tuning_types) - {
        "gaussian_tuning_curve",
        "gamma_poisson_mixed",
    }
    unknown_scores = set(selection_scores) - SELECTION_SCORES
    unknown_activities = set(selection_activities) - SELECTION_ACTIVITIES
    if unknown_cv_methods:
        raise ValueError(f"Unknown CV methods: {sorted(unknown_cv_methods)}")
    if unknown_tuning_types:
        raise ValueError(f"Unknown tuning types: {sorted(unknown_tuning_types)}")
    if unknown_scores:
        raise ValueError(f"Unknown selection scores: {sorted(unknown_scores)}")
    if unknown_activities:
        raise ValueError(f"Unknown selection activities: {sorted(unknown_activities)}")

    init_db(args.db, overwrite=args.overwrite)

    data_configs = [
        build_generation_config(
            tuning_type=tuning_type,
            num_neurons=args.num_neurons,
            k_best_neurons=args.k_best_neurons,
            num_simulation_runs=args.num_runs,
            shape_multiplier=args.shape_multiplier,
            spontaneous_sample_multiplier=args.spontaneous_sample_multiplier,
            mixed_num_high_rate_neurons=args.mixed_num_high_rate_neurons,
            mixed_rate_increase_factor=args.mixed_rate_increase_factor,
        )
        for tuning_type in tuning_types
    ]
    configs_to_run = [
        {
            **data_config,
            'selection_score': selection_score,
            'selection_activity': selection_activity,
        }
        for data_config in data_configs
        for selection_score in selection_scores
        for selection_activity in selection_activities
    ]
    max_n_samples: int = num_samples_values.max()

    db_connection = sqlite3.connect(args.db)
    try:
        for config in configs_to_run:
            logging.info(
                "\n===== Running Parallel Simulation Set for Config: "
                f"{config['tuning_type']}, {config['selection_score']} on "
                f"{config['selection_activity']} activity ====="
            )

            with Parallel(n_jobs=args.n_jobs) as parallel:
                jobs = [
                    delayed(run_single_simulation)(
                        run_id,
                        config,
                        max_n_samples,
                        num_samples_values,
                        cv_methods,
                        args.held_out_samples,
                    )
                    for run_id in range(config['num_simulation_runs'])
                ]
                results_from_runs = parallel(
                    tqdm(jobs, desc=f"Parallel Runs ({config['tuning_type']})")
                )

            all_results = [
                result
                for run_result_list in results_from_runs
                for result in run_result_list
            ]
            logging.info(f"Collected {len(all_results)} total results. Writing to database...")
            insert_results(db_connection, all_results)
    finally:
        db_connection.close()

    logging.info("All simulations complete. Results stored in database.")
