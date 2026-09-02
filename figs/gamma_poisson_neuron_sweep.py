# %%
import argparse
import sqlite3

import numpy as np
from joblib import Parallel, delayed
from loguru import logger as logging
from tqdm.auto import tqdm

from gamma_poisson import (
    DEFAULT_GAMMA_POISSON_SHAPE_MULTIPLIER,
    SELECTION_ACTIVITIES,
    SELECTION_SCORES,
    init_db,
    run_single_simulation,
)


DB_NAME = "data/neuron_sweep_evoked_variance.db"


def parse_num_neurons(values: list[str]) -> list[int]:
    """Parses whitespace- or comma-separated neuron-count arguments."""
    num_neurons = []
    for value in values:
        num_neurons.extend(int(item) for item in value.split(",") if item)
    return num_neurons


def parse_str_values(values: list[str]) -> list[str]:
    """Parses whitespace- or comma-separated string arguments."""
    parsed_values = []
    for value in values:
        parsed_values.extend(item for item in value.split(",") if item)
    return parsed_values


def build_config(
    num_neurons: int,
    num_simulation_runs: int,
    selection_fraction: float,
    tuning_type: str,
    shape_multiplier: float,
    selection_score: str,
    selection_activity: str,
) -> dict:
    k_best_neurons = max(1, int(np.ceil(num_neurons * selection_fraction)))
    tuning_params = {
        "num_high_rate_neurons": 0,
        "rate_increase_factor": 0.0,
        "spontaneous_sample_multiplier": 1,
    }
    if tuning_type == "gaussian_tuning_curve":
        tuning_params.update(
            {
                "tuning_curve_std_deg": 20.0,
                "max_firing_rate": 50.0,
                "baseline_firing_rate": 1.0,
            }
        )
    elif tuning_type == "gamma_poisson_mixed":
        tuning_params.update(
            {
                "shape_multiplier": shape_multiplier,
                "baseline_firing_rate": 0.0,
            }
        )
    else:
        raise ValueError(f"Unknown tuning_type: {tuning_type}")

    return {
        "num_neurons": num_neurons,
        "k_best_neurons": k_best_neurons,
        "num_simulation_runs": num_simulation_runs,
        "tuning_type": tuning_type,
        "xi_scale": 1.0,
        "selection_score": selection_score,
        "selection_activity": selection_activity,
        "tuning_params": tuning_params,
    }


def insert_results(db_connection: sqlite3.Connection, all_results: list[dict]):
    column_order = [
        "timestamp",
        "tuning_type",
        "num_samples",
        "num_neurons",
        "k_best_neurons",
        "xi_scale",
        "num_high_rate_neurons",
        "rate_increase_factor",
        "selection_score",
        "selection_activity",
        "baseline_firing_rate",
        "spontaneous_sample_multiplier",
        "shape_multiplier",
        "cv_type",
        "run_id_within_params",
        "leaky_mse",
        "non_leaky_mse",
        "leaky_gen_err_mse",
    ]
    data_to_insert = [[res[col] for col in column_order] for res in all_results]
    placeholders = ", ".join(["?"] * len(column_order))
    cursor = db_connection.cursor()
    cursor.executemany(
        f"INSERT INTO simulation_results ({', '.join(column_order)}) VALUES ({placeholders})",
        data_to_insert,
    )
    db_connection.commit()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep total neuron count for Gamma-Poisson simulations with "
            "unsupervised feature selection."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate the simulation_results table before running.",
    )
    parser.add_argument("--db", default=DB_NAME, help="SQLite database output path.")
    parser.add_argument("--num-runs", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--held-out-samples", type=int, default=2000)
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--cv", default="5-fold", choices=["5-fold", "loocv"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--tuning-type",
        default="gaussian_tuning_curve",
        choices=["gaussian_tuning_curve", "gamma_poisson_mixed"],
    )
    parser.add_argument(
        "--shape-multiplier",
        type=float,
        default=DEFAULT_GAMMA_POISSON_SHAPE_MULTIPLIER,
        help="Gamma shape multiplier used only for gamma_poisson_mixed.",
    )
    parser.add_argument(
        "--selection-scores",
        nargs="+",
        default=["mean_firing_rate", "variance"],
        help="Selection scores, as space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--selection-activities",
        nargs="+",
        default=["spontaneous", "evoked"],
        help="Selection activities, as space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--num-neurons",
        nargs="+",
        default=["30", "50", "100", "200", "400", "600", "750"],
        help="Neuron counts, as space-separated values or comma-separated groups.",
    )
    args = parser.parse_args()

    num_neurons_values = parse_num_neurons(args.num_neurons)
    selection_scores = parse_str_values(args.selection_scores)
    selection_activities = parse_str_values(args.selection_activities)
    unknown_scores = set(selection_scores) - SELECTION_SCORES
    unknown_activities = set(selection_activities) - SELECTION_ACTIVITIES
    if unknown_scores:
        raise ValueError(f"Unknown selection scores: {sorted(unknown_scores)}")
    if unknown_activities:
        raise ValueError(f"Unknown selection activities: {sorted(unknown_activities)}")

    init_db(args.db, overwrite=args.overwrite)

    db_connection = sqlite3.connect(args.db)
    try:
        for selection_score in selection_scores:
            for selection_activity in selection_activities:
                for num_neurons in num_neurons_values:
                    config = build_config(
                        num_neurons=num_neurons,
                        num_simulation_runs=args.num_runs,
                        selection_fraction=args.selection_fraction,
                        tuning_type=args.tuning_type,
                        shape_multiplier=args.shape_multiplier,
                        selection_score=selection_score,
                        selection_activity=selection_activity,
                    )
                    logging.info(
                        "Running neuron sweep: "
                        f"{args.tuning_type}, {selection_score} on "
                        f"{selection_activity}, d={num_neurons}, "
                        f"k={config['k_best_neurons']}, n={args.num_samples}, "
                        f"runs={args.num_runs}"
                    )
                    with Parallel(n_jobs=args.n_jobs) as parallel:
                        jobs = [
                            delayed(run_single_simulation)(
                                run_id,
                                config,
                                args.num_samples,
                                np.array([args.num_samples], dtype=int),
                                [(args.cv, None)],
                                args.held_out_samples,
                            )
                            for run_id in range(args.num_runs)
                        ]
                        results_from_runs = parallel(
                            tqdm(
                                jobs,
                                desc=(
                                    f"{selection_score}/{selection_activity}, "
                                    f"d={num_neurons}, k={config['k_best_neurons']}"
                                ),
                            )
                        )

                    all_results = [
                        result
                        for run_result_list in results_from_runs
                        for result in run_result_list
                    ]
                    insert_results(db_connection, all_results)
    finally:
        db_connection.close()

    logging.info(f"Neuron sweep complete. Results stored in {args.db}.")


if __name__ == "__main__":
    main()
