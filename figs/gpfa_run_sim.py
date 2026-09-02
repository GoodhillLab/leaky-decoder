from loguru import logger as logging
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from gpfa_core import NeuralDataGenerator, GPFADecoder


def run_single_simulation(run_id: int, params: dict) -> dict:
    """
    Worker function for a single simulation run. This is the target for joblib.
    It returns a dictionary with the results for this run.
    """
    logging.debug(f"Starting simulation run #{run_id}...")

    # 1. Generate Data
    generator = NeuralDataGenerator(
        num_spiketrains=params["num_spiketrains"],
        trial_duration_s=params["trial_duration_s"],
        max_rate_hz=params.get("max_rate_hz", 150),
        baseline_hz=params.get("baseline_hz", 10.0),
        seed=run_id,  # Ensure each parallel run has a different seed
    )

    # Create main dataset for CV and a large held-out test set
    neural_data = generator.generate_trials(params["num_cv_trials"] + params["num_test_trials"])
    spiketrains_cv = neural_data[:params["num_cv_trials"]]
    spiketrains_test = neural_data[params["num_cv_trials"]:]

    # Get the binned ground truth trajectory
    bin_size_int = int(params["bin_size_ms"])
    num_steps = generator.num_steps
    true_trajectory = generator.oscillator_trajectory_2dim.reshape(
        2, num_steps // bin_size_int, bin_size_int
    ).mean(-1)

    # 2. Perform Decoding
    decoder = GPFADecoder(
        bin_size_ms=params["bin_size_ms"],
        latent_dimensionality=params["latent_dimensionality"],
        n_neighbors=params.get("n_neighbors", 5),
    )

    avg_leaky_rmse, avg_gen_err_rmse, avg_no_leak_rmse, avg_gen_err_rmse2 = (
        decoder.run_decoding_comparison(
            spiketrains_cv, spiketrains_test, true_trajectory
        )
    )

    logging.debug(
        f"Finished run #{run_id}. Leaky RMSE: {avg_leaky_rmse:.2f}, Gen Err RMSE: {avg_gen_err_rmse:.2f}"
    )

    return {
        "run_id": run_id,
        "leaky_rmse": avg_leaky_rmse,
        "gen_err_rmse": avg_gen_err_rmse,
        "no_leak_rmse": avg_no_leak_rmse,
        "gen_err_rmse2": avg_gen_err_rmse2,
    }


if __name__ == "__main__":
    # --- Simulation Parameters ---
    simulation_params = {
        "num_runs": 100,  # Total number of independent simulation runs
        "num_spiketrains": 50,
        "trial_duration_s": 2,
        "bin_size_ms": 20,
        "latent_dimensionality": 2,
        # Weak tuning depth (peak ~23 Hz over a 10 Hz baseline) puts decoding in the
        # ~24 degree regime, where the GPFA leakage is large enough to be visible.
        "max_rate_hz": 13,
        "num_cv_trials": 10,  # Number of trials in the main dataset for CV
        "num_test_trials": 100,  # Large number of trials for stable generalization error
    }

    output_filename = "gpfa_simulation_results.h5"

    # --- Run Simulations in Parallel ---
    logging.debug(
        f"Starting {simulation_params['num_runs']} simulation runs in parallel..."
    )

    with Parallel(n_jobs=-1) as parallel:
        # Create a list of jobs to be executed
        jobs = [
            delayed(run_single_simulation)(i, simulation_params)
            for i in range(simulation_params["num_runs"])
        ]

        # Execute jobs with a progress bar
        results = parallel(tqdm(jobs, desc="Running Simulations"))

    # --- Collect and Save Results ---
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        logging.info("\n--- Simulation Summary ---")
        logging.info(
            f"Average Leaky CV RMSE: {results_df['leaky_rmse'].mean():.2f} +/- {results_df['leaky_rmse'].std():.2f} degrees"
        )
        logging.info(
            f"Average Generalization Error RMSE: {results_df['gen_err_rmse'].mean():.2f} +/- {results_df['gen_err_rmse'].std():.2f} degrees"
        )

        # Save to HDF5 format
        results_df.to_hdf(output_filename, key="gpfa_results", mode="w")
        logging.info(f"Results successfully saved to {output_filename}")
    else:
        logging.error("No successful runs were completed. Nothing to save.")
