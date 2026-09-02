# %%
import argparse
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger as logging

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


DB_NAME = "data/neural_decoding_simulations.db"
OUTPUT_DIR = "figs/main"

plt.style.use(Path(__file__).with_name("main.mplstyle"))

PLOT_CONDITIONS = [
    ("gaussian_tuning_curve", "mean_firing_rate"),
    ("gaussian_tuning_curve", "variance"),
    ("gamma_poisson_mixed", "mean_firing_rate"),
    ("gamma_poisson_mixed", "variance"),
]

METHOD_COLUMNS = {
    "CV (leaky)": "leaky_mse",
    "CV": "non_leaky_mse",
    "Generalization error": "leaky_gen_err_mse",
}
PIYG_PALETTE = sns.color_palette("PiYG")
METHOD_PALETTE = {
    "CV (leaky)": PIYG_PALETTE[0],
    "CV": PIYG_PALETTE[-1],
    "Generalization error": "black",
}


def _sqlite_int(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return int.from_bytes(value, byteorder="little")
    return int(value)


def _filename_part(value):
    return str(value).replace("-", "_").replace(" ", "_").replace(".", "p")


def _output_stem(
    tuning_type: str,
    selection_score: str,
    filename_suffix: str = "",
) -> str:
    stem = (
        f"sample_sweep_{_filename_part(tuning_type)}_"
        f"{_filename_part(selection_score)}_rmse"
    )
    if filename_suffix:
        stem = f"{stem}_{_filename_part(filename_suffix)}"
    return stem


def parse_str_values(values: list[str]) -> list[str]:
    """Parses whitespace- or comma-separated string arguments."""
    parsed_values = []
    for value in values:
        parsed_values.extend(item for item in value.split(",") if item)
    return parsed_values


def parse_output_formats(values: list[str]) -> list[str]:
    """Parses output formats and strips optional leading dots."""
    formats = [value.lstrip(".").lower() for value in parse_str_values(values)]
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def fetch_data_from_db(db_name: str = DB_NAME) -> pd.DataFrame:
    assert Path(db_name).is_file(), f"Database file '{db_name}' does not exist."
    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql_query("SELECT * FROM simulation_results", conn)
    finally:
        conn.close()

    for column in ["num_samples", "num_neurons", "k_best_neurons"]:
        if column in df.columns:
            df[column] = df[column].apply(_sqlite_int)
    if "selection_score" not in df.columns:
        df["selection_score"] = "variance"
    if "selection_activity" not in df.columns:
        df["selection_activity"] = "evoked"
    df["selection_score"] = df["selection_score"].fillna("variance")
    df["selection_activity"] = df["selection_activity"].fillna("evoked")
    logging.info(f"Fetched {len(df)} records from {db_name}.")
    return df


def prepare_rmse_data(
    df: pd.DataFrame,
    tuning_type: str,
    selection_score: str,
    selection_activity: str,
    cv_type: str,
) -> pd.DataFrame:
    df_filtered = df.query(
        "tuning_type == @tuning_type and "
        "selection_score == @selection_score and "
        "selection_activity == @selection_activity and "
        "cv_type == @cv_type"
    ).copy()
    if df_filtered.empty:
        raise ValueError(
            "No matching rows for "
            f"{tuning_type}, {selection_score}, {selection_activity}, {cv_type}."
        )

    plot_frames = []
    for method, mse_column in METHOD_COLUMNS.items():
        method_df = df_filtered[["num_samples", "run_id_within_params", mse_column]].copy()
        method_df = method_df.rename(columns={mse_column: "mse"})
        method_df["method"] = method
        plot_frames.append(method_df)

    plot_df = pd.concat(plot_frames, ignore_index=True)
    plot_df["rmse"] = np.sqrt(plot_df["mse"])
    return plot_df


def plot_rmse_vs_samples(
    plot_df: pd.DataFrame,
    output_stem: Path,
    save_formats: list[str],
):
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.lineplot(
        data=plot_df,
        x="num_samples",
        y="rmse",
        hue="method",
        hue_order=list(METHOD_COLUMNS),
        palette=METHOD_PALETTE,
        marker="o",
        alpha=0.7,
        errorbar="se",
        markerfacecolor="auto",
        markeredgewidth=0,
        ax=ax,
    )
    ax.set(
        xlabel="Number of samples",
        ylabel="Root mean squared\nerror (degrees)",
    )
    ax.legend(frameon=False)
    [x.patch.set_visible(False) for x in (fig, ax)]
    for output_format in save_formats:
        fig.savefig(output_stem.with_suffix(f".{output_format}"), transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate RMSE-vs-sample-count plots for feature-selection sweeps."
    )
    parser.add_argument("--db", default=DB_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--cv", default="5-fold")
    parser.add_argument("--selection-activity", default="evoked")
    parser.add_argument(
        "--tuning-types",
        nargs="+",
        default=None,
        help="Optional tuning types to plot; defaults to all configured conditions.",
    )
    parser.add_argument(
        "--selection-scores",
        nargs="+",
        default=None,
        help="Optional selection scores to plot; defaults to all configured conditions.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg", "pdf"],
        help="Output formats to save, as space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--filename-suffix",
        default="",
        help="Optional suffix appended to the output filename stem.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_formats = parse_output_formats(args.formats)
    all_simulation_data = fetch_data_from_db(args.db)
    plot_conditions = PLOT_CONDITIONS
    if args.tuning_types is not None or args.selection_scores is not None:
        requested_tuning_types = (
            set(parse_str_values(args.tuning_types))
            if args.tuning_types is not None
            else {condition[0] for condition in PLOT_CONDITIONS}
        )
        requested_selection_scores = (
            set(parse_str_values(args.selection_scores))
            if args.selection_scores is not None
            else {condition[1] for condition in PLOT_CONDITIONS}
        )
        plot_conditions = [
            condition
            for condition in PLOT_CONDITIONS
            if condition[0] in requested_tuning_types
            and condition[1] in requested_selection_scores
        ]
        if not plot_conditions:
            raise ValueError("No plot conditions match the requested filters.")

    for tuning_type, selection_score in plot_conditions:
        plot_df = prepare_rmse_data(
            all_simulation_data,
            tuning_type=tuning_type,
            selection_score=selection_score,
            selection_activity=args.selection_activity,
            cv_type=args.cv,
        )
        output_stem = output_dir / _output_stem(
            tuning_type,
            selection_score,
            filename_suffix=args.filename_suffix,
        )
        plot_rmse_vs_samples(
            plot_df,
            output_stem=output_stem,
            save_formats=output_formats,
        )
        saved_paths = [
            str(output_stem.with_suffix(f".{output_format}"))
            for output_format in output_formats
        ]
        logging.info(f"Saved {', '.join(saved_paths)}")


if __name__ == "__main__":
    main()
