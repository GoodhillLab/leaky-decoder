# %%
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


DB_NAME = "data/neuron_sweep_evoked_variance.db"
OUTPUT_DIR = "figs/main"

plt.style.use(Path(__file__).with_name("main.mplstyle"))

PLOT_SELECTION_SCORES = ["mean_firing_rate", "variance"]
ERRORBAR_SE_MULTIPLIER = 3.0
SELECTION_ACTIVITY_LABELS = {
    "spontaneous": "Spontaneous selection",
    "evoked": "Evoked selection",
}
SELECTION_PALETTE = {
    SELECTION_ACTIVITY_LABELS["spontaneous"]: "blue",
    SELECTION_ACTIVITY_LABELS["evoked"]: "orange",
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
        f"neuron_sweep_{_filename_part(tuning_type)}_"
        f"{_filename_part(selection_score)}_"
        "spontaneous_vs_evoked_ratio"
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
        df[column] = df[column].apply(_sqlite_int)
    return df


def _ratio_of_mean_rmse(group: pd.DataFrame) -> pd.Series:
    leaky_mse = group["leaky_mse"].to_numpy()
    gen_mse = group["leaky_gen_err_mse"].to_numpy()
    mean_leaky = leaky_mse.mean()
    mean_gen = gen_mse.mean()
    rmse_ratio = np.sqrt(mean_leaky / mean_gen)

    covariance = np.cov(np.column_stack((leaky_mse, gen_mse)), rowvar=False, ddof=1)
    gradient = np.array(
        [
            rmse_ratio / (2 * mean_leaky),
            -rmse_ratio / (2 * mean_gen),
        ]
    )
    se = np.sqrt(gradient @ covariance @ gradient / len(group))
    return pd.Series({"rmse_ratio": rmse_ratio, "rmse_ratio_se": se})


def prepare_ratio_data(
    df: pd.DataFrame,
    num_samples: int,
    cv_type: str,
    tuning_type: str,
    selection_score: str,
) -> pd.DataFrame:
    df = df.query(
        "tuning_type == @tuning_type and "
        "selection_score == @selection_score and "
        "selection_activity in ['spontaneous', 'evoked'] and "
        "num_samples == @num_samples and "
        "cv_type == @cv_type"
    ).copy()

    if df.empty:
        raise ValueError(
            f"No matching rows for tuning_type={tuning_type}, "
            f"selection_score={selection_score}, num_samples={num_samples}, "
            f"cv_type={cv_type}."
        )
    missing_activities = set(SELECTION_ACTIVITY_LABELS) - set(df["selection_activity"])
    if missing_activities:
        raise ValueError(
            f"Missing selection activities for {selection_score}: "
            f"{sorted(missing_activities)}"
        )

    ratio_df = (
        df.groupby(["num_neurons", "selection_activity"])[
            ["leaky_mse", "leaky_gen_err_mse"]
        ]
        .apply(_ratio_of_mean_rmse)
        .reset_index()
    )
    ratio_df["selection"] = ratio_df["selection_activity"].map(SELECTION_ACTIVITY_LABELS)
    return ratio_df[["num_neurons", "selection", "rmse_ratio", "rmse_ratio_se"]]


def plot_ratio_panels(
    ratio_df: pd.DataFrame,
    output_stem: Path,
    save_formats: list[str],
):
    fig, ax = plt.subplots(figsize=(3, 2))
    for selection in [
        SELECTION_ACTIVITY_LABELS["spontaneous"],
        SELECTION_ACTIVITY_LABELS["evoked"],
    ]:
        selection_df = ratio_df.query("selection == @selection").sort_values(
            "num_neurons"
        )
        ax.errorbar(
            selection_df["num_neurons"],
            selection_df["rmse_ratio"],
            yerr=ERRORBAR_SE_MULTIPLIER * selection_df["rmse_ratio_se"],
            marker="o",
            alpha=0.7,
            color=SELECTION_PALETTE[selection],
            label=selection,
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xscale="log",
        xlabel="Number of neurons",
        ylabel="CV RMSE /\ngeneralization RMSE",
    )
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.2)
    [x.patch.set_visible(False) for x in (fig, ax)]
    for output_format in save_formats:
        fig.savefig(output_stem.with_suffix(f".{output_format}"), transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot neuron-count sweep ratios for spontaneous vs evoked selection."
    )
    parser.add_argument("--db", default=DB_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--cv", default="5-fold")
    parser.add_argument(
        "--tuning-type",
        default="gaussian_tuning_curve",
        choices=["gaussian_tuning_curve", "gamma_poisson_mixed"],
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

    df = fetch_data_from_db(args.db)
    for selection_score in PLOT_SELECTION_SCORES:
        ratio_df = prepare_ratio_data(
            df,
            num_samples=args.num_samples,
            cv_type=args.cv,
            tuning_type=args.tuning_type,
            selection_score=selection_score,
        )
        output_stem = output_dir / _output_stem(
            args.tuning_type,
            selection_score,
            filename_suffix=args.filename_suffix,
        )
        plot_ratio_panels(
            ratio_df,
            output_stem=output_stem,
            save_formats=output_formats,
        )


if __name__ == "__main__":
    main()
