"""Shared I/O for the numbers behind a figure panel.

A panel script saves exactly what it plots; ``export_source_data.py`` then reads
that file instead of re-running the simulation.  Keeping the writer and the
reader in one place is the point: it is what stops the published figure and the
Source Data workbook from drifting apart.
"""

from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

import numpy as np
import pandas as pd

SCORES_KEY = "scores"
SHUFFLE_KEY = "shuffle"


def panel_results_path(script: Union[str, Path]) -> Path:
    """Map ``<script>.py`` to ``<script>_results.h5`` alongside it."""
    script = Path(script)
    return script.with_name(f"{script.stem}_results.h5")


def save_panel_tables(path: Union[str, Path], mode: str = "w", **tables: pd.DataFrame) -> Path:
    """Write one or more named tables to a panel results store."""
    path = Path(path)
    for index, (key, frame) in enumerate(tables.items()):
        frame.reset_index(drop=True).to_hdf(
            path, key=key, mode=mode if index == 0 else "a"
        )
    return path


def load_panel_tables(path: Union[str, Path], *keys: str) -> Tuple[pd.DataFrame, ...]:
    """Read named tables back out of a panel results store."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Re-run the panel script or notebook that writes it; "
            "it saves the numbers behind the panel so the source-data export never "
            "has to re-simulate them."
        )
    return tuple(pd.read_hdf(path, key=key) for key in keys)


def save_shuffle_control_results(
    script: Union[str, Path],
    scores: pd.DataFrame,
    shuffle_res: Mapping[str, np.ndarray],
) -> Path:
    """Save the decoding scores and shuffle control drawn in a panel.

    Parameters
    ----------
    script
        Path of the calling panel script; pass ``__file__``.
    scores
        Long-form frame of the plotted per-split scores, with at least
        ``accuracy``, ``method`` and ``region`` columns.
    shuffle_res
        Mapping of region to an ``(n_shuffles, n_splits)`` array of
        shuffled-label scores.  Only the per-shuffle mean across splits is
        stored, because that is what the shaded control interval is built from.

    Returns
    -------
    Path
        Where the results were written.
    """
    counts = [len(values) for values in shuffle_res.values()]
    shuffle = pd.DataFrame(
        {
            "region": np.repeat(list(shuffle_res), counts),
            "shuffle_id": np.concatenate([np.arange(n) for n in counts]),
            "mean_accuracy": np.concatenate(
                [values.mean(1) for values in shuffle_res.values()]
            ),
        }
    )
    return save_panel_tables(
        panel_results_path(script),
        **{SCORES_KEY: scores, SHUFFLE_KEY: shuffle},
    )


def load_shuffle_control_results(
    script: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load what :func:`save_shuffle_control_results` wrote for ``script``."""
    return load_panel_tables(panel_results_path(script), SCORES_KEY, SHUFFLE_KEY)


TRAJECTORIES_KEY = "trajectories"
PREDICTIONS_KEY = "predictions"


def save_gpfa_panel_results(
    path: Union[str, Path],
    trajectories: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
) -> Path:
    """Save the GPFA latent trajectories and held-out predictions a panel shows.

    Call this from ``gpfa.ipynb`` once the trajectories are fitted and the first
    cross-validation fold has been predicted, so the exported source data is the
    same realisation the figure is drawn from.

    Parameters
    ----------
    path
        Destination store, e.g. ``panel_results_path("gpfa.ipynb")``.
    trajectories
        ``(n_trials, n_latents, n_bins)`` array of single-trial latents; the
        trial-averaged path plotted alongside them is derived here.
    y_test, y_pred
        ``(n_samples, 2)`` true and predicted latent coordinates of the plotted
        fold, in the order the scatter received them.
    test_idx
        Held-out trial indices of that fold, as ``KFold`` yielded them.
    """
    rows = []
    for trial_id, trajectory in enumerate(trajectories, start=1):
        for time_bin in range(trajectory.shape[1]):
            rows.append(
                {
                    "trajectory_type": "single trial",
                    "trial_id": trial_id,
                    "time_bin": time_bin + 1,
                    "latent_1": trajectory[0, time_bin],
                    "latent_2": trajectory[1, time_bin],
                }
            )
    average = np.mean(trajectories, axis=0)
    for time_bin in range(average.shape[1]):
        rows.append(
            {
                "trajectory_type": "trial average",
                "trial_id": 0,
                "time_bin": time_bin + 1,
                "latent_1": average[0, time_bin],
                "latent_2": average[1, time_bin],
            }
        )

    test_idx = np.asarray(test_idx)
    n_bins = len(y_test) // len(test_idx)
    predictions = pd.DataFrame(
        {
            "observation_id": np.arange(1, len(y_test) + 1),
            "held_out_trial": np.repeat(test_idx + 1, n_bins),
            "time_bin": np.tile(np.arange(1, n_bins + 1), len(test_idx)),
            "true_angle_deg": np.rad2deg(np.arctan2(y_test[:, 1], y_test[:, 0])),
            "predicted_angle_deg": np.rad2deg(np.arctan2(y_pred[:, 1], y_pred[:, 0])),
        }
    )
    return save_panel_tables(
        path,
        **{TRAJECTORIES_KEY: pd.DataFrame(rows), PREDICTIONS_KEY: predictions},
    )


def load_gpfa_panel_results(
    path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load what :func:`save_gpfa_panel_results` wrote."""
    return load_panel_tables(path, TRAJECTORIES_KEY, PREDICTIONS_KEY)


def shuffle_control_interval(
    shuffle: pd.DataFrame, region: str, percentiles: Tuple[float, float] = (2.5, 97.5)
) -> Dict[str, float]:
    """Percentile interval of the shuffle control for one region."""
    low, high = np.percentile(
        shuffle.loc[shuffle["region"] == region, "mean_accuracy"], percentiles
    )
    return {"region": region, "lower_95_percent": low, "upper_95_percent": high}
