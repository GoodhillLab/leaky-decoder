import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


N_TRIALS = 5
N_SAMPLES_PER_TRIAL = 3
N_FOLDS = 5
GAP_SIZE = 2

TRAIN_COLOR = "#0072B2"
TEST_COLOR = "#D55E00"


def fold_ids_to_test_matrix(fold_ids):
    """Return a rows-by-time boolean matrix where True marks the held-out fold."""
    fold_ids = np.asarray(fold_ids)
    return fold_ids[None, :] == np.arange(N_FOLDS)[:, None]


def grouped_trial_cv_matrix():
    """Leave one full trial out per fold."""
    fold_ids = np.repeat(np.arange(N_TRIALS), N_SAMPLES_PER_TRIAL)
    return fold_ids_to_test_matrix(fold_ids)


def randomized_sample_cv_matrix(seed=0):
    """Randomized sample-wise k-fold CV, which mixes nearby time points."""
    fold_ids = np.arange(N_TRIALS * N_SAMPLES_PER_TRIAL) % N_FOLDS
    np.random.RandomState(seed).shuffle(fold_ids)
    return fold_ids_to_test_matrix(fold_ids)


def max_temporal_leakage_cv_matrix():
    """Sample-wise k-fold CV with every within-trial neighbor in a different fold."""
    fold_ids = np.arange(N_TRIALS * N_SAMPLES_PER_TRIAL) % N_FOLDS
    return fold_ids_to_test_matrix(fold_ids)


def sample_x(sample_idx):
    trial_idx = sample_idx // N_SAMPLES_PER_TRIAL
    sample_idx_within_trial = sample_idx % N_SAMPLES_PER_TRIAL
    return trial_idx * (N_SAMPLES_PER_TRIAL + GAP_SIZE) + sample_idx_within_trial


def add_cell(ax, sample_idx, row_idx, is_test):
    ax.add_patch(
        patches.Rectangle(
            (sample_x(sample_idx) - 0.4, row_idx - 0.4),
            0.8,
            0.8,
            facecolor=TEST_COLOR if is_test else TRAIN_COLOR,
            edgecolor="black",
            linewidth=0.5,
        )
    )


def add_trial_guides(ax, n_rows):
    for trial_idx in range(N_TRIALS):
        x_start = trial_idx * (N_SAMPLES_PER_TRIAL + GAP_SIZE)
        x_center = x_start + (N_SAMPLES_PER_TRIAL - 1) / 2
        ax.text(
            x_center,
            n_rows,
            f"Trial {trial_idx + 1}",
            ha="center",
            va="bottom",
            fontsize=11,
            style="italic",
            color="gray",
        )

        if trial_idx > 0:
            # Dashed separators mark trial boundaries where temporal correlation resets.
            ax.axvline(x_start - GAP_SIZE / 2, color="gray", linestyle="--", linewidth=1)


def draw_cv_plot(test_matrix, y_labels, filename):
    fig, ax = plt.subplots(figsize=(3, 2))
    n_rows, n_samples = test_matrix.shape

    for row_idx, row in enumerate(test_matrix):
        for sample_idx, is_test in enumerate(row):
            add_cell(ax, sample_idx, row_idx, is_test)

    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xlim(-1, sample_x(n_samples - 1) + 1)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.invert_yaxis()

    add_trial_guides(ax, n_rows)

    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Time →", fontsize=12, labelpad=15)

    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    fig.savefig(filename, transparent=True)
    plt.close(fig)


def create_cv_leakage_plot():
    draw_cv_plot(
        randomized_sample_cv_matrix(),
        [f"Fold {i + 1}" for i in range(N_FOLDS)],
        "cv_leakage_plot.svg",
    )


def create_correct_cv_plot():
    draw_cv_plot(
        grouped_trial_cv_matrix(),
        [f"Trial {i + 1}" for i in range(N_TRIALS)],
        "correct_cv_plot.svg",
    )


def create_max_cv_leakage_plot():
    draw_cv_plot(
        max_temporal_leakage_cv_matrix(),
        [f"Fold {i + 1}" for i in range(N_FOLDS)],
        "max_cv_leakage_plot.svg",
    )


if __name__ == "__main__":
    plt.style.use("main.mplstyle")
    create_cv_leakage_plot()
    create_correct_cv_plot()
    create_max_cv_leakage_plot()
