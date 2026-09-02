NUM_RUNS="${NUM_RUNS:-10000}"
CONDA_ENV="${CONDA_ENV:-leaky}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
TYPE1_DB="${TYPE1_DB:-sample_sweep_unsupervised_feature_selection_loocv_10000.db}"
TYPE2_DB="${TYPE2_DB:-neuron_sweep_gaussian_selection_activity_10000.db}"

if [[ -z "${N_JOBS:-}" ]]; then
    if [[ "${SLURM_NTASKS_PER_NODE:-}" =~ ^([0-9]+) ]]; then
        N_JOBS="${BASH_REMATCH[1]}"
    else
        N_JOBS="-1"
    fi
fi

source ~/.bashrc
conda activate "$CONDA_ENV"

echo "Using N_JOBS=$N_JOBS."

echo "Type 1: sample sweep, LOOCV, Gaussian/non-parametric x mean/variance."
python gamma_poisson.py \
    --overwrite \
    --db "$TYPE1_DB" \
    --num-runs "$NUM_RUNS" \
    --num-neurons 100 \
    --k-best-neurons 10 \
    --held-out-samples 10000 \
    --n-jobs "$N_JOBS" \
    --num-samples 20 25 30 35 40 45 50 55 60 \
    --cv loocv \
    --tuning-types gaussian_tuning_curve gamma_poisson_mixed \
    --selection-scores mean_firing_rate variance \
    --selection-activities evoked \
    --shape-multiplier 1 \
    --mixed-num-high-rate-neurons 5 \
    --mixed-rate-increase-factor 5 \
    --spontaneous-sample-multiplier 1

python fig4.py \
    --db "$TYPE1_DB" \
    --output-dir "$OUTPUT_DIR" \
    --cv loocv \
    --selection-activity evoked \
    --tuning-types gaussian_tuning_curve gamma_poisson_mixed \
    --selection-scores mean_firing_rate variance \
    --formats svg \
    --filename-suffix loocv

echo "Type 2: neuron sweep ratios, 5-fold CV, Gaussian tuning curves."
python gamma_poisson_neuron_sweep.py \
    --overwrite \
    --db "$TYPE2_DB" \
    --num-runs "$NUM_RUNS" \
    --num-samples 100 \
    --held-out-samples 10000 \
    --selection-fraction 0.10 \
    --cv 5-fold \
    --n-jobs "$N_JOBS" \
    --tuning-type gaussian_tuning_curve \
    --selection-scores mean_firing_rate variance \
    --selection-activities spontaneous evoked \
    --num-neurons 30 50 100 200 400 600 750

python fig4_neuron_sweep.py \
    --db "$TYPE2_DB" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples 100 \
    --cv 5-fold \
    --tuning-type gaussian_tuning_curve \
    --formats svg

echo "Done. SVG outputs are in $OUTPUT_DIR."
