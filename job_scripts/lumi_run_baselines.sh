#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=project_465002687
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=350G
#SBATCH --partition=standard-g
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
# NOTE: mem/partition sized for the exp3/exp4 full-year HOURLY baselines (a ~27 GB
# climatology/persistence field per baseline + 27 GB targets + metric intermediates).
# exp2's 12h baselines are tiny and would fit in the old 64 GB/small-g, but this
# superset allocation is safe for every experiment. GPUs are idle (deterministic
# numpy work) — requested only because LUMI-G ties RAM to GCD count.
#
# Score the deterministic baselines (climatology + persistence, plus
# interpolated_truth for exp3 / rescaled_shape for exp4) through the exact same
# report path as the model run whose grid they copy. Their metrics files are
# what the compares use for the skill-% column and the exp3/exp4 reference CIs.
#
# Usage:
#   sbatch job_scripts/lumi_run_baselines.sh                          # exp2 full-variable run
#   RUN_ID=<run> REPORT=exp3 sbatch job_scripts/lumi_run_baselines.sh # any other run

mkdir -p logs

RUN_ID="${RUN_ID:-experiment_2_all_gencast_fixed_stage2}"
REPORT="${REPORT:-exp2}"
PREDICTIONS_DIR="/scratch/project_465002687/Predictions/$RUN_ID"

PREDICTIONS_FILE=$(ls "$PREDICTIONS_DIR"/predictions_*_ar14.nc 2>/dev/null | head -1)
if [ -z "$PREDICTIONS_FILE" ]; then
    PREDICTIONS_FILE=$(ls "$PREDICTIONS_DIR"/predictions_*.nc 2>/dev/null | head -1)
fi
TARGETS_FILE="${PREDICTIONS_FILE/predictions_/targets_}"

if [ -z "$PREDICTIONS_FILE" ] || [ ! -f "$TARGETS_FILE" ]; then
    echo "Missing cached predictions/targets for $RUN_ID: pred='$PREDICTIONS_FILE' tgt='$TARGETS_FILE'."
    exit 1
fi

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.run_baselines \
        --run-id "$RUN_ID" \
        --predictions "$PREDICTIONS_FILE" \
        --targets "$TARGETS_FILE" \
        --report "$REPORT"
