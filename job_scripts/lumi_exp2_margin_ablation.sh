#!/bin/bash
#SBATCH --job-name=exp2_margin
#SBATCH --account=project_465002687
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=300G
#SBATCH --partition=small-g
#SBATCH --output=logs/exp2_margin_%j.out
#SBATCH --error=logs/exp2_margin_%j.err
#
# Rerun ONE exp2 variant with boundary forcing on (margin=8.5), then compare it
# against that same variant's already-run free-running (margin=0) metrics, to
# decide whether to keep boundary forcing for the final exp2 setup.
#
# Usage: sbatch job_scripts/lumi_exp2_margin_ablation.sh <variant>
#   e.g. sbatch job_scripts/lumi_exp2_margin_ablation.sh surface
# Run this AFTER lumi_compare.sh / rerun_exp2_2021_pipeline.sh, once you've
# picked the winning variant from the variable-reduction sweep.

mkdir -p logs

VARIANT="$1"
if [ -z "$VARIANT" ]; then
    echo "Usage: sbatch lumi_exp2_margin_ablation.sh <variant>  (e.g. surface, all_gencast, combined ...)"
    exit 1
fi

RUN_ID="experiment_2_${VARIANT}_fixed_stage2"
START_DATE="2021-01-01"
END_DATE="2021-12-31"
AR_STEPS=14
NUM_ENSEMBLE=8
PREDICTIONS_DIR="/scratch/project_465002687/Predictions/${RUN_ID}"

export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

FREE_RUNNING_METRICS=$(ls "$PREDICTIONS_DIR"/metrics_*_ar${AR_STEPS}.nc 2>/dev/null | head -1)
if [ -z "$FREE_RUNNING_METRICS" ]; then
    echo "No free-running (margin=0) metrics found for $RUN_ID at $PREDICTIONS_DIR — run the main exp2 sweep first."
    exit 1
fi

echo "=== Running $RUN_ID with boundary forcing ON (margin=8.5) ==="
srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.eval \
        --run-id "$RUN_ID" \
        --start-date "$START_DATE" --end-date "$END_DATE" \
        --autoregressive-steps "$AR_STEPS" --num-ensemble "$NUM_ENSEMBLE" \
        --boundary-margin-degrees 8.5

MARGIN_METRICS=$(ls "$PREDICTIONS_DIR"/metrics_*_margin8.5.nc 2>/dev/null | head -1)
if [ -z "$MARGIN_METRICS" ]; then
    echo "Margin run did not produce a metrics file as expected; check the eval log above."
    exit 1
fi

echo "=== Compare 3 — margin forcing vs free-running, for $VARIANT ==="
srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.compare_runs \
        --runs "${FREE_RUNNING_METRICS}=free_running" "${MARGIN_METRICS}=with_margin" \
        --comparison var_margin \
        --out "figs/comparison/margin_ablation_${VARIANT}"
