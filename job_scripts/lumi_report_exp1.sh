#!/bin/bash
#SBATCH --job-name=report_exp1
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --array=0-1
#SBATCH --output=logs/report_exp1_%a_%j.out
#SBATCH --error=logs/report_exp1_%a_%j.err
#
# Metric caches for the pretrained checkpoint: globally, and on the regional crop.
#
#   sbatch job_scripts/lumi_report_exp1.sh

mkdir -p logs
IDX=${SLURM_ARRAY_TASK_ID:-0}
P=/scratch/project_465002687/Predictions

MODULES=(modified_Gencast.report_exp1 modified_Gencast.report_exp2)
ARGUMENTS=(
    "--run exp1"
    "--predictions $P/exp1_original_weights/predictions_exp1_original_weights_0.25_2021-01-01_2021-12-31_ar14_freerun.nc --name exp1_original_weights --group exp1"
)
MODULE="${MODULES[$IDX]}"
ARGS="${ARGUMENTS[$IDX]}"

echo "task $IDX -> $MODULE $ARGS"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m "$MODULE" $ARGS
