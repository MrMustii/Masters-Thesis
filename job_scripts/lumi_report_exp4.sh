#!/bin/bash
#SBATCH --job-name=report_exp4
#SBATCH --account=project_465002687
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --array=0-5
#SBATCH --output=logs/report_exp4_%a_%j.out
#SBATCH --error=logs/report_exp4_%a_%j.err
#
# Metric caches for the exp4 daily-statistic sweep, and the same checkpoint in two other regions.
#
#   sbatch job_scripts/lumi_report_exp4.sh

mkdir -p logs
IDX=${SLURM_ARRAY_TASK_ID:-0}
P=/scratch/project_465002687/Predictions
STAGE=step_00016000_2021-01-01_2021-12-31_daily_fixed

NAMES=(offsets_0 offsets_m1_0_p1 offsets_m2_m1_0 offsets_m2_m1_0_p1_p2 nw_america      n_africa)
MODELS=(offsets_0 offsets_m1_0_p1 offsets_m2_m1_0 offsets_m2_m1_0_p1_p2 offsets_m1_0_p1 offsets_m1_0_p1)
TAILS=("" "" "" "" _nw_america _n_africa)

NAME=${NAMES[$IDX]}
STORE=experiment_4_${MODELS[$IDX]}_stage2
ARGS="--predictions $P/$STORE/predictions_${STORE}_${STAGE}${TAILS[$IDX]}_churn0.0.nc --name $NAME"

echo "task $IDX -> modified_Gencast.report_exp4 $ARGS"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.report_exp4 $ARGS
