#!/bin/bash
#SBATCH --job-name=report_exp2
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --array=0-7
#SBATCH --output=logs/report_exp2_%a_%j.out
#SBATCH --error=logs/report_exp2_%a_%j.err
#
# Metric caches for the exp2 input-variable sweep and the boundary-margin ablation.
#
#   sbatch job_scripts/lumi_report_exp2.sh

mkdir -p logs
IDX=${SLURM_ARRAY_TASK_ID:-0}
P=/scratch/project_465002687/Predictions
STAGE=step_00016000_2021-01-01_2021-12-31_ar14

NAMES=(retrained_all surface  temp1000 temp850 z500 z700 combined margin)
STORES=(all_gencast  surface  temp1000 temp850 z500 z700 combined combined)
TAILS=(""           ""       ""       ""      ""   ""   ""       _margin8.5)

NAME=${NAMES[$IDX]}
STORE=experiment_2_${STORES[$IDX]}_fixed_stage2
ARGS="--predictions $P/$STORE/predictions_${STORE}_${STAGE}${TAILS[$IDX]}.nc --name $NAME"

echo "task $IDX -> modified_Gencast.report_exp2 $ARGS"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.report_exp2 $ARGS
