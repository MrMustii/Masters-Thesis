#!/bin/bash
#SBATCH --job-name=datatest_array
#SBATCH --account=project_465002687
#SBATCH --time=06:00:00        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=small-g
#SBATCH --array=0-3            
#SBATCH --output=logs/datatest_%A_%a.out
#SBATCH --error=logs/datatest_%A_%a.err

mkdir -p logs

STARTS=("2019-03-28" "2019-12-30" "2019-03-28" "2019-12-30")
ENDS=("2019-03-29" "2023-01-01" "2019-03-29" "2023-01-01")
DEGREES=("1" "1" "0.25" "0.25")

START_DATE=${STARTS[$SLURM_ARRAY_TASK_ID]}
END_DATE=${ENDS[$SLURM_ARRAY_TASK_ID]}
DEG=${DEGREES[$SLURM_ARRAY_TASK_ID]}

echo "Starting task $SLURM_ARRAY_TASK_ID: Date=$START_DATE to $END_DATE, Degree=$DEG"

srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/container.sif \
    python -u gencast_infernce/datatest.py \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --degree "$DEG"