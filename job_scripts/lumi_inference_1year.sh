#!/bin/bash
### General options
### -- set the job Name and array range --
#SBATCH --job-name=run_notebook
#SBATCH --array=1-12
### -- account and partition --
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
### -- resources --
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=320G
### -- walltime --
#SBATCH --time=10:00:00
### -- output and error files. %x=job-name, %A=job-id, %a=array-index --
#SBATCH --output=output_%x_%A_%a.out
#SBATCH --error=error_%x_%A_%a.err

# Dates for inference run — one month per array task
last_day=(31 28 31 30 31 30 31 31 30 31 30 31)
prev=$((SLURM_ARRAY_TASK_ID - 1))
START_DATE="2020-$(printf '%02d' $prev)-${last_day[$((prev-1))]}"
END_DATE="2020-$(printf '%02d' $SLURM_ARRAY_TASK_ID)-${last_day[$((SLURM_ARRAY_TASK_ID-1))]}"

MODEL="modified"
DATA_PATH="/scratch/project_465002687/data/era5/wb2_${START_DATE}_${END_DATE}_6h_0.25deg.zarr"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/container.sif \
    python -u gencast_infernce/1year.py \
        --model "$MODEL" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --num-ensemble 1 \
        --save-type nc

if [ $? -eq 0 ] && [ -d "$DATA_PATH" ]; then
    echo "Deleting downloaded data: $DATA_PATH"
    rm -rf "$DATA_PATH"
fi
