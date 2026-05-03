#!/bin/bash
#SBATCH --job-name=gencast_eval
#SBATCH --account=project_465002687
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=320G
#SBATCH --partition=small-g
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

mkdir -p logs

MODEL_PATH="/scratch/project_465002687/checkpoints/epoch_015.npz"
START_DATE="2020-04-09"
END_DATE="2020-04-16"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/container.sif \
    python -u -m modified_Gencast.eval_model \
        --model-path "$MODEL_PATH" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"
