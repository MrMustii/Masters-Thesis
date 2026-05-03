#!/bin/bash
#SBATCH --job-name=gencast_train
#SBATCH --account=project_465002687
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1cd 
#SBATCH --mem=320G
#SBATCH --partition=small-g
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/container.sif \
    python -u modified_Gencast/training_test.py
