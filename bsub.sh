#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

srun singularity exec --bind "$PWD:$PWD" containers/container.sif python3 Googles_gencast/gencast_test.py

