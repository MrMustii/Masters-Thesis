#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --account=project_465002687
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# srun singularity exec --bind "$PWD:$PWD" containers/container.sif python3 Googles_gencast/gencast_test.py

# srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python gencast_infernce/datatest.py
srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python gencast_infernce/1year.py
