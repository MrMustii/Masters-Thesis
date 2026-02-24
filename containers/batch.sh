#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=update-container
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --account=project_465002687

module load CrayEnv
module load cotainr
module load PRoot
srun singularity build new_container.sif update.def