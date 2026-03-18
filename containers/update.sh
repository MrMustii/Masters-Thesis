#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=update-container
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=00:50:00
#SBATCH --mem=128G
#SBATCH --account=project_465002687
#SBATCH --mail-type=END
#SBATCH --mail-user=s215225@dtu.dk


module load LUMI/25.03
module load CrayEnv
module load cotainr
module load PRoot
singularity build containers/my_new_container.sif containers/update.def