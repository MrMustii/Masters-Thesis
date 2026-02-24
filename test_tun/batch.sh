#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=singlegpu_gputest_example
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=00:05:00
#SBATCH --account=project_465002687

srun singularity exec --bind $PWD:$PWD my_container.sif python3 test.py
