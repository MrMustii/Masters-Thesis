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
#SBATCH --mail-type=END
#SBATCH --mail-user=s215225@dtu.dk

mkdir -p /tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
module load LUMI/25.03
module load cotainr
module load CrayEnv
cotainr build my_container.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.4.sif  --conda-env=conda_venv.yml