#!/bin/bash
#SBATCH --job-name=report_exp1
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --output=logs/report_exp1_%j.out
#SBATCH --error=logs/report_exp1_%j.err

mkdir -p logs

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.report_tmax \
        --zarr "experiment-1/predictions_2021-01-01_2021-12-31.zarr" \
        --config "modified_Gencast/traninig configs/experiment_2_surface.json"
