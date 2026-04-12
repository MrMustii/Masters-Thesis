#!/bin/bash
#SBATCH --job-name=gencast025
#SBATCH --account=project_465002687
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

START_DATE="2019-03-29"
END_DATE="2019-03-30"
ulimit -c 0 
SINGULARITYENV_ROCR_VISIBLE_DEVICES=0 \
srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/container.sif \
    python -u gencast_infernce/1year.py \
        --model "0.25" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --num-ensemble 1 \
        --sanity-check  
