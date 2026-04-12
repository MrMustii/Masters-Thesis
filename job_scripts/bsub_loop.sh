#!/bin/bash
#SBATCH --job-name=gencast_models_test
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=300G
#SBATCH --partition=small-g
#SBATCH --array=0-3
#SBATCH --output=logs/gencast_%A_%a.out
#SBATCH --error=logs/gencast_%A_%a.err

# Define the model list
MODELS=("mini" "1" "0.25" "0.25_operational")

# Get the model for this specific task
MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Dates for the 1-year inference run
START_DATE="2019-03-28"
END_DATE="2019-03-30" 

echo "Starting task $SLURM_ARRAY_TASK_ID: Model=$MODEL_NAME"

# Create log directory if it doesn't exist
mkdir -p logs

# Run the inference
srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/container.sif \
    python -u gencast_infernce/1year.py \
    --model "$MODEL_NAME" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"