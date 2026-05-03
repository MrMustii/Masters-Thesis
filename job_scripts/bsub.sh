#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --account=project_465002687
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=320G
#SBATCH --partition=small-g
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# START_DATE="2019-03-29"
# END_DATE="2019-03-30"

START_DATE="2019-12-31"
END_DATE="2020-01-31"
DEGREE="1.0"
MODEL="modified"


# chmod -R o+rx Masters-Thesis/


# srun singularity exec --bind "$PWD:$PWD" containers/container.sif python3 Googles_gencast/gencast_test.py
# srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python -u gencast_infernce/datatest.py --start "$START_DATE" --end "$END_DATE" --degree "$DEGREE"
# srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python gencast_infernce/1year.py
# srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python -u gencast_infernce/1year.py --model "0.25" --start-date "$START_DATE" --end-date "$END_DATE"  --sanity-check
# srun singularity exec --bind $PWD:$PWD containers/container.sif python -u modified_Gencast/denoiser.py
# srun singularity exec --bind $PWD:$PWD containers/container.sif python -u gencast_infernce/ploting.py
# srun singularity exec \
#     --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
#     containers/container.sif \
#     python -u gencast_infernce/1year.py \
#         --model "$MODEL" \
#         --start-date "$START_DATE" \
#         --end-date "$END_DATE" \
#         --num-ensemble 1 \
#         --save-type nc
# rm -rf /scratch/project_465002687/data/era5/wb2_${START_DATE}_${END_DATE}_6h_0.25deg.zarr
# echo "Dataset deleted."

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    containers/container.sif \
    python -u modified_Gencast/training_test.py

# srun singularity exec \
#     --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
#     containers/container.sif \
#     python -u -m modified_Gencast.eval_model \
#         --model-path /scratch/project_465002687/checkpoints/epoch_015.npz \
#         --start-date 2020-04-09 \
#         --end-date 2020-04-16
