#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --account=project_465002687
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=320G
#SBATCH --partition=-g
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

START_DATE="2019-03-29"
END_DATE="2019-03-30" 

# START_DATE="2019-12-30"
# END_DATE="2021-01-1" 
DEGREE="1.0"




# srun singularity exec --bind "$PWD:$PWD" containers/container.sif python3 Googles_gencast/gencast_test.py
# srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python -u gencast_infernce/datatest.py --start "$START_DATE" --end "$END_DATE" --degree "$DEGREE"
# srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python gencast_infernce/1year.py
srun singularity exec --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 containers/container.sif python -u gencast_infernce/1year.py --model "0.25" --start-date "$START_DATE" --end-date "$END_DATE"  --sanity-check
# srun singularity exec --bind $PWD:$PWD containers/container.sif python -u modified_Gencast/denoiser.py
# srun singularity exec --bind $PWD:$PWD containers/container.sif python -u gencast_infernce/ploting.py

