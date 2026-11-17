#!/bin/bash
#SBATCH --job-name=report_exp3
#SBATCH --account=project_465002687
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --array=0-7
#SBATCH --output=logs/report_exp3_%a_%j.out
#SBATCH --error=logs/report_exp3_%a_%j.err
#
# Metric caches for the exp3 noise-design sweep and the reconstruction-horizon sweep.
#
#   sbatch job_scripts/lumi_report_exp3.sh

mkdir -p logs
IDX=${SLURM_ARRAY_TASK_ID:-0}
P=/scratch/project_465002687/Predictions
STAGE=step_00016000_2021-01-01_2021-12-31_AR168h

NAMES=(fixed  fixed_churn 	ou_rho1 	ou_rho2.3 	ou_rho10000 	1day 	3day 	7day)
MODELS=(1day  1day        	1day    	1day      	1day        	1day 	3day 	7day)
NOISE=(fixed_churn0.0 fixed_churn2.5 ou_rho1.0_churn0.0 ou_rho2.302585093_churn0.0 ou_rho10000.0_churn0.0 fixed_churn0.0 fixed_churn0.0 fixed_churn0.0)

NAME=${NAMES[$IDX]}
STORE=experiment_3_${MODELS[$IDX]}_stage2
ARGS="--predictions $P/$STORE/predictions_${STORE}_${STAGE}_${NOISE[$IDX]}_margin8.5.nc --name $NAME"

echo "task $IDX -> modified_Gencast.report_exp3 $ARGS"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.report_exp3 $ARGS
