#!/bin/bash
#SBATCH --job-name=report_baselines
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --array=0-10
#SBATCH --output=logs/report_baselines_%a_%j.out
#SBATCH --error=logs/report_baselines_%a_%j.err
#
# The CRPSS denominators: climatology and persistence on every case grid the models
# are scored on. Each borrows its grid from the targets file of a real run.
#
#   sbatch job_scripts/lumi_report_baselines.sh

mkdir -p logs
IDX=${SLURM_ARRAY_TASK_ID:-0}
P=/scratch/project_465002687/Predictions
EXP2=step_00016000_2021-01-01_2021-12-31_ar14
EXP3=step_00016000_2021-01-01_2021-12-31_AR168h
EXP4=step_00016000_2021-01-01_2021-12-31_daily_fixed

FIXED_LEAD=$P/experiment_2_all_gencast_fixed_stage2/targets_experiment_2_all_gencast_fixed_stage2_$EXP2.nc
CONTINUOUS=$P/experiment_3_1day_stage2/targets_experiment_3_1day_stage2_${EXP3}_fixed_churn0.0_margin8.5.nc
DAILY=$P/experiment_4_offsets_0_stage2/targets_experiment_4_offsets_0_stage2_${EXP4}_churn0.0.nc
NW_AMERICA=$P/experiment_4_offsets_m1_0_p1_stage2/targets_experiment_4_offsets_m1_0_p1_stage2_${EXP4}_nw_america_churn0.0.nc
N_AFRICA=$P/experiment_4_offsets_m1_0_p1_stage2/targets_experiment_4_offsets_m1_0_p1_stage2_${EXP4}_n_africa_churn0.0.nc

NAMES=(climatology  persistence  persistence_24h  climatology_hourly persistence_hourly climatology_daily persistence_daily climatology_nw_america persistence_nw_america climatology_n_africa persistence_n_africa)
GRIDS=(FIXED_LEAD   FIXED_LEAD   FIXED_LEAD       CONTINUOUS         CONTINUOUS         DAILY             DAILY             NW_AMERICA             NW_AMERICA             N_AFRICA             N_AFRICA)
KINDS=(climatology  persistence  persistence      climatology        persistence        climatology       persistence       climatology            persistence            climatology          persistence)
LAGS=(""            ""           "--lag-hours 24" ""                 ""                 ""                ""                ""                     ""                     ""                   "")

NAME=${NAMES[$IDX]}
TARGETS=${!GRIDS[$IDX]}
ARGS="--targets $TARGETS --baseline ${KINDS[$IDX]} --name $NAME ${LAGS[$IDX]}"

echo "task $IDX -> modified_Gencast.report_baselines $ARGS"

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.report_baselines $ARGS
