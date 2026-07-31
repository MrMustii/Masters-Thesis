#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=project_465002687
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=350G
#SBATCH --partition=small-g
#SBATCH --array=0-7
#SBATCH --output=logs/eval_%a_%j.out
#SBATCH --error=logs/eval_%a_%j.err
#
# Unified eval launcher. Pick a pipeline with MODE and override --array/--gpus to
# match it (the header defaults to the heaviest case: 8 GPUs, array 0-7, 12 h):
#
#   MODE=exp2        sbatch --array=0-7 job_scripts/lumi_eval.sh
#       exp2 variable sweep on modified_Gencast.eval. Tasks 0-6 free-running
#       (margin 0), task 7 = combined with boundary forcing (margin 8.5). 8 GPUs.
#   MODE=recompute   sbatch --array=0-7 --gpus-per-task=1 --time=03:00:00 job_scripts/lumi_eval.sh
#       Re-report exp2 from CACHED predictions (adds --recompute-metrics). 1 GPU.
#       Task map: 0-5 var sweep, 6 = combined+margin8.5, 7 = all_gencast.
#   MODE=exp3        sbatch --array=0-2 --gpus-per-task=1 job_scripts/lumi_eval.sh
#       Continuous (exp3) eval on modified_Gencast.eval_continuous. 1 GPU.
#   MODE=exp4        sbatch --array=0-3 --gpus-per-task=1 job_scripts/lumi_eval.sh
#       Daily-stats (exp4) eval on modified_Gencast.eval_daily. 1 GPU.
#   MODE=noise_sweep sbatch --array=0-4 job_scripts/lumi_eval.sh
#       OU-rho/churn sweep on the exp3 1-day model (eval_continuous). 8 GPUs.
#   MODE=original    sbatch --array=0 job_scripts/lumi_eval.sh
#       Pretrained GenCast (d_model=512) on modified_Gencast.eval_original_weights.
#
# PLOTS_ONLY=1 appends --plots-only for exp2/exp3/noise_sweep (report from cached
# predictions, no inference — the eval_stage2 / report_noise_sweep use case).

mkdir -p logs

MODE="${MODE:-exp2}"
PLOTS_ONLY="${PLOTS_ONLY:-0}"
NUM_ENSEMBLE="${NUM_ENSEMBLE:-8}"
IDX=${SLURM_ARRAY_TASK_ID:-0}

export XLA_FLAGS="--xla_gpu_autotune_level=0"

case "$MODE" in
    exp2)
        MODULE="modified_Gencast.eval"
        START_DATE="2021-01-01"; END_DATE="2021-12-31"; AR_STEPS=14
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
        RUNS=(
            experiment_2_surface_fixed_stage2
            experiment_2_temp1000_fixed_stage2
            experiment_2_temp850_fixed_stage2
            experiment_2_z500_fixed_stage2
            experiment_2_z700_fixed_stage2
            experiment_2_combined_fixed_stage2
            experiment_2_all_gencast_fixed_stage2
            experiment_2_combined_fixed_stage2
        )
        MARGINS=(0 0 0 0 0 0 0 8.5)
        RUN_ID="${RUNS[$IDX]}"; MARGIN="${MARGINS[$IDX]}"
        echo "MODE=$MODE task $IDX → run=$RUN_ID margin=$MARGIN"
        EVAL_ARGS=(--run-id "$RUN_ID" --start-date "$START_DATE" --end-date "$END_DATE"
                   --autoregressive-steps "$AR_STEPS" --num-ensemble "$NUM_ENSEMBLE"
                   --boundary-margin-degrees "$MARGIN")
        [ "$PLOTS_ONLY" = "1" ] && EVAL_ARGS+=(--plots-only)
        ;;
    recompute)
        MODULE="modified_Gencast.eval"
        START_DATE="2021-01-01"; END_DATE="2021-12-31"; AR_STEPS=14
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
        RUNS=(
            experiment_2_surface_fixed_stage2
            experiment_2_temp1000_fixed_stage2
            experiment_2_temp850_fixed_stage2
            experiment_2_z500_fixed_stage2
            experiment_2_z700_fixed_stage2
            experiment_2_combined_fixed_stage2
            experiment_2_combined_fixed_stage2
            experiment_2_all_gencast_fixed_stage2
        )
        MARGINS=(0 0 0 0 0 0 8.5 0)
        RUN_ID="${RUNS[$IDX]}"; MARGIN="${MARGINS[$IDX]}"
        echo "MODE=$MODE task $IDX → run=$RUN_ID margin=$MARGIN"
        EVAL_ARGS=(--run-id "$RUN_ID" --start-date "$START_DATE" --end-date "$END_DATE"
                   --autoregressive-steps "$AR_STEPS" --num-ensemble "$NUM_ENSEMBLE"
                   --boundary-margin-degrees "$MARGIN" --recompute-metrics)
        ;;
    exp3)
        MODULE="modified_Gencast.eval_continuous"
        START_DATE="2019-01-01"; END_DATE="2019-12-31"; FORECAST_HOURS=168
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
        RUN_IDS=(variable_1day_stage2 variable_3day_stage2 variable_7day_stage2)
        RUN_ID="${RUN_IDS[$IDX]}"
        echo "MODE=$MODE task $IDX → run=$RUN_ID forecast=${FORECAST_HOURS}h"
        EVAL_ARGS=(--run-id "$RUN_ID" --start-date "$START_DATE" --end-date "$END_DATE"
                   --forecast-hours "$FORECAST_HOURS" --num-ensemble "$NUM_ENSEMBLE"
                   --noise-method ou --ou-rho 1)
        [ "$PLOTS_ONLY" = "1" ] && EVAL_ARGS+=(--plots-only)
        ;;
    exp4)
        MODULE="modified_Gencast.eval_daily"
        START_DATE="2020-01-01"; END_DATE="2020-12-31"
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
        RUN_IDS=(exp4_offsets_0 exp4_offsets_m1_0_p1 exp4_offsets_m2_m1_0 exp4_offsets_m2_m1_0_p1_p2)
        RUN_ID="${RUN_IDS[$IDX]}"
        echo "MODE=$MODE task $IDX → run=$RUN_ID"
        EVAL_ARGS=(--run-id "$RUN_ID" --start-date "$START_DATE" --end-date "$END_DATE"
                   --num-ensemble "$NUM_ENSEMBLE" --noise-method ou --ou-rho 0.1 --preload)
        ;;
    noise_sweep)
        MODULE="modified_Gencast.eval_continuous"
        RUN_ID="experiment_3_1day_stage2"
        START_DATE="2021-01-01"; END_DATE="2021-12-31"; FORECAST_HOURS=168
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
        # task 0 fixed rho0 churn0 | 1 fixed rho0 churn2.5 | 2 ou rho1 | 3 ou ln10 | 4 ou rho10000
        NOISE_METHODS=(fixed fixed ou ou ou)
        OU_RHOS=(0 0 1 2.302585093 10000)
        CHURNS=(0 2.5 0 0 0)
        NOISE_METHOD="${NOISE_METHODS[$IDX]}"; OU_RHO="${OU_RHOS[$IDX]}"; CHURN="${CHURNS[$IDX]}"
        echo "MODE=$MODE task $IDX → run=$RUN_ID noise=$NOISE_METHOD ou_rho=$OU_RHO churn=$CHURN"
        EVAL_ARGS=(--run-id "$RUN_ID" --start-date "$START_DATE" --end-date "$END_DATE"
                   --forecast-hours "$FORECAST_HOURS" --num-ensemble "$NUM_ENSEMBLE"
                   --noise-method "$NOISE_METHOD" --ou-rho "$OU_RHO" --stochastic-churn-rate "$CHURN")
        [ "$PLOTS_ONLY" = "1" ] && EVAL_ARGS+=(--plots-only)
        ;;
    original)
        MODULE="modified_Gencast.eval_original_weights"
        START_DATE="2021-01-01"; END_DATE="2021-12-31"; AR_STEPS=14
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
        echo "MODE=$MODE → pretrained GenCast (0.25) margin=8.5"
        EVAL_ARGS=(--model "0.25" --start-date "$START_DATE" --end-date "$END_DATE"
                   --autoregressive-steps "$AR_STEPS" --num-ensemble "$NUM_ENSEMBLE"
                   --boundary-margin-degrees 8.5)
        ;;
    *)
        echo "Unknown MODE='$MODE' (expected exp2|recompute|exp3|exp4|noise_sweep|original)" >&2
        exit 1
        ;;
esac

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m "$MODULE" "${EVAL_ARGS[@]}"
