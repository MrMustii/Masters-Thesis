#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_465002687
#SBATCH --time=35:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --array=0-6
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#
# Unified training launcher. Pick a pipeline with the MODE env var; override
# --array to the config indices (0-based, in the order listed per mode) you want:
#
#   MODE=exp2     sbatch --array=0-6 job_scripts/lumi_train.sh
#       7 exp2 variable-set configs, stage 2, resuming from each run's stage-1 ckpt.
#   MODE=exp34    sbatch --array=0-6 job_scripts/lumi_train.sh
#       7 exp3/exp4 configs, BOTH stages from scratch (needs the full 35 h).
#   MODE=resume   sbatch --array=0,2 job_scripts/lumi_train.sh
#       Resume stage 2 for the exp3/exp4 configs that crashed mid-stage-2, from
#       their latest stage-2 checkpoint (LR continues at that step, not re-warmed).
#   MODE=variable sbatch --array=0-3 job_scripts/lumi_train.sh
#       Old variable/fixed-stride configs, stage 2, resuming from stage-1 ckpt.
#
# The header is sized for exp34's from-scratch run; exp2/variable/resume finish
# well inside 24 h and you can pass a shorter --time if you want.

mkdir -p logs

MODE="${MODE:-exp2}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
SEED="${SEED:-0}"

case "$MODE" in
    exp2)
        CONFIGS=(
            experiment_2_all_gencast.json
            experiment_2_combined.json
            experiment_2_surface.json
            experiment_2_temp1000.json
            experiment_2_temp850.json
            experiment_2_z500.json
            experiment_2_z700.json
        )
        STAGE=2
        ;;
    variable)
        CONFIGS=(
            variable_1day.json
            variable_3day.json
            variable_7day.json
            Fixed_12h.json
        )
        STAGE=2
        ;;
    exp34)
        CONFIGS=(
            experiment_3_1day.json
            experiment_3_3day.json
            experiment_3_7day.json
            experiment_4_offsets_0.json
            experiment_4_offsets_m1_0_p1.json
            experiment_4_offsets_m2_m1_0.json
            experiment_4_offsets_m2_m1_0_p1_p2.json
        )
        STAGE=both
        ;;
    resume)
        CONFIGS=(
            experiment_3_1day.json
            experiment_3_3day.json
            experiment_3_7day.json
            experiment_4_offsets_0.json
            experiment_4_offsets_m1_0_p1.json
            experiment_4_offsets_m2_m1_0.json
            experiment_4_offsets_m2_m1_0_p1_p2.json
        )
        STAGE=2
        ;;
    *)
        echo "Unknown MODE='$MODE' (expected exp2|variable|exp34|resume)" >&2
        exit 1
        ;;
esac

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$CONFIG" ]; then
    echo "No config at array index $SLURM_ARRAY_TASK_ID for MODE=$MODE" >&2
    exit 1
fi

ulimit -c 0
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

echo "MODE=$MODE  array task $SLURM_ARRAY_TASK_ID → config=$CONFIG  stage=$STAGE"

# Resolve the resume checkpoint for the modes that need one.
RESUME=""
START_STEP=""
if [ "$MODE" = "exp2" ] || [ "$MODE" = "variable" ]; then
    # Resume stage 2 from this run's stage-1 checkpoint (run_name = "<stem>_fixed").
    RUN_NAME="${CONFIG%.json}_fixed"
    STAGE1_DIR="/scratch/project_465002687/checkpoints/${RUN_NAME}_stage1"
    RESUME=$(ls -1 "$STAGE1_DIR"/step_*.pkl 2>/dev/null | sort | tail -1)
    if [ -z "$RESUME" ]; then
        echo "No stage-1 checkpoint found in $STAGE1_DIR" >&2
        exit 1
    fi
    echo "Resuming stage 2 from $RESUME"
elif [ "$MODE" = "resume" ]; then
    RUN_NAME="${CONFIG%.json}"
    STAGE2_DIR="/scratch/project_465002687/checkpoints/${RUN_NAME}_stage2"
    RESUME=$(ls -1 "$STAGE2_DIR"/step_*.pkl 2>/dev/null | sort | tail -1)
    if [ -z "$RESUME" ]; then
        echo "No stage-2 checkpoint in $STAGE2_DIR — nothing to resume." >&2
        exit 1
    fi
    START_STEP=$(basename "$RESUME" .pkl | sed 's/step_0*//')
    START_STEP=${START_STEP:-0}
    echo "Resuming stage 2 from $RESUME (LR continues at step $START_STEP)"
fi

TRAIN_ARGS=(--config "$CONFIG" --stage "$STAGE" --seed "$SEED" --save-every "$SAVE_EVERY")
[ -n "$RESUME" ] && TRAIN_ARGS+=(--resume "$RESUME")
[ -n "$START_STEP" ] && TRAIN_ARGS+=(--start-step "$START_STEP")

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u modified_Gencast/training.py "${TRAIN_ARGS[@]}"
