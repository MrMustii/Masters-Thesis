#!/bin/bash
#SBATCH --job-name=tests
#SBATCH --account=project_465002687
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=small-g
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err
#
# Unified test launcher. Pick with MODE (the header carries a GPU for the
# training smoke test; unit/part1 don't need it, pass --gpus-per-task=0 if you
# want to free it):
#
#   MODE=unit     sbatch job_scripts/lumi_tests.sh
#       Data/loader/stats/eval_daily/integrity unit tests (no model forward).
#   MODE=part1    sbatch job_scripts/lumi_tests.sh
#       Part I shared-foundation metrics on synthetic data with known answers.
#   MODE=training sbatch job_scripts/lumi_tests.sh config1.json [config2.json ...]
#       Training smoke test (runs the model forward) for each config passed.

mkdir -p logs

MODE="${MODE:-unit}"
PROJECT_DIR="/project/project_465002687/Masters-Thesis"
CONTAINER_PATH="$PROJECT_DIR/containers/my_new_container2.sif"

run_test() {
    srun singularity exec \
        --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
        "$CONTAINER_PATH" \
        python -u "$@"
}

EXIT_CODE=0
case "$MODE" in
    unit)
        for TEST in test_loaders test_normalization test_stats test_eval_daily test_data_integrity; do
            echo "############ $TEST ############"
            run_test "modified_Gencast/tests/$TEST.py" || EXIT_CODE=1
        done
        ;;
    part1)
        echo "############ validate_part1 ############"
        run_test "$PROJECT_DIR/modified_Gencast/tests/validate_part1.py" || EXIT_CODE=1
        ;;
    training)
        if [ "$#" -eq 0 ]; then
            echo "Usage: MODE=training sbatch lumi_tests.sh <config1.json> [config2.json ...]"
            exit 1
        fi
        ulimit -c 0
        export XLA_FLAGS="--xla_gpu_autotune_level=0"
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
        for CONFIG in "$@"; do
            echo "############ $CONFIG ############"
            run_test "modified_Gencast/tests/test_training_smoke.py" --config "$CONFIG" || EXIT_CODE=1
        done
        ;;
    *)
        echo "Unknown MODE='$MODE' (expected unit|part1|training)" >&2
        exit 1
        ;;
esac

echo "############ overall: $([ $EXIT_CODE -eq 0 ] && echo PASS || echo FAIL) ############"
exit $EXIT_CODE
