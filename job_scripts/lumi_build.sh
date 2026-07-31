#!/bin/bash
#SBATCH --job-name=build
#SBATCH --account=project_465002687
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=290G
#SBATCH --partition=small-g
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err
#
# Unified data-build launcher (CPU work; the header is sized for the heaviest
# build, exp5's 48 h — pass a shorter --time for the others). Pick with MODE:
#
#   MODE=daily_bucket sbatch job_scripts/lumi_build.sh
#       Daily-statistics bucket for the config's train+val span. To APPEND a year
#       (append-safe): MODE=daily_bucket START_DATE=2021-01-01 END_DATE=2021-12-31 sbatch ...
#   MODE=stats        sbatch job_scripts/lumi_build.sh config1.json [config2.json ...]
#       Normalization stats, one per config passed as a positional arg.
#   MODE=climatology  sbatch job_scripts/lumi_build.sh   # hourly-climatology baseline store
#   MODE=tmax_threshold sbatch job_scripts/lumi_build.sh # +/-15-day windowed p90 Tmax threshold
#   MODE=exp5         sbatch job_scripts/lumi_build.sh   # nw_america + n_africa region data (resumable)
#   MODE=tp12         sbatch job_scripts/lumi_build.sh   # rebuild tp12 store (11 std vars + tp_12hr)
#   MODE=prep_2021    sbatch job_scripts/lumi_build.sh   # 2021 hourly ERA5 + daily bucket (single pass)

mkdir -p logs

MODE="${MODE:-daily_bucket}"
PROJECT_DIR="/project/project_465002687/Masters-Thesis"
CONTAINER_PATH="$PROJECT_DIR/containers/my_new_container2.sif"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Container not found at $CONTAINER_PATH"
    exit 1
fi

run_build() {
    srun singularity exec \
        --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
        "$CONTAINER_PATH" \
        python -u "$@"
}

case "$MODE" in
    daily_bucket)
        DATE_ARGS=()
        [ -n "$START_DATE" ] && DATE_ARGS+=(--start-date "$START_DATE")
        [ -n "$END_DATE" ] && DATE_ARGS+=(--end-date "$END_DATE")
        echo "Building daily-statistics bucket... ${DATE_ARGS[*]:-(default train+val span)}"
        run_build "$PROJECT_DIR/modified_Gencast/build_daily_bucket.py" "${DATE_ARGS[@]}"
        ;;
    stats)
        if [ "$#" -eq 0 ]; then
            echo "Usage: MODE=stats sbatch lumi_build.sh <config1.json> [config2.json ...]"
            exit 1
        fi
        for CONFIG in "$@"; do
            echo "======================================== Building stats for: $CONFIG"
            run_build "$PROJECT_DIR/modified_Gencast/build_stats.py" "$CONFIG"
        done
        ;;
    climatology)
        echo "Building hourly-climatology baseline store from the hourly store..."
        run_build "$PROJECT_DIR/modified_Gencast/build_climatology.py"
        ;;
    tmax_threshold)
        echo "Building +/-15-day windowed p90 daily-Tmax threshold (2005-2019 pool)..."
        run_build "$PROJECT_DIR/modified_Gencast/build_tmax_threshold.py"
        ;;
    exp5)
        echo "Building exp5 region data (nw_america + n_africa)..."
        run_build "$PROJECT_DIR/modified_Gencast/build_exp5_regions.py"
        ;;
    tp12)
        echo "Rebuilding tp12 store (2021)..."
        run_build "$PROJECT_DIR/modified_Gencast/build_tp12_store.py" \
            --start-date 2021-01-01 --end-date 2021-12-31
        ;;
    prep_2021)
        echo "Prepping 2021 hourly ERA5 + daily bucket (single-threaded pass)..."
        run_build "$PROJECT_DIR/modified_Gencast/build_daily_bucket.py" \
            --start-date 2021-01-01 --end-date 2021-12-31
        ;;
    *)
        echo "Unknown MODE='$MODE' (expected daily_bucket|stats|climatology|tmax_threshold|exp5|tp12|prep_2021)" >&2
        exit 1
        ;;
esac
