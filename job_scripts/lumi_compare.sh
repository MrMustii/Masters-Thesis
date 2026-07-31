#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --account=project_465002687
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#
# Unified compare_runs launcher. Pick a comparison with MODE:
#
#   MODE=var    sbatch job_scripts/lumi_compare.sh   # exp2 variable sweep (7 sets vs full)
#   MODE=margin sbatch job_scripts/lumi_compare.sh   # combined free-running vs boundary forcing
#   MODE=region sbatch job_scripts/lumi_compare.sh   # exp1 global vs exp2 full-variable regional
#   MODE=noise  sbatch job_scripts/lumi_compare.sh   # exp3 1-day OU-rho/churn sweep
#
# All read the metrics_*.nc that the eval + baseline jobs already wrote.

mkdir -p logs

MODE="${MODE:-noise}"
PREDICTIONS_DIR="/scratch/project_465002687/Predictions"
RUN_ARGS=()

case "$MODE" in
    var)
        # Variable sweep: every reduced set difference-tested against the full set.
        RUNS=(
            experiment_2_surface_fixed_stage2
            experiment_2_temp1000_fixed_stage2
            experiment_2_temp850_fixed_stage2
            experiment_2_z500_fixed_stage2
            experiment_2_z700_fixed_stage2
            experiment_2_combined_fixed_stage2
            experiment_2_all_gencast_fixed_stage2
        )
        OUT_DIR="figs/comparison/exp2_all"
        for RUN_ID in "${RUNS[@]}"; do
            # *_ar14.nc only — excludes the *_ar14_margin8.5.nc LBC-ablation file.
            METRICS_FILE=$(ls "$PREDICTIONS_DIR/$RUN_ID"/metrics_*_ar14.nc 2>/dev/null | head -1)
            if [ -z "$METRICS_FILE" ]; then
                echo "WARNING: no free-running metrics file for $RUN_ID, skipping."
                continue
            fi
            RUN_ARGS+=("${METRICS_FILE}=${RUN_ID}")
        done
        for BASELINE in climatology persistence; do
            BASELINE_FILE=$(ls "$PREDICTIONS_DIR/$BASELINE"/metrics_${BASELINE}_experiment_2_combined_fixed_stage2_*.nc 2>/dev/null | head -1)
            [ -n "$BASELINE_FILE" ] && RUN_ARGS+=("${BASELINE_FILE}=${BASELINE}")
        done
        COMPARISON=var_margin
        REFERENCE=experiment_2_all_gencast_fixed_stage2
        ;;
    margin)
        # LBC ablation: combined free-running (margin 0) vs boundary forcing (margin 8.5).
        COMBINED_DIR="$PREDICTIONS_DIR/experiment_2_combined_fixed_stage2"
        FREE_METRICS=$(ls "$COMBINED_DIR"/metrics_*_ar14.nc 2>/dev/null | head -1)
        MARGIN_METRICS=$(ls "$COMBINED_DIR"/metrics_*_ar14_margin8.5.nc 2>/dev/null | head -1)
        if [ -z "$FREE_METRICS" ] || [ -z "$MARGIN_METRICS" ]; then
            echo "Missing metrics: free='$FREE_METRICS' margin='$MARGIN_METRICS'. Run MODE=exp2 eval (tasks 5 and 7) first."
            exit 1
        fi
        RUN_ARGS=("${FREE_METRICS}=no_replace" "${MARGIN_METRICS}=boundary_replace")
        OUT_DIR="figs/comparison/margin_ablation_combined"
        COMPARISON=var_margin
        REFERENCE=no_replace
        ;;
    region)
        # Region-reduction: global exp1 vs full-variable regional exp2 (all_gencast).
        EXP1_METRICS=$(ls "$PREDICTIONS_DIR/exp1"/metrics_*.nc 2>/dev/null | head -1)
        EXP2_METRICS=$(ls "$PREDICTIONS_DIR/experiment_2_all_gencast_fixed_stage2"/metrics_*.nc 2>/dev/null | head -1)
        if [ -z "$EXP1_METRICS" ] || [ -z "$EXP2_METRICS" ]; then
            echo "Missing metrics: exp1='$EXP1_METRICS' exp2_full='$EXP2_METRICS'. Run lumi_report_exp1.sh and MODE=exp2 eval first."
            exit 1
        fi
        RUN_ARGS=("${EXP1_METRICS}=exp1_global" "${EXP2_METRICS}=exp2_full_vars")
        for BASELINE in climatology persistence; do
            BASELINE_FILE=$(ls "$PREDICTIONS_DIR/$BASELINE"/metrics_${BASELINE}_exp1_*.nc 2>/dev/null | head -1)
            [ -n "$BASELINE_FILE" ] && RUN_ARGS+=("${BASELINE_FILE}=${BASELINE}")
        done
        OUT_DIR="figs/comparison/region_reduction_exp1_vs_exp2full"
        COMPARISON=region
        REFERENCE=exp1_global
        ;;
    noise)
        # OU-rho/churn sweep on the exp3 1-day model.
        RUN_ID="experiment_3_1day_stage2"
        START_DATE="2021-01-01"; END_DATE="2021-12-31"; AR_HOURS=168
        RUN_DIR="$PREDICTIONS_DIR/$RUN_ID"
        SUFFIXES=(fixed_churn0.0 fixed_churn2.5 ou_rho1.0_churn0.0 ou_rho2.302585093_churn0.0 ou_rho10000.0_churn0.0)
        LABELS=(fixed fixed_churn2.5 ou_rho1 ou_ln10 ou_inf)
        for i in "${!SUFFIXES[@]}"; do
            metrics_file=$(ls "$RUN_DIR"/metrics_${RUN_ID}_*_${START_DATE}_${END_DATE}_AR${AR_HOURS}h_${SUFFIXES[$i]}.nc 2>/dev/null | head -1)
            if [ -z "$metrics_file" ]; then
                echo "WARNING: no metrics file for noise=${SUFFIXES[$i]}, skipping."
                continue
            fi
            RUN_ARGS+=("${metrics_file}=${LABELS[$i]}")
        done
        for BASELINE in climatology persistence; do
            BASELINE_FILE=$(ls "$PREDICTIONS_DIR/$BASELINE"/metrics_${BASELINE}_${RUN_ID}_*.nc 2>/dev/null | head -1)
            [ -n "$BASELINE_FILE" ] && RUN_ARGS+=("${BASELINE_FILE}=${BASELINE}")
        done
        OUT_DIR="figs/comparison/exp3_1day_noise_sweep"
        COMPARISON=ou_rho
        REFERENCE=fixed
        ;;
    *)
        echo "Unknown MODE='$MODE' (expected var|margin|region|noise)" >&2
        exit 1
        ;;
esac

if [ "${#RUN_ARGS[@]}" -lt 2 ]; then
    echo "Need at least two runs to compare; found ${#RUN_ARGS[@]}. Did the evals/baselines finish?"
    exit 1
fi

srun singularity exec \
    --bind $PWD:$PWD,/scratch/project_465002687:/scratch/project_465002687 \
    containers/my_new_container2.sif \
    python -u -m modified_Gencast.compare_runs \
        --runs "${RUN_ARGS[@]}" \
        --comparison "$COMPARISON" \
        --reference "$REFERENCE" \
        --out "$OUT_DIR"
