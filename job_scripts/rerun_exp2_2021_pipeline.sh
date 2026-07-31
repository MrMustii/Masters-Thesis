#!/bin/bash
# Orchestrator, run from the login node (repo root): bash job_scripts/rerun_exp2_2021_pipeline.sh
# Submits every job up front and chains them with --dependency=afterok, so the
# SLURM controller does the waiting, not this script — it exits in seconds and
# the pipeline keeps running even if you log off. Does not run any python
# itself; every python step is a separate job script, run inside the
# container, same as the MODE=exp2 lumi_eval.sh convention.
#
# Order:
#   0. MODE=prep_2021 lumi_build.sh   (single-threaded: download 2021 hourly ERA5 for the
#                           region + build the 2021 daily bucket) — EVERYTHING depends on this,
#                           because 7 parallel array tasks downloading 2021 at once would
#                           corrupt the zarr store.
#   1. delete old (2019) exp2 predictions/targets/metrics + figures
#   2. MODE=exp2 lumi_eval.sh (8-task array: 7 free-running variants + 1 combined
#                           run with boundary forcing on) after prep            \_ both must
#      lumi_report_exp1.sh (exp1, already 2021) after prep                      /  finish for 3
#   3. MODE=region lumi_compare.sh   (exp1 vs full-variable exp2) — deps: 2's both jobs
#   4. MODE=var    lumi_compare.sh   (all 7 free-running variants) — deps: eval array
#   5. MODE=margin lumi_compare.sh   (combined: replace vs no-replace) — deps: eval array
set -euo pipefail

PRED=/scratch/project_465002687/Predictions
VARIANTS="surface temp1000 temp850 z500 z700 combined all_gencast"

echo "=== Step 1: deleting old exp2 predictions/targets/metrics (~66.5GB) ==="
for variant in $VARIANTS; do
    rm -rf "$PRED/experiment_2_${variant}_fixed_stage2"
done

echo "=== Step 2: deleting old exp2 figures ==="
for variant in $VARIANTS; do
    rm -rf "figs/experiment_2_${variant}_fixed_stage2"
done

echo "=== Step 3: submitting 2021 data prep (single-threaded; everything waits on this) ==="
PREP_JOBID=$(MODE=prep_2021 sbatch --parsable job_scripts/lumi_build.sh)
echo "  prep job: $PREP_JOBID"

echo "=== Step 4: submitting exp2 array job (2021) and exp1, both after prep finishes OK ==="
EXP2_JOBID=$(MODE=exp2 sbatch --parsable --dependency=afterok:${PREP_JOBID} job_scripts/lumi_eval.sh)
EXP1_JOBID=$(sbatch --parsable --dependency=afterok:${PREP_JOBID} job_scripts/lumi_report_exp1.sh)
echo "  exp2 array job: $EXP2_JOBID   exp1 job: $EXP1_JOBID"

echo "=== Step 5: submitting compare 1 — region-reduction (runs after both finish OK) ==="
COMPARE1_JOBID=$(MODE=region sbatch --parsable --dependency=afterok:${EXP2_JOBID}:${EXP1_JOBID} job_scripts/lumi_compare.sh)
echo "  compare-region job: $COMPARE1_JOBID"

echo "=== Step 6: submitting compare 2 — variable-reduction (runs after the exp2 array finishes OK) ==="
COMPARE2_JOBID=$(MODE=var sbatch --parsable --dependency=afterok:${EXP2_JOBID} job_scripts/lumi_compare.sh)
echo "  compare-variants job: $COMPARE2_JOBID"

echo "=== Step 7: submitting compare 3 — LBC ablation (combined replace vs no-replace) ==="
COMPARE3_JOBID=$(MODE=margin sbatch --parsable --dependency=afterok:${EXP2_JOBID} job_scripts/lumi_compare.sh)
echo "  compare-margin job: $COMPARE3_JOBID"

echo
echo "All jobs submitted; SLURM will run them as dependencies resolve. Track with: squeue -u \$USER"
echo "Results land in figs/comparison/: region_reduction_exp1_vs_exp2full, exp2_all (variable sweep),"
echo "and margin_ablation_combined (lateral boundary conditions: replace vs no-replace)."
