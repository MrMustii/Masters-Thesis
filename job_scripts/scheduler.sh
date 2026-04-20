#!/bin/sh
# Submit GenCast inference in 4 sequential batches (~3 months each).
# Each batch waits for the previous one to fully succeed (done()) before starting.

HPC="/work3/s215225/thiesis/Masters-Thesis/job_scripts/DTU_hpc.sh"

# Batch 1: Feb–Apr (indices 2-4)
bsub -J "gencast_b1[2-4]" < "$HPC"

# Batch 2: May–Jul (indices 5-7) — starts only when all of batch 1 succeed
bsub -w "done(gencast_b1)" -J "gencast_b2[5-7]" < "$HPC"

# Batch 3: Aug–Oct (indices 8-10)
bsub -w "done(gencast_b2)" -J "gencast_b3[8-10]" < "$HPC"

# Batch 4: Nov–Dec (indices 11-12)
bsub -w "done(gencast_b3)" -J "gencast_b4[11-12]" < "$HPC"
