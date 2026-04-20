#!/bin/sh
### General options
#BSUB -J gencast_download
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=60GB]"
#BSUB -M 63GB
#BSUB -W 04:00
#BSUB -B
#BSUB -N
#BSUB -o Output_download_%J.out
#BSUB -e Output_download_%J.err

START_DATE="2019-12-31"
END_DATE="2020-01-31"




source /work3/s215225/thiesis/.venv/bin/activate

python -u /work3/s215225/thiesis/Masters-Thesis/gencast_infernce/datatest.py \
    --start  "$START_DATE" \
    --end    "$END_DATE" \
    --degree "0.25"
