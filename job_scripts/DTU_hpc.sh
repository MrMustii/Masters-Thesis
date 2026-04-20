#!/bin/sh
### General options
### -- specify queue 
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J gencast_inference[2-12]
# NOTE: scheduler.sh overrides -J with per-batch ranges; this line is the fallback only
### -- ask for number of cores --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- 300 GB total RAM: 75 GB per core x 4 cores --
#BSUB -R "rusage[mem=75GB]"
#BSUB -M 76GB
### -- request 1 x A100 80 GB GPU (satisfies >=60 GB VRAM requirement) --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
### -- set walltime limit: hh:mm --
#BSUB -W 10:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o Output_gc_2020_%I_%J.out
#BSUB -e Output_gc_2020_%I_%J.err
# Dates for inference run — scheduler.sh controls dependencies
last_day=(31 29 31 30 31 30 31 31 30 31 30 31)
prev=$((LSB_JOBINDEX - 1))
START_DATE="2020-$(printf '%02d' $prev)-${last_day[$((prev-1))]}"
END_DATE="2020-$(printf '%02d' $LSB_JOBINDEX)-${last_day[$((LSB_JOBINDEX-1))]}"
DATA_PATH="/work3/s215225/data/era5/wb2_${START_DATE}_${END_DATE}_6h_0.25deg.zarr"

# Project root on DTU HPC
PROJECT_DIR="/work3/s215225/thiesis/Masters-Thesis"
VENV="/work3/s215225/thiesis/.venv"

# Activate the venv
source "$VENV/bin/activate"

python -u "$PROJECT_DIR/gencast_infernce/1year.py" \
        --model "0.25" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --num-ensemble 1

if [ $? -eq 0 ] && [ -d "$DATA_PATH" ]; then
    echo "Deleting downloaded data: $DATA_PATH"
    rm -rf "$DATA_PATH"
fi
