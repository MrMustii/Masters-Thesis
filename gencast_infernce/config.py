"""
HPC-aware path configuration.
Detects which cluster we're on by checking known project directories,
and sets SCRATCH_PATH and GENCAST_LIB_PATH accordingly.
"""
import os

# ── LUMI (CSC) ──────────────────────────────────────────────────────────────
_LUMI_PROJECT = "/project/project_465002687"
_LUMI_SCRATCH  = "../../../scratch/project_465002687/"

# ── DTU HPC ──────────────────────────────────────────────────────────────────
_DTU_WORK    = "/work3/s215225/thiesis"
_DTU_SCRATCH = "/work3/s215225/data/era5/"

# ── Detection ────────────────────────────────────────────────────────────────
if os.path.isdir(_LUMI_PROJECT):
    SCRATCH_PATH   = _LUMI_SCRATCH
    DATA_PATH      = _LUMI_SCRATCH + "data/era5/"
    GENCAST_LIB    = _LUMI_PROJECT + "/Masters-Thesis/Googles_gencast"
    DEFAULT_OUTPUT = _LUMI_SCRATCH + "Predictions"
    WORKING_DIR    = _LUMI_PROJECT + "/Masters-Thesis"
elif os.path.isdir(_DTU_WORK):
    SCRATCH_PATH   = _DTU_SCRATCH
    DATA_PATH      = _DTU_SCRATCH
    GENCAST_LIB    = _DTU_WORK + "/Masters-Thesis/Googles_gencast"
    DEFAULT_OUTPUT = "/work3/s215225/Predictions"
    WORKING_DIR    = _DTU_WORK + "/Masters-Thesis"
else:
    # Fallback: resolve relative to this file's location
    _base          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    WORKING_DIR    = _base
    DATA_PATH      = os.path.join(_base, "data", "era5") + "/"
    SCRATCH_PATH   = os.path.join(_base, "scratch") + "/"
    GENCAST_LIB    = os.path.join(_base, "Googles_gencast")
    DEFAULT_OUTPUT = os.path.join(_base, "scratch", "Predictions")

CONFIGS_DIR = os.path.join(WORKING_DIR, "modified_Gencast", "traninig configs")