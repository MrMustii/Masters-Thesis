module load CrayEnv
module load cotainr

mkdir -p /tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER


cotainr build container.sif --system=rocm-6.1 --conda-env=conda_venv.yml

# update 
module load PRoot
srun --account=project_465002687 --time=00:30:00 --mem=64G --partition=small singularity build my_new_container.sif update.def