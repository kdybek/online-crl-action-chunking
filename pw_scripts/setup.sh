#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=small-g  # partition name
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --account=project_465003228
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

unset LD_LIBRARY_PATH

module load LUMI
module load cray-python/3.11.7

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd ~/online-crl-action-chunking

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install .
