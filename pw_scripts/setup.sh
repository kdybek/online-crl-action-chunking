#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=plgcontrastivesft-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

unset LD_LIBRARY_PATH

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

mkdir -p $SCRATCH/online-crl-action-chunking
cd $SCRATCH/online-crl-action-chunking
cp -rf ~/online-crl-action-chunking/* .

python -m venv .venv
source .venv/bin/activate

pip install -e .
