#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200

#SBATCH --array=0-24

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

unset LD_LIBRARY_PATH

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd $SCRATCH/online-q-chunking
cp -rf ~/online-q-chunking/* .

source .venv/bin/activate

env=""
group=""
random_replanning=0
action_shuffling=0
single_action_actor=0
big_net=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      env="$2"
      shift 2
      ;;
    --group)
      group="$2"
      shift 2
      ;;
    --random_replanning)
      random_replanning=1
      shift
      ;;
    --action_shuffling)
      action_shuffling=1
      shift
      ;;
    --single_action_actor)
      single_action_actor=1
      shift
      ;;
    --big_net)
      big_net=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$env" ]]; then
    echo "Error: --env is required"
    exit 1
fi

if [[ -z "$group" ]]; then
    echo "Error: --group is required"
    exit 1
fi

FLAGS="--num_evals 64 \
  --num_envs 512 \
  --num_eval_envs 512 \
  --discounting 0.99 \
  --action_repeat 1 \
  --unroll_length 60 \
  --min_replay_size 1000 \
  --max_replay_size 10000 \
  --contrastive_loss_fn sym_infonce \
  --energy_fn l2 \
  --train_step_multiplier 1 \
  --log_wandb"

if [[ $random_replanning -eq 1 ]]; then
    FLAGS="$FLAGS --random_replanning"
fi

if [[ $action_shuffling -eq 1 ]]; then
    FLAGS="$FLAGS --action_shuffling"
fi

if [[ $single_action_actor -eq 1 ]]; then
    FLAGS="$FLAGS --single_action_actor"
fi

if [[ $big_net -eq 1 ]]; then
    FLAGS="$FLAGS --total_env_steps 100000000 --batch_size 512 --n_hidden 8 --use_ln"
else
    FLAGS="$FLAGS --total_env_steps 60000000 --batch_size 256 --n_hidden 2"
fi

seeds=(0 1 2 3 4)
chunk_lengths=(1 3 5 10 15)

seed_idx=$((SLURM_ARRAY_TASK_ID / 5))
chunk_idx=$((SLURM_ARRAY_TASK_ID % 5))

seed=${seeds[$seed_idx]}
action_chunk_length=${chunk_lengths[$chunk_idx]}

jaxgcrl accrl \
    --env "$env" \
    --action_chunk_length "$action_chunk_length" \
    --seed "$seed" \
    --wandb_group "$group" \
    --exp_name "${env}_acl_${action_chunk_length}_seed_${seed}" \
    $FLAGS
