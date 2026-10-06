#!/usr/bin/env bash
# Train UniVerse on one node with torchrun.
#
#   bash scripts/train.sh <512|1024> [num_gpus (default 8)] [key=value config overrides ...]
#
#   512 : stage 1, 320x512, configs/train_512.yaml (initialised from ViewCrafter_25_sparse)
#   1024: stage 2, 576x1024, configs/train_1024.yaml (initialised from the stage-1 weights)
#
# The global batch is kept at 8 (as in the paper; per-GPU batch size 1) by gradient accumulation
# when fewer GPUs are used, e.g. 2 GPUs -> accumulate_grad_batches=4, so num_gpus must divide 8.
# Arguments containing '=' are config overrides (num_gpus may then be omitted), e.g.
#   bash scripts/train.sh 1024 8 model.pretrained_checkpoint=/path/to/stage1.ckpt \
#       data.params.train.params.data_dir=/path/to/data
# Logs, configs and checkpoints go to $SAVE_ROOT/universe_<stage> (default SAVE_ROOT: <repo>/save_dir).
set -e

USAGE="Usage: bash scripts/train.sh <512|1024> [num_gpus: 1, 2, 4 or 8 (default)] [key=value ...]"
if [ $# -lt 1 ]; then
    echo "$USAGE" >&2
    exit 1
fi
STAGE=$1
shift
NUM_GPUS=8
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
    NUM_GPUS=$1
    shift
fi

if [ "$STAGE" != "512" ] && [ "$STAGE" != "1024" ]; then
    echo "$USAGE" >&2
    exit 1
fi
GLOBAL_BATCH=8
# check the format first: no arithmetic on arbitrary strings
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || [ $((GLOBAL_BATCH % NUM_GPUS)) -ne 0 ]; then
    echo "num_gpus must be a positive integer that divides the global batch size ($GLOBAL_BATCH), got '$NUM_GPUS'" >&2
    echo "$USAGE" >&2
    exit 1
fi
ACCUM=$((GLOBAL_BATCH / NUM_GPUS))

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SAVE_ROOT=${SAVE_ROOT:-$REPO_ROOT/save_dir}
mkdir -p "$SAVE_ROOT"
SAVE_ROOT=$(cd "$SAVE_ROOT" && pwd)  # absolute logdir
NAME=universe_${STAGE}

# run from the repo root so that the relative paths in the configs resolve
cd "$REPO_ROOT"

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --master_port=${MASTER_PORT:-12352} \
    main/trainer.py \
    --base configs/train_${STAGE}.yaml \
    --train \
    --name $NAME \
    --logdir "$SAVE_ROOT" \
    --devices $NUM_GPUS \
    lightning.trainer.num_nodes=1 \
    lightning.trainer.devices=$NUM_GPUS \
    lightning.trainer.accumulate_grad_batches=$ACCUM \
    "$@"
