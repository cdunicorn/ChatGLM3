#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
MAX_SOURCE_LEN=1024
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=32
MAX_STEP=1000
SAVE_INTERVAL=500

AUTORESUME_FROM_CHECKPOINT=True
DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=rups1

BASE_MODEL_PATH=/home/lvchao/projects/model/chatglm3-6b
DATASET_PATH=feeds/rups1.json
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --resume_from_checkpoint $AUTORESUME_FROM_CHECKPOINT 2>&1 | tee ${OUTPUT_DIR}/train.log
