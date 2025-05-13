#!/bin/bash

# 资源配置（手动设置）
export NUM_GPUS=8  # 你可以根据实际情况修改

# 设置 CUDA 设备
export NCCL_IB_DISABLE=1  # NCCL IB

# check CUDA
./check_cuda.sh

# 训练参数
QUERY_NAME="blood"
DATASET="/ai/data/public/scgpt/human/${QUERY_NAME}/all_counts"
JOB_NAME="cellxgene_census_${QUERY_NAME}"
LOG_INTERVAL=200
VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=12
MODEL_SCALE=8
SAVE_DIR="../checkpint_save"
VOCAB_PATH="/ai/home/jcw/Project/scGPT/scgpt/tokenizer/default_census_vocab.json"

torchrun --nproc_per_node=$NUM_GPUS --master_port=1234 pretrain_init.py \
    --data-source $DATASET \
    --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --vocab-path ${VOCAB_PATH} \
    --valid-size-or-ratio $VALID_SIZE_OR_RATIO \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $((MODEL_SCALE * 64)) \
    --d-hid $((MODEL_SCALE * 64)) \
    --grad-accu-steps 1 \
    --epochs 6 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --save-interval $(($LOG_INTERVAL * 30)) \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 |
    awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'