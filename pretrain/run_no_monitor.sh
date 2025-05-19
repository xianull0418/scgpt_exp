#!/bin/bash
# 禁用监控屏障的简化启动脚本

# 设置环境变量
export TORCH_DISTRIBUTED_BARRIER_DISABLE_MONITORING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=23
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 获取参数
DATASET=${1:-"/ai/data/public/scgpt/human"}
SAVE_DIR=${2:-"./save/cellxgene_census_human-$(date +%b%d-%H-%M-%Y)"}
NUM_GPUS=${3:-8}
VOCAB_PATH=${4:-"/ai/home/jcw/Project/scGPT/scgpt/tokenizer/default_census_vocab.json"}

echo "启动训练，禁用监控屏障..."
echo "数据集: $DATASET"
echo "保存目录: $SAVE_DIR"
echo "GPU数量: $NUM_GPUS"
echo "词汇表路径: $VOCAB_PATH"

# 使用torchrun启动，带有简化参数
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --max_restarts=0 \
    fix_torchtext_warning.py \
    --data-source $DATASET \
    --save-dir $SAVE_DIR \
    --vocab-path $VOCAB_PATH \
    --valid-size-or-ratio 0.03 \
    --max-seq-len 1200 \
    --batch-size 128 \
    --eval-batch-size 256 \
    --nlayers 12 \
    --nheads 8 \
    --embsize 512 \
    --d-hid 512 \
    --epochs 6 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval 2000 \
    --save-interval 6000 \
    --trunc-by-sample \
    --no-cls \
    --bf16 \
    --tf32 \
    --fast-transformer true

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "训练失败，退出代码: $exit_code"
    echo "检查日志获取更多信息"
else
    echo "训练成功完成"
fi 