#!/bin/bash

# 参数设置
DATASET=${1:-"path_to/datasets/your_dataset.scb"}
OUTPUT_DIR=${2:-"./save/training-$(date +%b%d-%H-%M-%Y)"}
NUM_GPUS=${3:-8}
MAX_LENGTH=${4:-1536}
BATCH_SIZE=${5:-32}
EPOCHS=${6:-10}

# 设置进程间通信环境变量
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口，根据实际环境调整
export NCCL_DEBUG=INFO         # 增加NCCL日志级别便于调试
export NCCL_IB_TIMEOUT=23      # 增加InfiniBand超时
export NCCL_SOCKET_NTHREADS=8  # 增加socket线程数
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理
export NCCL_P2P_LEVEL=NVL      # 使用NVLink优先级通信
export NCCL_BUFFSIZE=4194304   # 增加缓冲区大小

# 设置PyTorch分布式环境变量
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 增加分布式调试日志
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用PyTorch NCCL异步错误处理

# 打印环境信息
echo "=============================================="
echo "启动分布式训练 - $(date)"
echo "数据集: $DATASET"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NUM_GPUS"
echo "最大序列长度: $MAX_LENGTH"
echo "批处理大小: $BATCH_SIZE"
echo "训练周期: $EPOCHS"
echo "=============================================="

# 使用torchrun启动分布式训练
torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_id=$(uuidgen) \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    pretrain_init.py \
    --data-source $DATASET \
    --save-dir $OUTPUT_DIR \
    --max-seq-len $MAX_LENGTH \
    --batch-size $BATCH_SIZE \
    --eval-batch-size $(($BATCH_SIZE * 2)) \
    --epochs $EPOCHS \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --bf16 \
    --fast-transformer \
    --vocab-path ../data/vocab.json

# 检查训练是否成功完成
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "训练失败，退出代码: $exit_code"
    # 收集错误日志
    mkdir -p $OUTPUT_DIR/error_logs
    find $OUTPUT_DIR -name "*.log" -exec cp {} $OUTPUT_DIR/error_logs/ \;
    # 保存环境变量到文件
    env > $OUTPUT_DIR/error_logs/environment.txt
else
    echo "训练成功完成！"
fi

echo "训练结束时间: $(date)" 