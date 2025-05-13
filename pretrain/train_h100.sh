#!/bin/bash
# 资源配置

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

export NUM_GPUS=8  # 可根据实际情况修改
export GPU_IDS="0,1,2,3,4,5,6,7"  # 指定要使用的GPU ID

# 创建日志目录
mkdir -p /ai/home/jcw/Project/scGPT/pretrain/logs

# 停止可能正在运行的旧磁盘监控进程
echo "正在检查并停止之前可能运行的磁盘监控进程..."
OLD_MONITOR_PIDS=$(ps -ef | grep "while true; do.*df -h /ai/home /ai/data" | grep -v grep | awk '{print $2}')
if [ ! -z "$OLD_MONITOR_PIDS" ]; then
    echo "发现以下正在运行的磁盘监控进程: $OLD_MONITOR_PIDS"
    for pid in $OLD_MONITOR_PIDS; do
        echo "停止进程 $pid"
        kill -9 $pid 2>/dev/null
    done
    echo "所有旧监控进程已停止"
else
    echo "未发现正在运行的磁盘监控进程"
fi

# 设置日志文件名
LOG_FILE="/ai/home/jcw/Project/scGPT/pretrain/logs/h100_$(date +%Y%m%d_%H%M%S).log"
DISK_LOG_FILE="/ai/home/jcw/Project/scGPT/pretrain/logs/disk_usage_$(date +%Y%m%d_%H%M%S).log"
echo "所有输出将被记录到: $LOG_FILE"
echo "磁盘使用情况将被记录到: $DISK_LOG_FILE"

# 检查并设置GPU环境变量
if [ ! -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    # 计算实际使用的GPU数量
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "使用指定的 GPU: $GPU_IDS，共 $NUM_GPUS 个"
fi

# 设置 CUDA 设备和NCCL参数
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
# H100特定优化
export NCCL_P2P_LEVEL=NVL  # 使用NVLink进行P2P通信
export NCCL_ALGO=RING      # 使用环形算法，通常在H100上效果好

# 训练参数 - 与train_single.sh保持一致
DATASET="/ai/data/public/scgpt/human"
JOB_NAME="cellxgene_census_human_h100"
LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=36
LAYERS=12
MODEL_SCALE=8
SAVE_DIR="../checkpint_save"
VOCAB_PATH="/ai/home/jcw/Project/scGPT/scgpt/tokenizer/default_census_vocab.json"

# 设置之前训练的模型检查点路径（如果要继续训练）
CHECKPOINT="./save/cellxgene_census_human-Apr30-09-05-2025/"

# 打印训练配置信息
echo "开始训练 scGPT 模型 (H100优化):"
echo "- 使用 GPU 数量: $NUM_GPUS"
echo "- 数据集路径: $DATASET"
echo "- 模型配置: 层数=$LAYERS, 注意力头=8, 嵌入维度=$((MODEL_SCALE * 64))"
echo "- 训练配置: 批次大小=$per_proc_batch_size, 总批次大小=$((per_proc_batch_size * NUM_GPUS)), 训练轮数=6"
echo "- 性能优化: BF16混合精度, TF32矩阵乘法, Flash Attention"
if [ -f "$CHECKPOINT" ]; then
    echo "- 继续训练: 从检查点 $CHECKPOINT 加载模型"
fi

# 重定向所有输出到日志文件
{
torchrun --nproc_per_node=$NUM_GPUS --master_port=1234 pretrain_h100.py \
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
    --save-interval $(($LOG_INTERVAL * 3)) \
    --trunc-by-sample \
    --no-cls \
    --bf16 \
    --tf32 \
    --fast-transformer \
    $([[ -f "$CHECKPOINT" ]] && echo "--load-model $CHECKPOINT") |
    awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
} 2>&1 | tee -a $LOG_FILE

# 可选：添加磁盘使用监控
{
    while true; do
        echo "$(date +"%Y-%m-%d %H:%M:%S") - 磁盘使用情况:"
        df -h /ai/home /ai/data
        echo "--------------------------"
        sleep 300  # 每5分钟检查一次
    done
} > $DISK_LOG_FILE &
MONITOR_PID=$!

# 记录监控进程ID
echo "磁盘监控进程ID: $MONITOR_PID"