#!/bin/bash
# 资源配置（手动设置）

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0  # 设为0提高性能

export NUM_GPUS=8  # 你可以根据实际情况修改
export GPU_IDS="0,1,2,3,4,5,6,7"  # 指定要使用的GPU ID，默认使用前NUM_GPUS个GPU

# 创建日志目录（确保存在）
mkdir -p ./logs

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

# 设置日志文件名（使用当前时间）
LOG_FILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"
DISK_LOG_FILE="./logs/disk_usage_$(date +%Y%m%d_%H%M%S).log"
echo "所有输出将被记录到: $LOG_FILE"
echo "磁盘使用情况将被记录到: $DISK_LOG_FILE"

# 增加系统限制
ulimit -n 65536  # 增加文件描述符限制
echo "已增加文件描述符限制: $(ulimit -n)"

# 设置OpenMP线程数，避免系统过载
export OMP_NUM_THREADS=$(($(nproc --all) / NUM_GPUS))
echo "设置每个进程的OpenMP线程数: $OMP_NUM_THREADS"

# 检查并设置GPU环境变量
if [ ! -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    # 计算实际使用的GPU数量（用于torchrun）
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "使用指定的 GPU: $GPU_IDS，共 $NUM_GPUS 个"
fi

# 设置 CUDA 设备和NCCL参数
export NCCL_IB_DISABLE=1  # NCCL IB
export NCCL_DEBUG=WARNING  # 降低NCCL日志级别，减少日志量
export NCCL_P2P_DISABLE=0  # 确保启用P2P通信
export NCCL_SOCKET_IFNAME=lo  # 使用本地回环接口提高稳定性 
export NCCL_BLOCKING_WAIT=1  # 启用阻塞等待以提高稳定性
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理
export NCCL_TIMEOUT=3600  # 设置NCCL超时时间为1小时

# PyTorch配置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # 添加内存分配配置

# DDP配置
export TORCH_DISTRIBUTED_DEBUG=OFF  # 关闭分布式调试信息
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用PyTorch NCCL异步错误处理


# 训练参数
DATASET="/ai/data/public/scgpt/human"
JOB_NAME="cellxgene_census_human_off_0.5B"
LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=120
LAYERS=12
MODEL_SCALE=8
SAVE_DIR="../checkpint_save"
VOCAB_PATH="/ai/home/jcw/Project/scGPT/scgpt/tokenizer/default_census_vocab.json"

# 设置之前训练的模型检查点路径（如果要继续训练）
# CHECKPOINT="./save/cellxgene_census_human-Apr30-09-05-2025/"

# 打印训练配置信息
echo "开始训练 scGPT 模型:"
echo "- 使用 GPU 数量: $NUM_GPUS"
echo "- 数据集路径: $DATASET"
echo "- 模型配置: 层数=$LAYERS, 注意力头=8, 嵌入维度=$((MODEL_SCALE * 64))"
echo "- 训练配置: 批次大小=$per_proc_batch_size, 总批次大小=$((per_proc_batch_size * NUM_GPUS)), 训练轮数=6"
if [ -n "$CHECKPOINT" ]; then
    echo "- 继续训练: 从检查点 $CHECKPOINT 加载模型"
fi

# 开始记录磁盘使用情况
(
    while true; do
        echo "================== $(date) =================="
        df -h /ai/home /ai/data
        echo -e "\n"
        sleep 300  # 每5分钟记录一次
    done
) > "$DISK_LOG_FILE" 2>&1 &
DISK_MONITOR_PID=$!
echo "磁盘监控进程已启动，PID: $DISK_MONITOR_PID"

# 创建添加时间戳的函数
add_timestamp() {
    while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done
}

# 重定向所有输出（标准输出和错误输出）到日志文件，并添加时间戳
{
echo "============ $(date) 训练开始 ============"
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:1234 \
    --max_restarts=0 \
    --master_port=1234 \
    pretrain_init.py \
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
    $([[ -n "$CHECKPOINT" ]] && echo "--load-model $CHECKPOINT") 2>&1 | add_timestamp

echo "============ $(date) 训练结束 ============"

# 训练结束后停止磁盘监控
if [ -n "$DISK_MONITOR_PID" ]; then
    echo "正在停止磁盘监控进程 (PID: $DISK_MONITOR_PID)"
    kill $DISK_MONITOR_PID 2>/dev/null || true
fi

} 2>&1 | tee -a "$LOG_FILE"

echo "训练过程已完成，日志已保存到: $LOG_FILE"


