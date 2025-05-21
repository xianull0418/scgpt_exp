#!/bin/bash

# #############################################################################
# 运行 scGPT 反卷积微调流程的示例脚本
#
# 使用方法:
# 1. 将下面的占位符路径和参数替换为您的实际值。
# 2. 可选: 根据您的计算环境调整 GPU 设置或并行处理参数。
# 3. 在终端中执行脚本: bash run_deconvolution.sh
# #############################################################################

echo "开始执行 scGPT 反卷积微调流程..."

# --- 配置变量 ---

# 输入单细胞 AnnData 文件路径 (.h5ad)
# ADATA_FILE_PATH="/path/to/your/single_cell_data.h5ad"
ADATA_FILE_PATH="./data/pbmc3k_processed.h5ad" # 示例路径，请替换

# 预训练 scGPT 模型目录路径
# MODEL_DIR_PATH="/path/to/your/pretrained_scGPT_model_directory/"
MODEL_DIR_PATH="./scgpt_human" # 示例路径，请替换 (假设模型文件在此目录下)

# scGPT 词汇表文件路径 (通常在模型目录中，名为 vocab.json)
# VOCAB_FILE_PATH="/path/to/your/pretrained_scGPT_model_directory/vocab.json"
VOCAB_FILE_PATH="${MODEL_DIR_PATH}/vocab.json" # 假设词汇表在模型目录中

# 输出目录 (用于保存检查点、结果和日志)
OUTPUT_DIR="./deconv_finetune_output_example"

# 单细胞数据中细胞类型注释的列名 (在 .obs 中)
CELL_TYPE_COL="celltype" # 示例列名，请根据您的 AnnData 文件修改

# --- 伪批量样本生成参数 ---
N_PSEUDO_BULK_SAMPLES=1500    # 生成的伪批量样本数量 (示例值)
MIN_CELLS_PER_SAMPLE=50       # 每个伪批量样本的最小细胞数
MAX_CELLS_PER_SAMPLE=200      # 每个伪批量样本的最大细胞数
AGGREGATION_METHOD="sum"      # 基因表达聚合方法 ("sum" 或 "mean")

# --- 预处理参数 ---
NORM_TARGET_SUM=10000         # 标准化时的目标总和 (例如 CPM 为 1e4)
N_BINS=51                     # 基因表达值分箱数量 (scGPT 常用 51)
# HVG_COL=None                  # 可选: .var 中标记高变基因的列名，用于筛选伪批量数据中的基因 (默认为 None，不筛选)

# --- 模型参数 ---
FREEZE_SCGPT_BASE="--freeze_scgpt_base" # 是否冻结 scGPT 基础模型参数 (若要训练整个模型，请注释或移除此行)
# FREEZE_SCGPT_BASE="" # 取消冻结基础模型
HEAD_DROPOUT_RATE=0.1         # 反卷积头的 Dropout 率
# SCGPT_DROPOUT_RATE=None       # 可选: 覆盖 scGPT 内部的 Dropout 率 (默认为 None，使用原始值)

# --- 训练参数 ---
BATCH_SIZE=32                 # 批量大小
N_EPOCHS=30                   # 训练轮数 (示例值，可能需要根据数据调整)
LEARNING_RATE=1e-4            # 学习率
LOSS_TYPE="L1"                # 损失函数类型 ("L1", "MSE", "KLDivergence")
OPTIMIZER_TYPE="AdamW"        # 优化器类型 ("AdamW", "Adam")
SCHEDULER_TYPE="OneCycleLR"   # 学习率调度器类型 ("OneCycleLR", "LinearWarmup", "ReduceLROnPlateau", "None")
WEIGHT_DECAY=1e-5             # 权重衰减
WARMUP_FRAC=0.1               # 学习率调度器预热阶段占总步数的比例 (用于 OneCycleLR 和 LinearWarmup)

# --- 数据集/数据加载器参数 ---
CLS_VALUE_BIN=0               # CLS 标记在数据集中的分箱索引值
EXPRESSION_PAD_VALUE=-2       # 数据集中表达量填充值
NUM_WORKERS=0                 # DataLoader 使用的子进程数量 (大于0时可加速数据加载，但需注意 Windows 兼容性)

# --- 其他参数 ---
SEED=42                       # 随机种子，用于可复现性

# --- 构建 Python 执行命令 ---
# 使用 " \" 来换行，使命令更易读
PYTHON_CMD="python run_deconvolution_finetuning.py \
    --adata_file_path "${ADATA_FILE_PATH}" \
    --model_dir_path "${MODEL_DIR_PATH}" \
    --vocab_file_path "${VOCAB_FILE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --cell_type_col "${CELL_TYPE_COL}" \
    --n_pseudo_bulk_samples ${N_PSEUDO_BULK_SAMPLES} \
    --min_cells_per_sample ${MIN_CELLS_PER_SAMPLE} \
    --max_cells_per_sample ${MAX_CELLS_PER_SAMPLE} \
    --aggregation_method "${AGGREGATION_METHOD}" \
    --norm_target_sum ${NORM_TARGET_SUM} \
    --n_bins ${N_BINS} \
    --batch_size ${BATCH_SIZE} \
    --n_epochs ${N_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --loss_type "${LOSS_TYPE}" \
    --optimizer_type "${OPTIMIZER_TYPE}" \
    --scheduler_type "${SCHEDULER_TYPE}" \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_frac ${WARMUP_FRAC} \
    --head_dropout_rate ${HEAD_DROPOUT_RATE} \
    --cls_value_bin ${CLS_VALUE_BIN} \
    --expression_pad_value ${EXPRESSION_PAD_VALUE} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}"

# 可选: 添加 HVG 列参数 (如果定义了 HVG_COL)
# if [ ! -z "${HVG_COL}" ] && [ "${HVG_COL}" != "None" ]; then
#   PYTHON_CMD="${PYTHON_CMD} --hvg_col "${HVG_COL}""
# fi

# 可选: 添加 scGPT Dropout 率参数 (如果定义了)
# if [ ! -z "${SCGPT_DROPOUT_RATE}" ] && [ "${SCGPT_DROPOUT_RATE}" != "None" ]; then
#   PYTHON_CMD="${PYTHON_CMD} --scgpt_dropout_rate ${SCGPT_DROPOUT_RATE}"
# fi

# 添加冻结基础模型参数 (如果设置了)
if [[ ! -z "${FREEZE_SCGPT_BASE}" ]]; then
  PYTHON_CMD="${PYTHON_CMD} ${FREEZE_SCGPT_BASE}"
fi

# --- 执行命令 ---
echo "将要执行的命令:"
echo "${PYTHON_CMD}"
echo ""

# 执行 Python 脚本
eval ${PYTHON_CMD}

# 检查退出状态
if [ $? -eq 0 ]; then
  echo "反卷积微调流程成功完成。"
else
  echo "反卷积微调流程执行失败。"
  exit 1
fi

echo "脚本执行结束。"
