# scGPT 反卷积微调流程中文 README

本项目包含一系列 Python 脚本，用于基于 scGPT 模型进行单细胞数据反卷积的微调。反卷积的目标是根据细胞的基因表达谱，推断混合样本中不同细胞类型的比例。

## 流程概览

1.  **数据准备**: 从原始单细胞 AnnData 数据开始，筛选与预训练 scGPT 模型词汇表共有的基因。
2.  **伪批量样本生成**: 利用筛选后的单细胞数据，生成模拟的批量 RNA-seq 样本（伪批量样本），并记录每个伪批量样本中已知细胞类型的真实比例。
3.  **伪批量数据预处理**: 对生成的伪批量样本进行预处理，包括标准化、log1p 转换和基因表达值分箱，以适配 scGPT 模型输入。
4.  **模型定义**: 加载预训练的 scGPT 模型，并在其基础上添加一个专门用于反卷积任务的预测头。可以选择冻结 scGPT 基础模型的参数，仅微调预测头。
5.  **数据加载器创建**: 为预处理后的训练和验证伪批量数据集创建 PyTorch DataLoader。
6.  **微调训练**: 使用定义的模型、数据加载器、损失函数、优化器和学习率调度策略，对模型进行微调。
7.  **执行脚本**: `run_deconvolution_finetuning.py` 是主执行脚本，整合了上述所有步骤，并可通过命令行参数进行配置。

## 脚本说明

以下是各个 Python 脚本的功能摘要：

*   **`prepare_scgpt_data.py`**:
    *   加载单细胞 AnnData 对象 (`.h5ad` 文件) 和 scGPT 模型词汇表 (`vocab.json`)。
    *   筛选 AnnData 对象，使其仅包含与模型词汇表共有的基因，确保后续处理的基因一致性。

*   **`create_pseudo_bulk.py`**:
    *   根据输入的单细胞 AnnData 对象（已筛选基因）生成伪批量 RNA-seq 样本。
    *   用户可以指定生成的伪批量样本数量、每个样本中包含的细胞数量范围以及基因表达的聚合方法（如求和或均值）。
    *   输出一个 AnnData 对象，其中 `.X` 存储伪批量基因表达，`.obs` 存储每个样本中细胞类型的真实比例。

*   **`preprocess_bulk_for_scgpt.py`**:
    *   对 `create_pseudo_bulk.py` 生成的伪批量 AnnData 对象进行预处理。
    *   主要步骤包括：
        *   可选的基于高变基因 (HVG) 的基因筛选。
        *   总计数标准化（例如，归一化到每百万计数 CPM）。
        *   Log1p 转换。
        *   将log转换后的表达值分箱（binning）为离散整数，以符合 scGPT 模型的输入要求。
    *   处理后的表达数据（如分箱后的值）通常存储在 AnnData 对象的层（`.layers`）中。

*   **`deconvolution_model_def.py`**:
    *   定义并创建用于反卷积微调的 PyTorch 模型。
    *   加载预训练的 scGPT 模型权重和配置。
    *   在 scGPT 模型基础上添加一个“反卷积头”（通常是几个线性层，最后通过 Softmax 输出细胞类型比例）。
    *   提供选项以冻结 scGPT 基础模型的参数，从而仅训练新添加的反卷积头。

*   **`deconvolution_dataloaders.py`**:
    *   定义一个自定义的 PyTorch `Dataset` 类 (`PseudoBulkDataset`)，用于处理预处理后的伪批量数据。
    *   该 Dataset 将每个伪批量样本转换为模型所需的输入格式，包括基因 ID 序列、对应的分箱表达值、注意力机制的填充掩码 (padding mask) 以及真实的细胞类型比例（作为标签）。
    *   提供一个函数 `create_dataloaders` 来为训练集和验证集创建相应的 DataLoader 实例。

*   **`training_utils.py`**:
    *   包含用于设置训练过程的辅助函数：
        *   `get_loss_function`: 根据名称（如 "L1", "MSE", "KLDivergence"）获取损失函数实例。
        *   `get_optimizer`: 根据名称（如 "AdamW", "Adam"）获取优化器实例。
        *   `get_scheduler`: 根据名称（如 "OneCycleLR", "LinearWarmup", "ReduceLROnPlateau"）获取学习率调度器实例。

*   **`fine_tuning_loop.py`**:
    *   包含核心的训练和验证循环函数 `run_fine_tuning`。
    *   该函数负责：
        *   在每个 epoch 中迭代训练数据加载器。
        *   执行模型的前向传播和反向传播。
        *   计算损失并更新模型参数。
        *   在每个 epoch 结束后，在验证集上评估模型性能。
        *   根据验证集上的性能保存最佳模型。
        *   记录训练过程中的损失和学习率等指标。

*   **`run_deconvolution_finetuning.py`**:
    *   这是整个反卷积微调流程的主执行脚本。
    *   它整合了上述所有脚本的功能，按顺序执行数据准备、伪批量生成、预处理、模型创建、数据加载和微调训练。
    *   通过命令行参数接收各种配置，如文件路径、模型参数、训练超参数等。
    *   负责协调整个流程的运行。

## 如何运行

主要通过 `run_deconvolution_finetuning.py` 脚本来运行整个流程。你需要准备好单细胞数据 (`.h5ad` 格式) 和预训练的 scGPT 模型文件（包括模型权重 `best_model.pt`，配置文件 `args.json` 和词汇表 `vocab.json`）。

示例命令（需要替换为你的实际路径和参数）：

```bash
python run_deconvolution_finetuning.py \
    --adata_file_path /path/to/your/single_cell_data.h5ad \
    --model_dir_path /path/to/your/pretrained_scGPT_model_directory/ \
    --vocab_file_path /path/to/your/pretrained_scGPT_model_directory/vocab.json \
    --output_dir ./deconv_finetune_output \
    --cell_type_col "name_of_cell_type_column_in_adata_obs" \
    --n_pseudo_bulk_samples 2000 \
    --min_cells_per_sample 50 \
    --max_cells_per_sample 200 \
    --aggregation_method "sum" \
    --norm_target_sum 10000 \
    --n_bins 51 \
    --batch_size 32 \
    --n_epochs 50 \
    --learning_rate 1e-4 \
    --loss_type "L1" \
    --optimizer_type "AdamW" \
    --scheduler_type "OneCycleLR" \
    --weight_decay 1e-5 \
    --warmup_frac 0.1 \
    --freeze_scgpt_base \
    --seed 42
```

请根据你的具体需求调整上述参数。脚本会创建 `output_dir` 指定的目录，并在其中保存训练好的模型和训练历史记录。
