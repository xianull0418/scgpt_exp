# -*- coding: utf-8 -*-
# 这是一个主脚本文件

# 标准库导入
import sys # 导入sys以修改路径
sys.path.insert(0, "../TAPE") # 假设TAPE目录在上一级
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings
from typing import List, Tuple, Dict, Union, Optional

# 常见的ML/数据科学库导入
import torch
import scvi # 导入scvi
import anndata
import scanpy
import numpy as np
import pandas as pd # 导入pandas
from scipy.sparse import issparse # 导入issparse
import wandb # 导入wandb
import scipy.sparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab

# scGPT项目导入
import scgpt
from scgpt.model import TransformerModel
from TAPE.simulation import generate_simulated_data # 导入TAPE的模拟函数
from scgpt.tokenizer import GeneVocab
from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed, add_file_handler
from scgpt import logger

# 超参数默认值
hyperparameter_defaults = {
    "dataset_name": "PBMC10k_Deconvolution",
    "seed": 42,
    "save_dir": "./save/pbmc10k_deconvolution_finetune/",
    # 从finetune_integration.py复制的其他占位符参数
    "epochs": 10,
    "lr": 1e-4,
    "batch_size": 64,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_hvg": 0, # 0表示使用预选的HVG
    "n_bins": 51,
    "mask_value": -1, # 非数值掩码值，也是填充值
    "dropout": 0.2,
    "n_layers": 4,
    "n_heads": 4,
    "emb_size": 256,
    "hidden_size": 256,
    "load_model": "save/scGPT_bc",  # 预训练模型路径 (示例)
    "max_seq_len_hvgs": True, # 是否基于HGV数量设置max_seq_len (n_hvg + 1)
    "max_seq_len": 2001, # 用户指定的固定max_seq_len (仅在max_seq_len_hvgs为False时使用)
    "pad_token": "<pad>",  # padding token
    "special_tokens": ["<pad>", "<cls>", "<eoc>"],  # 特殊token列表
    "pad_value": -2,  # padding的值 (例如, 用于基因表达值)
    "append_cls_token": True, # 是否在序列前添加<cls>token
    "include_zero_gene": False, # tokenize时是否包含零表达基因
    "wandb_project": "scGPT_Deconvolution",
    "wandb_entity": "your_wandb_entity", # 替换为你的wandb实体
    "patience": 5, # 早停的耐心轮数
    "log_interval": 100, # 每隔多少批次记录一次日志
    "tape_sample_num": 5000,  # TAPE 生成的模拟样本数量
    "tape_generation_sparse": True,  # TAPE 是否生成稀疏细胞组分
    "n_hvg": 1200,  # 用于预处理的高变基因数量
    # "n_bins" is already defined, will keep its current value: 51
    # "batch_size" is already defined, will keep its current value: 64
    # "num_workers" is already defined, will keep its current value: 0
    "filter_gene_by_counts": 3, # 预处理：按计数过滤基因的阈值，或设为False禁用
    "normalize_total": 1e4, # 预处理：标准化后的总计数
    "log1p_after_norm": True, # 预处理：标准化后是否进行log1p转换
    "split_test_size": 0.1,  # 训练/验证集划分比例
    "num_dataloader_workers": 0,  # DataLoader使用的工作进程数 (0表示主进程)
    # 模型相关新增超参数
    "deconv_head_dims": [128, 64],  # 解卷积头的隐藏层维度列表
    "deconv_head_dropout": 0.1,  # 解卷积头的dropout率
    "freeze_transformer_weights": False, # 是否冻结预训练Transformer的权重
    "explicit_zero_prob": False, # 通常在微调解卷积任务中为False
    "use_batch_labels_model": False, # 解卷积任务中通常为False
    "domain_spec_batchnorm_model": False, # 解卷积任务中通常为False
    "mvc_task": False, # Masked Value Prediction任务 (解卷积微调中为False)
    "ecs_task": False, # Elastic Cell Similarity任务 (解卷积微调中为False)
    "dab_task": False, # Domain Adversarial training (解卷积微调中为False)
    "ecs_thres": 0.8, # ECS任务阈值 (即使ecs_task为False也定义)
    "fast_transformer": True, # 是否使用Flash Attention优化的Transformer
    "pre_norm": True, # 是否在Transformer层中使用Pre-Normalization
    # 损失函数、优化器和调度器相关超参数
    "loss_fn_type": "L1",  # 解卷积损失函数类型 ("L1", "MSE", "KLDiv")
    "optimizer_type": "AdamW",  # 优化器类型 ("AdamW", "Adam")
    "weight_decay": 1e-5,  # 优化器的权重衰减
    "scheduler_type": "StepLR",  # 学习率调度器类型 ("StepLR", "OneCycleLR")
    "schedule_ratio_stepLR": 0.9, # StepLR的gamma参数 (即每个step衰减的比例)
    "lr_scheduler_step_size": 1, # StepLR的step_size
    # 训练循环相关超参数
    # "epochs" is already defined with value 10, will update to 30.
    "log_interval_batches": 10,  # 每隔多少批次记录一次训练日志
    "amp_enabled": True,  # 是否启用自动混合精度 (AMP)
    "save_model_interval_epochs": 5, # 每隔多少轮保存一次模型检查点
    # "wandb_project" is already defined, will confirm value.
    "wandb_log_model": False, # 是否在wandb中记录模型文件 (可能较大)
}
# 更新已存在的超参数
hyperparameter_defaults["epochs"] = 30 # 更新训练轮数
hyperparameter_defaults["wandb_project"] = "scGPT_deconvolution" # 更新wandb项目名称


# 一个简单的配置类/字典
class SimpleConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

config = SimpleConfig(**hyperparameter_defaults)

# wandb 初始化 (如果使用)
# try:
#     wandb.init(
#         project=config.wandb_project,
#         entity=config.wandb_entity,
#         config=hyperparameter_defaults,
#         reinit=True, # 如果在同一个脚本中多次调用wandb.init，则需要此参数
#     )
#     wandb.run.name = f"deconv_{config.dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}"
#     wandb.save("*.pt") # 保存所有模型文件
#     wandb_run_id = wandb.run.id
# except Exception as e:
#     print(f"Wandb 初始化失败: {e}")
#     wandb = None


# 设置随机种子
set_seed(config.seed)

# 设置保存目录
save_dir = Path(config.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# 设置日志记录
logger.setLevel("INFO") # 设置日志级别
log_file = save_dir / "run.log"
add_file_handler(logger, log_file) # 将日志输出到文件

logger.info(f"PID: {os.getpid()}") # 记录进程ID
logger.info(f"配置: {config.__dict__}") # 记录配置信息
logger.info(f"保存目录: {save_dir}") # 记录保存目录

# ## 加载PBMC_10K数据
logger.info("开始加载 PBMC_10K 数据...")
adata = scvi.data.pbmc_dataset() # 加载数据
data_is_raw = True # pbmc_dataset 通常是原始计数
logger.info(f"成功加载 PBMC_10K 数据, 原始形状: {adata.shape}")

# ## 初始 AnnData 预处理
# 确保 'gene_symbols' 列存在并且设置为索引
if "gene_symbols" not in adata.var.columns and adata.var.index.name != "gene_symbols":
    logger.warning("adata.var 中缺少 'gene_symbols' 列或索引。将尝试使用当前索引作为基因名称。")
    adata.var["gene_name"] = adata.var.index.tolist()
elif "gene_symbols" in adata.var.columns and adata.var.index.name != "gene_symbols":
    logger.info("将 'gene_symbols' 列设置为 adata.var 的索引。")
    adata.var = adata.var.set_index("gene_symbols")
    adata.var["gene_name"] = adata.var.index.tolist()
elif adata.var.index.name == "gene_symbols":
    logger.info("adata.var 的索引已经是 'gene_symbols'。")
    adata.var["gene_name"] = adata.var.index.tolist()
else: # 兜底逻辑，如果 gene_symbols 是索引，但名字不是 "gene_symbols"
    logger.warning("adata.var 的索引可能已经是基因符号，但名称不是 'gene_symbols'。将使用当前索引作为基因名称。")
    adata.var["gene_name"] = adata.var.index.tolist()


# 检查 'str_labels' 列是否存在并创建 'celltype'
if "str_labels" in adata.obs.columns:
    logger.info("从 'str_labels' 创建 'celltype' 列。")
    adata.obs['celltype'] = adata.obs['str_labels'].astype('category')
else:
    logger.warning("adata.obs 中缺少 'str_labels' 列。无法创建 'celltype' 列。请检查数据源。")
    # 可以选择创建一个默认的 'celltype' 列或者抛出错误
    # adata.obs['celltype'] = 'Unknown' 

logger.info(f"数据预处理后, adata.var 示例: \n{adata.var.head()}")
logger.info(f"数据预处理后, adata.obs 示例: \n{adata.obs.head()}")

# ## 使用TAPE生成模拟bulk数据
logger.info("开始使用TAPE生成模拟bulk数据...")

# 为TAPE准备输入DataFrame
logger.info("为TAPE准备输入DataFrame...")
X_dense = adata.X.toarray() if issparse(adata.X) else adata.X # 确保数据是稠密的
# 使用 .astype(str) 来确保celltype是字符串，避免潜在的Categorical类型问题
df_for_tape = pd.DataFrame(X_dense, 
                           index=adata.obs["celltype"].astype(str), 
                           columns=adata.var["gene_name"].tolist())
df_for_tape["celltype"] = df_for_tape.index # 将细胞类型作为一列
df_for_tape.reset_index(drop=True, inplace=True) # 重置索引
logger.info(f"为TAPE生成的DataFrame形状: {df_for_tape.shape}")

# 调用TAPE的generate_simulated_data函数
logger.info(f"使用TAPE生成模拟数据，样本数: {config.tape_sample_num}, 是否稀疏: {config.tape_generation_sparse}")
try:
    simu_data = generate_simulated_data(
        sc_data=df_for_tape, 
        samplenum=config.tape_sample_num, 
        random_state=config.seed, 
        sparse=config.tape_generation_sparse
    )
    logger.info(f"TAPE模拟bulk数据 AnnData 形状: {simu_data.shape}")
    logger.info(f"TAPE模拟bulk数据细胞组分 .obs 形状: {simu_data.obs.shape}")

    # 存储细胞组分和类型
    cell_fractions = simu_data.obs.copy()  # 模拟的细胞组分
    cell_types = list(cell_fractions.columns)  # 细胞类型列表
    logger.info(f"从TAPE模拟数据中提取的细胞类型 ({len(cell_types)}个): {cell_types}")
    # logger.info(f"细胞组分示例 (前5个样本): \n{cell_fractions.head()}") # 可以选择记录一些组分示例

except Exception as e:
    logger.error(f"TAPE生成模拟数据时发生错误: {e}")
    logger.error(traceback.format_exc())
    simu_data = None # 错误发生时，将simu_data设为None
    cell_fractions = None
    cell_types = []

# ## 预处理TAPE生成的模拟bulk数据
if simu_data is not None:
    logger.info("开始预处理TAPE生成的模拟bulk数据...")
    preprocessor = Preprocessor(
        use_key="X",  # 使用 simu_data.X 作为原始数据
        filter_gene_by_counts=config.filter_gene_by_counts,
        filter_cell_by_counts=False,  # bulk数据中细胞即样本，不过滤
        normalize_total=config.normalize_total,
        result_normed_key="X_normed",
        log1p=config.log1p_after_norm,
        result_log1p_key="X_log1p",
        subset_hvg=config.n_hvg if config.n_hvg > 0 else False, # 如果n_hvg为0或负数，则禁用HVG
        hvg_flavor="seurat_v3", # HVG选择方法
        binning=config.n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(simu_data, batch_key=None) # bulk数据无batch_key
    logger.info(f"预处理后 simu_data 形状 (高变基因筛选后): {simu_data.shape}")

    # 准备用于模型输入的数据
    input_layer_key = "X_binned"  # 用于模型输入的层
    if input_layer_key not in simu_data.layers:
        logger.error(f"错误: 在simu_data.layers中找不到指定的input_layer_key '{input_layer_key}'。可用层: {list(simu_data.layers.keys())}")
        # 根据错误处理策略，可能需要退出或使用备用层
        all_counts = None # 或者抛出异常
    else:
        all_counts = simu_data.layers[input_layer_key].toarray() if issparse(simu_data.layers[input_layer_key]) else simu_data.layers[input_layer_key]
        logger.info(f"预处理后用于token化的基因表达矩阵形状: {all_counts.shape}")

    genes = simu_data.var_names.tolist() # 预处理后的基因列表
    logger.info(f"预处理后基因列表长度: {len(genes)}")
else:
    logger.warning("simu_data 为 None，跳过TAPE模拟数据的预处理步骤。")
    all_counts = None
    genes = []
tokenized_data = None # 在此初始化tokenized_data

# ## 加载词汇表并进行Tokenization
if simu_data is not None and genes is not None and len(genes) > 0 and config.load_model is not None:
    logger.info("开始加载词汇表并进行Tokenization...")

    # 确定 max_seq_len
    if config.max_seq_len_hvgs:
        max_seq_len = config.n_hvg + 1 if config.n_hvg > 0 else 2001 # n_hvg + <cls> token, 2001是一个备用值
        logger.info(f"基于HVG数量设置 max_seq_len 为: {max_seq_len} (n_hvg: {config.n_hvg} + 1 for <cls>)")
    else:
        max_seq_len = config.max_seq_len # 用户指定固定长度
        logger.info(f"使用用户指定的固定 max_seq_len: {max_seq_len}")

    # 加载词汇表
    logger.info(f"开始加载词汇表从预训练模型: {config.load_model}")
    model_dir = Path(config.load_model)
    vocab_file = model_dir / "vocab.json"
    vocab = None # 初始化vocab
    if not vocab_file.exists():
        logger.error(f"错误：词汇表文件 {vocab_file} 未找到。请检查 load_model路径。")
        # 此处可以添加更强的错误处理，例如 sys.exit()
    else:
        try:
            vocab = GeneVocab.from_file(vocab_file)
            logger.info(f"词汇表 {vocab_file} 加载成功。")
            for s_token in config.special_tokens:
                if s_token not in vocab:
                    vocab.append_token(s_token)
                    logger.info(f"特殊token '{s_token}' 已添加到词汇表。")
            logger.info(f"词汇表加载并处理完成，最终大小: {len(vocab)}")
        except Exception as e:
            logger.error(f"加载或处理词汇表 {vocab_file} 时发生错误: {e}")
            logger.error(traceback.format_exc())
            vocab = None # 确保vocab在出错时为None

    # Tokenize 数据 (仅当词汇表加载成功时)
    if vocab is not None:
        logger.info("开始对预处理后的bulk数据进行tokenization...")
        # 获取预处理后的基因在词汇表中的ID
        # 对于不在词汇表中的基因，使用pad_token的ID
        gene_ids = np.array([vocab[gene] if gene in vocab else vocab[config.pad_token] for gene in genes], dtype=int)
        
        # 检查是否有基因未在词汇表中找到 (除了pad_token之外)
        original_gene_indices_in_vocab = np.array([gene in vocab for gene in genes])
        num_genes_not_in_vocab = len(genes) - np.sum(original_gene_indices_in_vocab)
        if num_genes_not_in_vocab > 0:
            logger.warning(f"{num_genes_not_in_vocab} 个基因不在词汇表中，将被替换为 '{config.pad_token}'。")

        tokenized_data = tokenize_and_pad_batch(
            all_counts, # 预处理后的基因表达矩阵 (n_samples, n_genes)
            gene_ids,   # 基因ID列表 (n_genes,)
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value, # tokenized values的填充值
            append_cls=config.append_cls_token, # 是否添加<cls>
            include_zero_gene=config.include_zero_gene,
        )
        logger.info(f"Tokenized gene id 形状: {tokenized_data['genes'].shape}") # (n_samples, max_seq_len)
        logger.info(f"Tokenized values 形状: {tokenized_data['values'].shape}") # (n_samples, max_seq_len)
    else:
        logger.warning("词汇表加载失败或基因列表为空，跳过Tokenization。")
        tokenized_data = None # 确保在失败时为None
elif config.load_model is None:
    logger.warning("config.load_model 未指定，跳过词汇表加载和Tokenization。")
    tokenized_data = None
else:
    logger.warning("预处理数据 (simu_data 或 genes) 无效，跳过词汇表加载和Tokenization。")
    tokenized_data = None
train_loader, valid_loader = None, None # 初始化DataLoader

# ## 创建数据集和DataLoader
if tokenized_data is not None and cell_fractions is not None:
    logger.info("开始创建数据集和DataLoader...")

    # 将 cell_fractions 转换为 NumPy 数组
    logger.info("准备将 TAPE 生成的 cell_fractions 转换为 NumPy 数组...")
    if hasattr(cell_fractions, 'values') and isinstance(cell_fractions.values, np.ndarray):
        target_cell_fractions_np = cell_fractions.values
    elif isinstance(cell_fractions, pd.DataFrame) or isinstance(cell_fractions, pd.Series):
         target_cell_fractions_np = cell_fractions.to_numpy()
    elif isinstance(cell_fractions, np.ndarray):
        target_cell_fractions_np = cell_fractions
    else:
        logger.warning("cell_fractions 类型未知，尝试直接转换为NumPy数组。")
        target_cell_fractions_np = np.array(cell_fractions)
    logger.info(f"Cell fractions NumPy 数组形状: {target_cell_fractions_np.shape}")

    # 划分数据
    logger.info(f"开始划分数据为训练集和验证集，测试集比例: {config.split_test_size}")
    try:
        train_gene_ids, valid_gene_ids, \
        train_values, valid_values, \
        train_fractions, valid_fractions = train_test_split(
            tokenized_data['genes'],
            tokenized_data['values'],
            target_cell_fractions_np,
            test_size=config.split_test_size,
            random_state=config.seed
        )
        logger.info(f"训练集 gene_ids 形状: {train_gene_ids.shape}, values 形状: {train_values.shape}, fractions 形状: {train_fractions.shape}")
        logger.info(f"验证集 gene_ids 形状: {valid_gene_ids.shape}, values 形状: {valid_values.shape}, fractions 形状: {valid_fractions.shape}")

        # 定义 DeconvDataset 类
        class DeconvDataset(Dataset):
            # (这是一个用于细胞解卷积任务的数据集类)
            def __init__(self, gene_ids_data, values_data, fractions_data):
                self.gene_ids_data = torch.LongTensor(gene_ids_data)
                self.values_data = torch.FloatTensor(values_data)
                self.fractions_data = torch.FloatTensor(fractions_data)
                # (存储基因ID、表达值和细胞组分)
            def __len__(self):
                return self.gene_ids_data.shape[0]
                # (返回数据集中的样本数量)
            def __getitem__(self, idx):
                # (根据索引获取单个样本)
                return {
                    "gene_ids": self.gene_ids_data[idx],
                    "values": self.values_data[idx],
                    "cell_fractions": self.fractions_data[idx],
                }

        # 创建 Dataset 和 DataLoader
        logger.info("创建训练集和验证集的Dataset对象...")
        train_dataset = DeconvDataset(train_gene_ids, train_values, train_fractions)
        valid_dataset = DeconvDataset(valid_gene_ids, valid_values, valid_fractions)
        logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(valid_dataset)}")

        logger.info("创建训练集和验证集的DataLoader对象...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_dataloader_workers, # 使用新的超参数
            pin_memory=True,
            drop_last=False  # 是否丢弃最后一个不完整的批次
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_dataloader_workers, # 使用新的超参数
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"训练 DataLoader 批次数: {len(train_loader)}, 验证 DataLoader 批次数: {len(valid_loader)}")

    except Exception as e:
        logger.error(f"在数据划分或DataLoader创建过程中发生错误: {e}")
        logger.error(traceback.format_exc())
        train_loader, valid_loader = None, None # 确保在错误时重置
else:
    logger.warning("Tokenized data 或 cell_fractions 无效，跳过数据集和DataLoader的创建。")
    train_loader, valid_loader = None, None
model = None # 初始化模型变量

# ## 定义模型并加载预训练权重
# 确保 vocab 和 cell_types 已定义且有效 (vocab 在 tokenization 步骤中定义, cell_types 在 TAPE 模拟后定义)
if vocab is not None and cell_types is not None and len(cell_types) > 0 and config.load_model:
    logger.info("开始定义模型并加载预训练权重...")
    device = torch.device(config.device)
    logger.info(f"模型将使用设备: {device}")

    try:
        # 加载预训练模型的配置
        logger.info(f"加载预训练模型的配置从: {config.load_model}")
        model_dir_path = Path(config.load_model)
        model_config_file = model_dir_path / "args.json"
        
        if not model_config_file.exists():
            logger.error(f"错误: 预训练模型配置文件 {model_config_file} 未找到。请检查 load_model 路径。")
            # 可以考虑退出或抛出异常
        else:
            with open(model_config_file, "r") as f:
                model_configs_pretrain = json.load(f)
            logger.info(f"预训练模型配置: {model_configs_pretrain}")

            # 实例化基础 Transformer 模型
            logger.info("实例化基础 Transformer 模型...")
            base_model = TransformerModel(
                ntoken=len(vocab),
                d_model=model_configs_pretrain['embsize'], # 使用预训练模型的embsize
                nhead=model_configs_pretrain['nheads'],   # 使用预训练模型的nheads
                d_hid=model_configs_pretrain['d_hid'],     # 使用预训练模型的d_hid
                nlayers=model_configs_pretrain['nlayers'], # 使用预训练模型的nlayers
                vocab=vocab,
                dropout=config.dropout, # 使用当前配置的dropout
                pad_token=config.pad_token,
                pad_value=config.pad_value,
                do_mvc=config.mvc_task, # 当前任务设置为False
                do_dab=config.dab_task, # 当前任务设置为False
                use_batch_labels=config.use_batch_labels_model, # 当前任务设置为False
                num_batch_labels=None, # 因为use_batch_labels为False
                domain_spec_batchnorm=config.domain_spec_batchnorm_model, # 当前任务设置为False
                n_input_bins=config.n_bins, # 使用当前配置的n_bins
                ecs_threshold=0.0 if not config.ecs_task else config.ecs_thres, # 当前任务ecs_task为False
                explicit_zero_prob=config.explicit_zero_prob, # 当前任务设置为False
                use_fast_transformer=config.fast_transformer,
                pre_norm=config.pre_norm
            )

            # 添加解卷积头
            logger.info("为模型添加解卷积头...")
            n_cell_types = len(cell_types)
            head_layers = []
            # 使用预训练模型的 embsize 作为解卷积头的输入维度
            input_dim = model_configs_pretrain['embsize'] 
            for hidden_dim in config.deconv_head_dims:
                head_layers.append(nn.Linear(input_dim, hidden_dim))
                head_layers.append(nn.ReLU())
                head_layers.append(nn.Dropout(config.deconv_head_dropout))
                input_dim = hidden_dim
            head_layers.append(nn.Linear(input_dim, n_cell_types))
            head_layers.append(nn.Softmax(dim=1)) # Softmax确保输出为概率分布
            base_model.deconv_head = nn.Sequential(*head_layers)
            logger.info(f"解卷积头结构: {base_model.deconv_head}")

            # 加载预训练权重
            logger.info("加载预训练模型权重...")
            model_file = model_dir_path / "best_model.pt"
            if not model_file.exists():
                logger.error(f"错误: 预训练模型权重文件 {model_file} 未找到。")
            else:
                try:
                    pretrained_dict = torch.load(model_file, map_location=torch.device('cpu'))
                    model_dict = base_model.state_dict()
                    
                    # 过滤掉不匹配的键 (特别是新的解卷积头和可能不匹配的层)
                    pretrained_dict_filtered = {
                        k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape
                    }
                    model_dict.update(pretrained_dict_filtered)
                    
                    missing_keys, unexpected_keys = base_model.load_state_dict(model_dict, strict=False)
                    logger.info(f"加载模型权重完成。缺失的键: {missing_keys}, 未预期的键: {unexpected_keys}")
                    logger.info(f"成功加载了 {len(pretrained_dict_filtered)} 个参数层（在预训练模型和当前模型中均匹配）。")

                    # 冻结预训练权重 (可选)
                    if config.freeze_transformer_weights:
                        logger.info("冻结预训练Transformer的权重，仅训练解卷积头...")
                        for name, param in base_model.named_parameters():
                            if 'deconv_head' not in name: # 不冻结新加的解卷积头
                                param.requires_grad = False
                        logger.info("权重冻结完成。")
                    
                    model = base_model # 将配置好的模型赋值给全局model变量
                    model.to(device)
                    logger.info(f"模型已成功定义并移动到设备: {device}")

                    # 记录可训练参数数量
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    if total_params > 0:
                        logger.info(f"模型总参数: {total_params}, 可训练参数: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
                    else:
                        logger.warning("模型总参数为0，请检查模型定义。")

                except Exception as e:
                    logger.error(f"加载模型权重或处理模型时发生错误: {e}")
                    logger.error(traceback.format_exc())
                    model = None # 确保错误时model为None
    except Exception as e:
        logger.error(f"模型定义和加载过程中发生顶层错误: {e}")
        logger.error(traceback.format_exc())
        model = None # 确保错误时model为None
elif not config.load_model:
    logger.warning("config.load_model 未指定，跳过模型加载。")
    model = None
else:
    logger.warning("词汇表(vocab)或细胞类型(cell_types)无效，跳过模型定义和加载。")
    model = None
deconvolution_criterion, optimizer, scheduler = None, None, None # 初始化训练组件

# ## 定义损失函数、优化器和学习率调度器
if model is not None: # 只有在模型成功定义后才进行
    logger.info("开始定义损失函数、优化器和学习率调度器...")
    try:
        # 定义损失函数
        logger.info(f"定义解卷积任务的损失函数: {config.loss_fn_type}")
        if config.loss_fn_type == "L1":
            deconvolution_criterion = nn.L1Loss()
        elif config.loss_fn_type == "MSE":
            deconvolution_criterion = nn.MSELoss()
        elif config.loss_fn_type == "KLDiv":
            # 注意: KLDivLoss期望log-prob作为输入. 如果模型输出softmax, 需要先取log.
            # 或者模型最后一层不使用Softmax, 而是LogSoftmax.
            deconvolution_criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            logger.error(f"不支持的损失函数类型: {config.loss_fn_type}. 将使用L1 Loss作为默认.")
            deconvolution_criterion = nn.L1Loss()
        logger.info(f"损失函数 {type(deconvolution_criterion).__name__} 已定义.")

        # 定义优化器
        logger.info(f"定义优化器: {config.optimizer_type}")
        params_to_optimize = model.deconv_head.parameters() if config.freeze_transformer_weights else model.parameters()
        
        # 确保 params_to_optimize 不是一个空的迭代器
        # 将其转换为列表以检查是否为空或记录参数
        list_params_to_optimize = list(params_to_optimize)
        if not list_params_to_optimize:
            logger.warning("没有参数可供优化。如果冻结了Transformer权重，请确保deconv_head已正确定义且未被冻结。")
            optimizer = None # 无法创建优化器
        else:
            if config.optimizer_type == "AdamW":
                optimizer = torch.optim.AdamW(list_params_to_optimize, lr=config.lr, weight_decay=config.weight_decay)
            elif config.optimizer_type == "Adam":
                optimizer = torch.optim.Adam(list_params_to_optimize, lr=config.lr, weight_decay=config.weight_decay)
            else:
                logger.error(f"不支持的优化器类型: {config.optimizer_type}. 将使用AdamW作为默认.")
                optimizer = torch.optim.AdamW(list_params_to_optimize, lr=config.lr, weight_decay=config.weight_decay)
            
            if optimizer: # 只有在优化器成功创建后才记录
                # 重新获取可训练参数的数量，因为list_params_to_optimize可能包含未设置requires_grad=True的参数
                num_actual_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"优化器 {type(optimizer).__name__} 已定义. 实际可训练参数数量: {num_actual_trainable_params}")


        # 定义学习率调度器 (只有在优化器成功定义后)
        if optimizer:
            logger.info(f"定义学习率调度器: {config.scheduler_type}")
            if config.scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.schedule_ratio_stepLR)
            elif config.scheduler_type == "OneCycleLR":
                # OneCycleLR 需要 total_steps, 通常是 epochs * len(train_loader)
                # 这个值在此时可能还不可用, 可能需要后续在训练循环前定义或传递epochs和train_loader
                logger.warning("OneCycleLR调度器选择, 但total_steps未在此处设置。请确保在训练开始前正确配置。")
                # 占位符, 实际初始化应推迟或提供 total_steps
                scheduler = None # Or a StepLR as fallback for now
            else:
                logger.error(f"不支持的调度器类型: {config.scheduler_type}. 将不使用调度器或使用StepLR作为默认.")
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.schedule_ratio_stepLR)
            
            if scheduler:
                logger.info(f"学习率调度器 {type(scheduler).__name__} 已定义.")
            else:
                logger.warning("学习率调度器未成功定义 (例如，OneCycleLR需要后续设置).")
        else:
            logger.warning("优化器未定义，跳过学习率调度器的定义。")
            scheduler = None

    except Exception as e:
        logger.error(f"定义损失函数、优化器或调度器时发生错误: {e}")
        logger.error(traceback.format_exc())
        deconvolution_criterion, optimizer, scheduler = None, None, None # 确保错误时重置
else:
    logger.warning("模型未定义，跳过损失函数、优化器和学习率调度器的定义。")
    deconvolution_criterion, optimizer, scheduler = None, None, None

# ## 定义训练和评估函数

def train_epoch_deconv(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scaler: torch.cuda.amp.GradScaler, 
    device: torch.device, 
    epoch_num: int, 
    config: SimpleConfig, 
    vocab: GeneVocab,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # 调度器现在传入，用于OneCycleLR
):
    # (训练模型一个轮次)
    model.train() # 设置模型为训练模式
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        gene_ids = batch_data["gene_ids"].to(device)
        values = batch_data["values"].to(device)
        target_fractions = batch_data["cell_fractions"].to(device)
        
        # 创建padding mask
        src_key_padding_mask = gene_ids.eq(vocab[config.pad_token])

        with torch.cuda.amp.autocast(enabled=config.amp_enabled):
            output_dict = model(src=gene_ids, values=values, src_key_padding_mask=src_key_padding_mask)
            
            # 提取CLS嵌入
            if "cls_output" in output_dict:
                cls_embedding = output_dict["cls_output"]
            elif "encoder_output" in output_dict:
                cls_embedding = output_dict["encoder_output"][:, 0, :]  # CLS token 通常在第一个位置
            else:
                logger.error(f"Epoch {epoch_num}, Batch {batch_idx}: 无法从模型输出中找到CLS嵌入。")
                # 考虑是否跳过此批次或抛出错误
                continue 
            
            predicted_fractions = model.deconv_head(cls_embedding) # 通过解卷积头
            loss = criterion(predicted_fractions, target_fractions)

        optimizer.zero_grad()
        scaler.scale(loss).backward() # AMP: loss缩放和反向传播
        scaler.step(optimizer) # AMP: 更新优化器
        scaler.update() # AMP: 更新scaler

        total_loss += loss.item()

        if config.scheduler_type == "OneCycleLR" and scheduler is not None:
            scheduler.step() # OneCycleLR在每个批次后更新

        if batch_idx % config.log_interval_batches == 0:
            elapsed_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch_num} | Batch {batch_idx}/{len(train_loader)} | "
                f"训练损失: {loss.item():.4f} | LR: {current_lr:.6e} | "
                f"耗时: {elapsed_time:.2f}s"
            )
            start_time = time.time() # 重置批次计时器

    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


def evaluate_deconv(
    model: nn.Module, 
    valid_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    epoch_num: int, # 当前轮次 (用于日志)
    config: SimpleConfig, 
    vocab: GeneVocab
):
    # (评估模型在验证集上的性能)
    model.eval() # 设置模型为评估模式
    total_val_loss = 0.0
    
    with torch.no_grad(): # 评估时不需要计算梯度
        for batch_idx, batch_data in enumerate(valid_loader):
            gene_ids = batch_data["gene_ids"].to(device)
            values = batch_data["values"].to(device)
            target_fractions = batch_data["cell_fractions"].to(device)

            src_key_padding_mask = gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp_enabled): # AMP autocast保持一致性
                output_dict = model(src=gene_ids, values=values, src_key_padding_mask=src_key_padding_mask)
                
                if "cls_output" in output_dict:
                    cls_embedding = output_dict["cls_output"]
                elif "encoder_output" in output_dict:
                    cls_embedding = output_dict["encoder_output"][:, 0, :]
                else:
                    logger.error(f"Epoch {epoch_num} (Eval), Batch {batch_idx}: 无法找到CLS嵌入。")
                    continue

                predicted_fractions = model.deconv_head(cls_embedding)
                loss = criterion(predicted_fractions, target_fractions)
            
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(valid_loader)
    return avg_val_loss


# ## 开始训练和评估循环
if __name__ == "__main__":
    logger.info("脚本开始执行 (主模块 - 训练循环).")
    
    # 检查所有关键数据准备和模型设置步骤是否成功
    data_preparation_components = [
        adata, simu_data, all_counts, tokenized_data, train_loader, valid_loader, 
        cell_fractions, model, deconvolution_criterion, optimizer, vocab, cell_types
    ] # scheduler 可以是 None，单独处理

    if not all(item is not None for item in data_preparation_components):
        logger.error("一个或多个关键组件 (数据, 模型, 损失函数, 优化器, 词汇表, 细胞类型) 未成功初始化。无法开始训练。")
        sys.exit(1) # 严重错误，退出

    try:
        # Wandb 初始化
        if config.wandb_project:
            try:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity, # 可能为None，wandb会使用默认实体
                    config=config.__dict__, # 记录所有配置
                    name=f"deconv_{config.dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}",
                    reinit=True,
                )
                logger.info("Wandb 初始化成功。")
            except Exception as e:
                logger.error(f"Wandb 初始化失败: {e}")
                # 不设置wandb为None，因为wandb.log等调用会自动处理未初始化的运行
        
        # AMP GradScaler
        scaler = torch.cuda.amp.GradScaler(enabled=config.amp_enabled)
        logger.info(f"自动混合精度 (AMP) {'启用' if config.amp_enabled else '禁用'}.")

        # 如果使用OneCycleLR，在这里正确初始化
        if config.scheduler_type == "OneCycleLR" and optimizer is not None:
            if train_loader is not None and len(train_loader) > 0:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=config.lr, 
                    epochs=config.epochs, 
                    steps_per_epoch=len(train_loader)
                )
                logger.info(f"OneCycleLR 调度器已正确初始化. 总步数: {config.epochs * len(train_loader)}")
            else:
                logger.error("train_loader 为空或未定义，无法初始化 OneCycleLR。")
                scheduler = None # 保持 scheduler 为 None
        
        best_val_loss = float('inf')
        best_model_epoch = 0
        logger.info(f"开始训练，共 {config.epochs} 轮。")

        for epoch in range(1, config.epochs + 1):
            epoch_start_time = time.time()
            
            train_loss = train_epoch_deconv(
                model, train_loader, deconvolution_criterion, optimizer, scaler, 
                torch.device(config.device), epoch, config, vocab,
                scheduler if config.scheduler_type == "OneCycleLR" else None # 仅在OneCycleLR时传入并用于批次更新
            )
            
            val_loss = evaluate_deconv(
                model, valid_loader, deconvolution_criterion, 
                torch.device(config.device), epoch, config, vocab
            )
            
            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch {epoch}/{config.epochs} | 训练损失: {train_loss:.4f} | "
                f"验证损失: {val_loss:.4f} | LR: {current_lr:.6e} | 耗时: {epoch_duration:.2f}s"
            )

            if config.wandb_project:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch_duration_seconds": epoch_duration,
                }, step=epoch)

            # StepLR调度器在每个epoch后更新 (如果不是OneCycleLR)
            if config.scheduler_type == "StepLR" and scheduler is not None:
                scheduler.step()

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_epoch = epoch
                if save_dir:
                    best_model_path = save_dir / "best_model.pt"
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Epoch {epoch}: 新的最佳验证损失 {best_val_loss:.4f}。模型已保存至 {best_model_path}")
            
            # 定期保存模型检查点
            if epoch % config.save_model_interval_epochs == 0 and save_dir:
                checkpoint_path = save_dir / f"model_epoch_{epoch}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Epoch {epoch}: 模型检查点已保存至 {checkpoint_path}")

        logger.info(f"训练完成。最佳验证损失: {best_val_loss:.4f} (在第 {best_model_epoch} 轮)")
        
        if config.wandb_project and config.wandb_log_model and save_dir:
            # wandb.save(str(save_dir / "best_model.pt")) # deprecated
            # wandb.save(str(save_dir / "*.pt")) # deprecated
            # 使用 Artifacts
            # best_model_artifact = wandb.Artifact(f"{config.dataset_name}-best_model", type="model")
            # best_model_artifact.add_file(str(save_dir / "best_model.pt"))
            # wandb.log_artifact(best_model_artifact)
            logger.info("Wandb模型记录已配置 (wandb_log_model=True)，但具体实现 (如Artifacts) 可能需要根据wandb版本调整。")


    except KeyboardInterrupt:
        logger.warning("训练被用户手动中断。")
    except Exception as e:
        logger.error(f"训练过程中发生严重错误: {e}")
        logger.error(traceback.format_exc())
    finally:
        if config.wandb_project and wandb.run: # 检查 wandb.run 是否存在
            wandb.finish()
            logger.info("Wandb已结束。")
        logger.info("脚本执行完毕 (主模块 - 训练循环).")
else:
    logger.info("此脚本未作为主模块执行，训练循环未启动。")
    # 检查所有关键数据准备和模型设置步骤是否成功 (用于非主模块执行时的调试)
    data_preparation_successful = all(
        item is not None for item in 
        [adata, simu_data, all_counts, tokenized_data, train_loader, valid_loader, cell_fractions, model,
         deconvolution_criterion, optimizer, vocab, cell_types] # scheduler可以为None
    )
    if not data_preparation_successful:
        logger.warning("一个或多个关键组件未成功初始化。如果打算运行训练，请作为主模块执行此脚本。")

# pass # 移除之前脚本末尾多余的pass
