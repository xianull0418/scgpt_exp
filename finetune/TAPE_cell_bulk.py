#!/usr/bin/env python
# coding: utf-8

# # Fine-tunning on cell type deconvolution
# 1. 导入环境，这里使用scgpt_py3.9
# 2. 模型超参数设置
# 3. 数据处理
# 4. 导入预训练的模型
# 5. 确定损失函数，开始微调

# In[3]:


import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')




# In[ ]:


# 使用TAPE生成模拟数据
sample_num = 5000  # 生成的样本数量
sparse = True      # 是否生成稀疏细胞组成

# TAPE的函数对AnnData对象处理时有问题，我们需要预处理数据
# 先确保数据格式正确
print("预处理AnnData对象，准备生成模拟数据...")

# 将AnnData转换为DataFrame格式，TAPE可以处理
# 创建带有细胞类型的DataFrame
import pandas as pd
import numpy as np

# 确保数据是密集格式
X_dense = adata.X.toarray() if issparse(adata.X) else adata.X

# 创建数据框
df = pd.DataFrame(X_dense, index=adata.obs["CellType"], columns=adata.var_names)
df["celltype"] = df.index
df.reset_index(drop=True, inplace=True)

print(f"转换后的DataFrame形状: {df.shape}")
print(f"细胞类型列值计数:\n{df['celltype'].value_counts()}")

# 使用DataFrame调用generate_simulated_data函数
print(f"使用TAPE生成模拟数据，样本数: {sample_num}")
simu_data = generate_simulated_data(
    sc_data=df,  # 使用预处理后的DataFrame
    samplenum=sample_num,
    random_state=hyperparameter_defaults["seed"],
    sparse=sparse
)

# 显示模拟数据基本信息
print(f"模拟数据形状: {simu_data.shape}")
print(f"模拟数据细胞比例形状: {simu_data.obs.shape}")

# 显示前几个样本的细胞组成比例
print("\n前5个样本的细胞组成比例:")
display(simu_data.obs.head())

# 保存细胞类型信息
cell_fractions = simu_data.obs
cell_types = list(cell_fractions.columns)
print(f"\n细胞类型列表 ({len(cell_types)}个):")
print(cell_types)


# In[4]:


hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model="/ai/home/jcw/scGPT_new/pretrain/save/cellxgene_census_blood-May13-00-17-2025",
    mask_ratio=0.0,
    epochs=15,
    n_bins=51,
    MVC=False,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=5e-4,
    batch_size=64,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.1,
    schedule_ratio=0.9,
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,
    DSBN=False,
    weight_decay=1e-5,  # 新增：权重衰减
    reconstruction_weight=1.0,  # 新增：重构损失权重
    prediction_weight=1.0,  # 新增：预测损失权重
)


# # 2. 数据处理

# In[5]:


# 导入TAPE模块
import sys
import random
sys.path.append("../TAPE")
from TAPE.simulation import generate_simulated_data  # 修正导入路径，从TAPE而不是TAPE.TAPE导入

# 设置随机种子
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

# 设置随机种子
set_seed(hyperparameter_defaults["seed"])

# 指定数据路径
sc_data_path = "/ai/home/jcw/scGPT_new/finetune/data/ TAPE/pbmc3k_raw.h5ad"


# In[13]:


print(f"读取单细胞数据: {sc_data_path}")
adata = sc.read_h5ad(sc_data_path)
print(f"单细胞数据: {adata}")
# 显示数据基本信息


# In[21]:


# 使用TAPE生成模拟数据
sample_num = 5000  # 生成的样本数量
sparse = True      # 是否生成稀疏细胞组成

# TAPE的函数对AnnData对象处理时有问题，我们需要预处理数据
# 先确保数据格式正确
print("预处理AnnData对象，准备生成模拟数据...")

# 将AnnData转换为DataFrame格式，TAPE可以处理
# 创建带有细胞类型的DataFrame
import pandas as pd
import numpy as np

# 确保数据是密集格式
X_dense = adata.X.toarray() if issparse(adata.X) else adata.X

# 创建数据框
df = pd.DataFrame(X_dense, index=adata.obs["CellType"], columns=adata.var_names)
df["celltype"] = df.index
df.reset_index(drop=True, inplace=True)

print(f"转换后的DataFrame形状: {df.shape}")
print(f"细胞类型列值计数:\n{df['celltype'].value_counts()}")

# 使用DataFrame调用generate_simulated_data函数
print(f"使用TAPE生成模拟数据，样本数: {sample_num}")
simu_data = generate_simulated_data(
    sc_data=df,  # 使用预处理后的DataFrame
    samplenum=sample_num,
    random_state=hyperparameter_defaults["seed"],
    sparse=sparse
)

# 显示模拟数据基本信息
print(f"模拟数据形状: {simu_data.shape}")
print(f"模拟数据细胞比例形状: {simu_data.obs.shape}")

# 显示前几个样本的细胞组成比例
print("\n前5个样本的细胞组成比例:")
display(simu_data.obs.head())

# 保存细胞类型信息
cell_fractions = simu_data.obs
cell_types = list(cell_fractions.columns)
print(f"\n细胞类型列表 ({len(cell_types)}个):")
print(cell_types)


# In[25]:


# 查看TransformerModel类的参数
import inspect
from scgpt.model import TransformerModel

# 打印TransformerModel的签名
signature = inspect.signature(TransformerModel.__init__)
print("TransformerModel初始化参数:")
for param_name, param in signature.parameters.items():
    if param_name != 'self':
        print(f"- {param_name}: {param.default if param.default != inspect.Parameter.empty else '必需'}")


# In[26]:


# 定义特殊token和值
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = -2

# 设置模型参数
config = hyperparameter_defaults

# 获取预训练模型路径
pretrained_model_path = config["load_model"]
print(f"加载预训练模型: {pretrained_model_path}")

# 检查预训练模型文件
model_dir = Path(pretrained_model_path)
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

# 检查文件是否存在
for file_path in [model_config_file, model_file, vocab_file]:
    if not file_path.exists():
        print(f"警告: 文件 {file_path} 不存在!")
    else:
        print(f"文件存在: {file_path}")

# 首先加载预训练模型的词汇表
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# 加载预训练模型的词汇表
vocab = GeneVocab.from_file(vocab_file)
print(f"加载词汇表，大小: {len(vocab)}")

# 确保特殊token在词汇表中
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
        print(f"添加特殊token: {s}")

# 加载模型配置
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
print("模型配置:")
print(json.dumps(model_configs, indent=2))

# 获取模型参数
embsize = model_configs.get("embsize", config["layer_size"])
nlayers = model_configs.get("nlayers", config["nlayers"])
nhead = model_configs.get("nheads", config["nhead"])
dropout = model_configs.get("dropout", config["dropout"])

# 创建模型并加载预训练权重，使用正确的参数
model = TransformerModel(
    ntoken=len(vocab),
    d_model=embsize,
    nhead=nhead,
    d_hid=4 * embsize,
    nlayers=nlayers,
    nlayers_cls=3,  # 使用默认值
    n_cls=1,        # 默认值
    vocab=vocab,    # 传递词汇表
    dropout=dropout,
    pad_token=pad_token,  # 使用正确的参数名
    pad_value=pad_value,
    do_mvc=config["MVC"],
    do_dab=True if config["dab_weight"] > 0 else False,
    use_batch_labels=False,
    domain_spec_batchnorm=config["DSBN"],
    input_emb_style="continuous",
    n_input_bins=config["n_bins"],
    cell_emb_style="cls",
    use_fast_transformer=config["fast_transformer"],
    pre_norm=config["pre_norm"],
)

# 添加自定义的解卷积头
model.deconv_head = nn.Sequential(
    nn.Linear(embsize, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, len(cell_types)),
    nn.Softmax(dim=1)
)

# 加载预训练模型权重
if config["freeze"]:
    print("冻结预训练模型参数，只训练新添加的层...")
    # 冻结预训练模型参数
    for name, param in model.named_parameters():
        if 'deconv_head' not in name:  # 不冻结解卷积头
            param.requires_grad = False
else:
    print("加载预训练权重，允许所有参数微调...")

# 加载预训练权重
pretrained_dict = torch.load(model_file, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model_dict = model.state_dict()

# 过滤掉不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)

print(f"成功加载了 {len(pretrained_dict)}/{len(model_dict)} 层预训练权重")

# 移动模型到GPU(如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"模型已加载到设备: {device}")

print("完成模型加载")

# 打印模型结构摘要
print("模型结构:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}, 可训练参数数量: {trainable_params}")
print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")


# In[ ]:


# 使用TAPE生成模拟数据
sample_num = 5000  # 生成的样本数量
sparse = True      # 是否生成稀疏细胞组成

# TAPE的函数对AnnData对象处理时有问题，我们需要预处理数据
# 先确保数据格式正确
print("预处理AnnData对象，准备生成模拟数据...")

# 将AnnData转换为DataFrame格式，TAPE可以处理
# 创建带有细胞类型的DataFrame
import pandas as pd
import numpy as np

# 确保数据是密集格式
X_dense = adata.X.toarray() if issparse(adata.X) else adata.X

# 创建数据框
df = pd.DataFrame(X_dense, index=adata.obs["CellType"], columns=adata.var_names)
df["celltype"] = df.index
df.reset_index(drop=True, inplace=True)

print(f"转换后的DataFrame形状: {df.shape}")
print(f"细胞类型列值计数:\n{df['celltype'].value_counts()}")

# 使用DataFrame调用generate_simulated_data函数
print(f"使用TAPE生成模拟数据，样本数: {sample_num}")
simu_data = generate_simulated_data(
    sc_data=df,  # 使用预处理后的DataFrame
    samplenum=sample_num,
    random_state=hyperparameter_defaults["seed"],
    sparse=sparse
)

# 显示模拟数据基本信息
print(f"模拟数据形状: {simu_data.shape}")
print(f"模拟数据细胞比例形状: {simu_data.obs.shape}")

# 显示前几个样本的细胞组成比例
print("\n前5个样本的细胞组成比例:")
display(simu_data.obs.head())

# 保存细胞类型信息
cell_fractions = simu_data.obs
cell_types = list(cell_fractions.columns)
print(f"\n细胞类型列表 ({len(cell_types)}个):")
print(cell_types)


# In[29]:


# 创建预处理器
n_hvg = 1200  # 高变基因数量
n_bins = hyperparameter_defaults["n_bins"]  # 分箱数量

preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=3,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=True,
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,
    hvg_flavor="seurat_v3",
    binning=n_bins,
    result_binned_key="X_binned",
)

# 预处理模拟数据
print("预处理模拟数据...")
preprocessor(simu_data)

# 检查处理后的数据
print(f"预处理后的数据层: {list(simu_data.layers.keys())}")
print(f"选择的高变基因数量: {simu_data.shape[1]}")

# 准备输入数据
input_layer_key = "X_binned"
all_counts = (
    simu_data.layers[input_layer_key].toarray()
    if issparse(simu_data.layers[input_layer_key])
    else simu_data.layers[input_layer_key]
)
genes = simu_data.var_names.tolist()

# 准备标签（细胞组成比例）
fractions = cell_fractions.values  # shape: (n_samples, n_cell_types)

# 显示数据形状
print(f"基因表达矩阵形状: {all_counts.shape}")
print(f"细胞比例矩阵形状: {fractions.shape}")


# In[30]:


# 数据集拆分参数
test_size = 0.1  # 验证集比例
random_state = hyperparameter_defaults["seed"]

# 拆分训练集和验证集
(
    train_data,
    valid_data,
    train_fractions,
    valid_fractions,
) = train_test_split(
    all_counts, fractions, test_size=test_size, random_state=random_state
)

print(f"训练集形状: {train_data.shape}")
print(f"验证集形状: {valid_data.shape}")
print(f"训练集标签形状: {train_fractions.shape}")
print(f"验证集标签形状: {valid_fractions.shape}")


# In[32]:


# Tokenizing 参数
max_seq_len = n_hvg + 1  # 最大序列长度 (高变基因数量+1个cls token)

# 重用已加载的词汇表(vocab)
# GeneVocab 对象没有 unk_index 属性，但有 get_default_index() 方法
gene_ids = np.array([vocab[gene] if gene in vocab else vocab.get_default_index() for gene in genes], dtype=int)

print(f"词汇表大小: {len(vocab)}")
print(f"基因ID数量: {len(gene_ids)}")

# Tokenize数据
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # 在开始添加<cls>标记
    include_zero_gene=hyperparameter_defaults["include_zero_gene"],
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=hyperparameter_defaults["include_zero_gene"],
)

# 查看tokenize后的数据形状
print(f"训练集tokenized gene_ids形状: {tokenized_train['genes'].shape}")
print(f"训练集tokenized values形状: {tokenized_train['values'].shape}")
print(f"验证集tokenized gene_ids形状: {tokenized_valid['genes'].shape}")
print(f"验证集tokenized values形状: {tokenized_valid['values'].shape}")

# 准备最终数据和创建数据加载器
tensor_fractions_train = torch.tensor(train_fractions, dtype=torch.float32)
tensor_fractions_valid = torch.tensor(valid_fractions, dtype=torch.float32)

# 准备模型输入数据
train_data_pt = {
    "gene_ids": tokenized_train["genes"],
    "values": tokenized_train["values"],
    "target_values": tokenized_train["values"],  # 对于重构任务
    "cell_fractions": tensor_fractions_train,    # 细胞组成比例作为目标
}
valid_data_pt = {
    "gene_ids": tokenized_valid["genes"],
    "values": tokenized_valid["values"],
    "target_values": tokenized_valid["values"],  # 对于重构任务
    "cell_fractions": tensor_fractions_valid,    # 细胞组成比例作为目标
}

# 定义数据集类
class DeconvDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# 创建数据加载器
batch_size = hyperparameter_defaults["batch_size"]

train_dataset = DeconvDataset(train_data_pt)
valid_dataset = DeconvDataset(valid_data_pt)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=True,
)

print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(valid_loader)}")


# In[34]:


# 定义解卷积损失函数
def deconv_loss_fn(pred, target, reconstruction=None, input_values=None):
    """
    组合损失函数用于细胞解卷积任务
    
    参数:
    - pred: 预测的细胞比例
    - target: 真实的细胞比例
    - reconstruction: 基因表达重构值(可选)
    - input_values: 输入的基因表达值(可选)
    
    返回:
    - 总损失和各部分损失的字典
    """
    # 细胞比例预测损失 (使用L1损失)
    pred_loss = F.l1_loss(pred, target)
    
    loss_dict = {"pred_loss": pred_loss}
    total_loss = hyperparameter_defaults["prediction_weight"] * pred_loss
    
    # 如果提供了重构信息，计算重构损失
    if reconstruction is not None and input_values is not None:
        # 排除padding值
        mask = (input_values != pad_value)
        masked_recon_loss = F.mse_loss(
            reconstruction[mask], 
            input_values[mask]
        )
        loss_dict["recon_loss"] = masked_recon_loss
        total_loss += hyperparameter_defaults["reconstruction_weight"] * masked_recon_loss
    
    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict

# 定义优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=hyperparameter_defaults["lr"],
    weight_decay=hyperparameter_defaults["weight_decay"],
)

# 学习率调度器
total_steps = len(train_loader) * hyperparameter_defaults["epochs"]
warmup_steps = int(0.1 * total_steps)  # 预热10%的步数
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=hyperparameter_defaults["lr"],
    total_steps=total_steps,
    pct_start=hyperparameter_defaults["schedule_ratio"],
    anneal_strategy='cos',
    div_factor=25.0,
)

print(f"总训练步数: {total_steps}")
print(f"学习率预热步数: {warmup_steps}")
print(f"最大学习率: {hyperparameter_defaults['lr']}")


# In[51]:


# 修改train_epoch函数中处理模型输出的部分
def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_recon_loss = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        try:
            # 准备数据
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            target_values = batch["target_values"].to(device)
            cell_fractions = batch["cell_fractions"].to(device)
            
            optimizer.zero_grad()
            
            # 创建padding掩码
            src_key_padding_mask = (gene_ids == vocab[pad_token]).bool()
            
            # 使用混合精度
            with torch.cuda.amp.autocast():
                # 前向传播
                output_dict = model(
                    src=gene_ids,
                    values=values,
                    src_key_padding_mask=src_key_padding_mask,
                )
                
                # 处理输出 - 确保获取正确的张量
                if isinstance(output_dict, dict):
                    # 尝试获取不同的输出键
                    output = output_dict.get("mlm_output", 
                             output_dict.get("decoded", 
                             output_dict.get("last_hidden_state", None)))
                    
                    if output is None and len(output_dict) > 0:
                        # 如果没找到预期的键，使用第一个可用的值
                        output = list(output_dict.values())[0]
                else:
                    # 如果不是字典，直接使用
                    output = output_dict
                
                # 添加debug信息
                print(f"输出类型: {type(output)}, 形状: {output.shape if torch.is_tensor(output) else 'not a tensor'}")
                
                # 根据输出维度正确获取表示
                if output is not None and torch.is_tensor(output):
                    if output.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                        cls_output = output[:, 0, :]  # 第一个token是[CLS]
                    elif output.dim() == 2:  # [batch_size, hidden_dim]
                        cls_output = output  # 已经是所需表示
                    else:
                        raise ValueError(f"Unexpected output dimension: {output.dim()}")
                        
                    # 预测细胞比例
                    pred_fractions = model.deconv_head(cls_output)
                    
                    # 计算损失
                    loss, loss_dict = deconv_loss_fn(
                        pred_fractions, 
                        cell_fractions,
                        reconstruction=None,  # 暂时不使用重构损失
                        input_values=None
                    )
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # 更新统计
                    total_loss += loss.item()
                    total_pred_loss += loss_dict["pred_loss"].item()
                    if "recon_loss" in loss_dict:
                        total_recon_loss += loss_dict["recon_loss"].item()
                else:
                    print(f"警告：模型输出为None或非tensor")
                    continue
                    
            # 打印进度
            if batch_idx % 10 == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                progress = batch_idx / len(loader)
                print(f'| 轮次 {epoch:3d} | {progress:5.2f} | 损失 {loss.item():5.4f} | '\
                     f'预测损失 {loss_dict["pred_loss"].item():5.4f} | '\
                     f'ms/batch {ms_per_batch:5.2f} |')
                
        except Exception as e:
            print(f"批次处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算平均损失
    avg_loss = total_loss / max(1, len(loader))
    avg_pred_loss = total_pred_loss / max(1, len(loader))
    avg_recon_loss = total_recon_loss / max(1, len(loader)) if total_recon_loss > 0 else 0
    
    print(f'| 轮次结束 {epoch:3d} | 平均损失 {avg_loss:5.4f} | '\
          f'平均预测损失 {avg_pred_loss:5.4f} | '\
          f'平均重构损失 {avg_recon_loss:5.4f} |')
    
    return avg_loss, avg_pred_loss, avg_recon_loss


# 评估函数
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_recon_loss = 0.0
    num_valid_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            try:
                # 准备数据
                gene_ids = batch["gene_ids"].to(device)
                values = batch["values"].to(device)
                cell_fractions = batch["cell_fractions"].to(device)
                
                # 创建padding掩码
                src_key_padding_mask = (gene_ids == vocab[pad_token]).bool()
                
                # 前向传播
                output_dict = model(
                    src=gene_ids,
                    values=values,
                    src_key_padding_mask=src_key_padding_mask,
                )
                
                # 处理输出
                if isinstance(output_dict, dict):
                    output = output_dict.get("mlm_output", 
                             output_dict.get("decoded", 
                             output_dict.get("last_hidden_state", None)))
                    if output is None and len(output_dict) > 0:
                        output = list(output_dict.values())[0]
                else:
                    output = output_dict
                
                # 根据输出维度正确获取表示
                if output is not None and torch.is_tensor(output):
                    if output.dim() == 3:
                        cls_output = output[:, 0, :]
                    elif output.dim() == 2:
                        cls_output = output
                    else:
                        raise ValueError(f"Unexpected output dimension: {output.dim()}")
                    
                    # 预测细胞比例
                    pred_fractions = model.deconv_head(cls_output)
                    
                    # 计算损失
                    loss, loss_dict = deconv_loss_fn(
                        pred_fractions, 
                        cell_fractions,
                        reconstruction=None,
                        input_values=None
                    )
                    
                    # 更新统计
                    total_loss += loss.item()
                    total_pred_loss += loss_dict["pred_loss"].item()
                    if "recon_loss" in loss_dict:
                        total_recon_loss += loss_dict["recon_loss"].item()
                    
                    num_valid_batches += 1
                    
            except Exception as e:
                print(f"验证批次处理错误: {str(e)}")
                continue
    
    # 防止除零
    if num_valid_batches == 0:
        return float('inf'), float('inf'), float('inf')
        
    # 计算平均损失
    avg_loss = total_loss / num_valid_batches
    avg_pred_loss = total_pred_loss / num_valid_batches
    avg_recon_loss = total_recon_loss / num_valid_batches if total_recon_loss > 0 else 0
    
    print(f'| 评估 | 平均损失 {avg_loss:5.4f} | '\
          f'平均预测损失 {avg_pred_loss:5.4f} | '\
          f'平均重构损失 {avg_recon_loss:5.4f} |')
    
    return avg_loss, avg_pred_loss, avg_recon_loss
# 设置保存模型的路径
model_save_dir = Path(f"./save/tape_finetune_{time.strftime('%b%d-%H-%M')}")
model_save_dir.mkdir(parents=True, exist_ok=True)
print(f"模型将保存在: {model_save_dir}")


# In[52]:


# 在训练循环开始添加
print("检查模型结构")
for name, module in model.named_children():
    print(f"模块名: {name}, 类型: {type(module)}")


# In[53]:


# 开始训练
print("开始训练过程...")
best_val_loss = float('inf')
train_losses = []
val_losses = []

epochs = config["epochs"]
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loss, train_pred_loss, train_recon_loss = train_epoch(
        model, train_loader, optimizer, scheduler, device, epoch
    )
    val_loss, val_pred_loss, val_recon_loss = evaluate(model, valid_loader, device)
    
    # 记录损失
    train_losses.append((train_loss, train_pred_loss, train_recon_loss))
    val_losses.append((val_loss, val_pred_loss, val_recon_loss))
    
    # 打印本轮结果
    print('-' * 89)
    print(f'| 轮次 {epoch:3d} | 时间: {(time.time() - epoch_start_time):5.2f}s | '\
          f'验证损失 {val_loss:5.4f} | 验证预测损失 {val_pred_loss:5.4f} |')
    print('-' * 89)
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_dir / "best_model.pt")
        print(f"模型已保存: {model_save_dir / 'best_model.pt'}")
    
    # 每隔特定轮次保存检查点
    if epoch % config["save_eval_interval"] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
        }, model_save_dir / f"checkpoint_epoch_{epoch}.pt")
        print(f"检查点已保存: {model_save_dir / f'checkpoint_epoch_{epoch}.pt'}")

# 保存最终模型和配置
torch.save(model.state_dict(), model_save_dir / "final_model.pt")
print(f"最终模型已保存: {model_save_dir / 'final_model.pt'}")

# 保存词汇表和配置
with open(model_save_dir / "vocab.json", "w") as f:
    json.dump(vocab.get_stoi(), f)

with open(model_save_dir / "args.json", "w") as f:
    json.dump(hyperparameter_defaults, f, indent=4)
    
print("训练完成！")


# In[ ]:




