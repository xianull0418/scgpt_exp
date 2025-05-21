# 导入 PyTorch 相关的库
import torch
import torch.nn as nn # PyTorch 神经网络模块
import torch.optim as optim # PyTorch 优化器模块
from torch.utils.data import Dataset, DataLoader # PyTorch 数据集和数据加载器工具

# 导入其他常用库
import numpy as np # NumPy 用于数值计算
import pandas as pd # Pandas 用于数据处理 (此处主要用于DataFrame)
import scanpy as sc # Scanpy 用于单细胞数据分析
import json # JSON 模块用于读写 JSON 文件 (例如配置文件、词汇表)
from pathlib import Path # Pathlib 用于面向对象的文件路径操作
import sys # Sys 模块用于与 Python 解释器交互 (例如退出脚本)
import os # OS 模块用于与操作系统交互 (例如创建目录，尽管Pathlib更常用)
import time # Time 模块用于计时
import argparse # Argparse 用于解析命令行参数
import random # Random 模块用于生成随机数
from typing import Optional, Dict, Any, List, Union, Tuple # Typing 模块用于类型提示，增强代码可读性

# --- 用于可复现性的工具函数 ---
def set_seed(seed: int):
    """设置随机种子以确保结果可复现。"""
    random.seed(seed) # Python 内置 random 模块的种子
    np.random.seed(seed) # NumPy 的随机种子
    torch.manual_seed(seed) # PyTorch CPU 的随机种子
    if torch.cuda.is_available(): # 如果 CUDA 可用 (即有可用的 GPU)
        torch.cuda.manual_seed(seed) # PyTorch 当前 GPU 的随机种子
        torch.cuda.manual_seed_all(seed) # PyTorch 所有 GPU 的随机种子
    # 可以选择性地设置 torch.backends.cudnn.deterministic 和 benchmark 以进一步控制复现性
    # torch.backends.cudnn.deterministic = True # 确保 cuDNN 使用确定性算法
    # torch.backends.cudnn.benchmark = False    # 禁用 cuDNN 的基准测试模式 (可能导致非确定性)
    print(f"随机种子已设置为 {seed}")

# --- 1. 数据准备 (来自 prepare_scgpt_data.py) ---
# 定义加载和准备单细胞数据的函数
def load_and_prepare_data(
    adata_path_str: str, # AnnData 文件路径字符串
    model_vocab_path_str: str # 模型词汇表 (vocab.json) 的路径字符串
) -> Tuple[Optional[sc.AnnData], Optional[Any], Optional[List[str]]]: # 返回值类型提示
    """
    加载单细胞 AnnData 对象和 scGPT 模型词汇表，
    然后筛选 AnnData 对象以仅包含两者共有的基因。

    Args:
        adata_path_str (str): .h5ad AnnData 文件的路径。
        model_vocab_path_str (str): 模型词汇表 (vocab.json) 文件的路径。

    Returns:
        tuple: 一个元组，包含：
            - filtered_adata (sc.AnnData | None): 筛选后仅包含共同基因的 AnnData 对象。
            - vocab_object (GeneVocab | None): 加载的 GeneVocab 对象。
            - common_genes (list[str] | None): 共同基因名称列表。
    """
    print(f"从路径加载 AnnData: {adata_path_str}")
    adata_path = Path(adata_path_str) # 转换为 Path 对象
    if not adata_path.exists(): # 检查文件是否存在
        print(f"错误: 在 {adata_path_str} 未找到 AnnData 文件")
        return None, None, None # 文件不存在则返回 None
    
    try:
        adata = sc.read_h5ad(adata_path) # 读取 .h5ad 文件
        print(f"成功加载 AnnData 对象，包含 {adata.n_obs} 个细胞和 {adata.n_vars} 个基因。")
    except Exception as e: # 捕获读取错误
        print(f"读取 AnnData 文件 {adata_path_str} 时出错: {e}")
        return None, None, None

    vocab_file_path = Path(model_vocab_path_str) # 词汇表文件路径
    print(f"从路径加载词汇表: {vocab_file_path}")

    if not vocab_file_path.exists(): # 检查词汇表文件是否存在
        print(f"错误: 在 {vocab_file_path} 未找到词汇表文件 (vocab.json)")
        return None, None, None

    try:
        # 使用稍后在此脚本中定义的 GeneVocab 类
        vocab_object = GeneVocab.from_file(vocab_file_path) # 从文件加载词汇表
        # 从词汇表对象中提取基因列表
        if isinstance(vocab_object.stoi, dict): # stoi: string to index
            model_gene_list_from_vocab = list(vocab_object.stoi.keys())
        else: # 如果 stoi 不是字典 (正常 GeneVocab 不应发生此情况)
            print("警告: vocab_object.stoi 不是字典。尝试推断基因。")
            model_gene_list_from_vocab = [vocab_object.itos[i] for i in range(len(vocab_object))] # itos: index to string

        print(f"成功加载词汇表，包含 {len(model_gene_list_from_vocab)} 个基因/标记。")
    except Exception as e: # 捕获加载或解析词汇表时的错误
        print(f"加载或解析词汇表文件 {vocab_file_path} 时出错: {e}")
        return None, None, None

    original_adata_genes = list(adata.var_names) # AnnData 中的原始基因列表
    print(f"原始 AnnData 中的基因数量: {len(original_adata_genes)}")

    # 识别共同基因 (区分大小写)
    common_genes = sorted(list(set(original_adata_genes) & set(model_gene_list_from_vocab)))
    
    if not common_genes: # 如果没有共同基因
        print("错误: 在 AnnData 和词汇表之间未找到共同基因。")
        print("请检查基因名称约定 (例如 ENSEMBL ID 与基因符号) 和物种。")
        return None, vocab_object, None # 即使没有共同基因，也可能返回词汇表对象
        
    print(f"找到的共同基因数量: {len(common_genes)}")

    # 筛选 AnnData 以仅保留共同基因
    adata_filtered = adata[:, common_genes].copy() # 使用 .copy() 确保是副本
    print(f"已将 AnnData 筛选至 {adata_filtered.n_vars} 个共同基因。")

    return adata_filtered, vocab_object, common_genes # 返回筛选后的数据、词汇表对象和共同基因列表


# --- 2. 伪批量样本生成 (来自 create_pseudo_bulk.py) ---
# 定义生成伪批量样本的函数
def generate_pseudo_bulk_samples(
    adata_filtered: sc.AnnData, # 经过基因筛选的 AnnData 对象
    cell_type_col: str, # .obs 中细胞类型注释的列名
    n_pseudo_bulk_samples: int = 1000, # 要生成的伪批量样本数量
    n_cells_per_sample_min: int = 50, # 每个伪批量样本的最小细胞数
    n_cells_per_sample_max: int = 200, # 每个伪批量样本的最大细胞数
    aggregation_method: str = "sum" # 基因表达聚合方法 ("sum" 或 "mean")
) -> Optional[sc.AnnData]: # 返回包含伪批量数据的 AnnData 对象或 None
    print(f"开始生成伪批量样本: {n_pseudo_bulk_samples} 个样本...")
    print(f"参数: min_cells={n_cells_per_sample_min}, max_cells={n_cells_per_sample_max}, aggregation='{aggregation_method}'")

    # 参数验证
    if cell_type_col not in adata_filtered.obs.columns:
        print(f"错误: 未找到细胞类型列 '{cell_type_col}'。可用列: {list(adata_filtered.obs.columns)}")
        return None
    if aggregation_method not in ["sum", "mean"]:
        print(f"错误: 未知的聚合方法 '{aggregation_method}'。请选择 'sum' 或 'mean'。")
        return None
    if n_cells_per_sample_max < n_cells_per_sample_min:
        print(f"错误: n_cells_per_sample_max 不能小于 n_cells_per_sample_min。")
        return None
    if adata_filtered.n_obs < n_cells_per_sample_min:
        print(f"错误: 细胞不足 ({adata_filtered.n_obs})，无法满足每个样本最小细胞数 ({n_cells_per_sample_min})。")
        return None

    all_gene_names = list(adata_filtered.var_names) # 所有基因名称
    # 获取唯一的细胞类型并排序
    unique_cell_types = sorted(list(adata_filtered.obs[cell_type_col].astype('category').cat.categories))
    
    if not unique_cell_types: # 如果未找到细胞类型
        print(f"错误: 在列 '{cell_type_col}' 中未找到唯一的细胞类型。")
        return None
    print(f"找到 {len(unique_cell_types)} 种独特细胞类型: {unique_cell_types}")

    pseudo_bulk_expressions, pseudo_bulk_fractions = [], [] # 初始化列表存储结果
    # 判断输入数据是否为稀疏矩阵
    is_sparse = isinstance(adata_filtered.X, (sc.sparse.csr_matrix, sc.sparse.csc_matrix))
    print(f"输入 AnnData.X 是 {'稀疏' if is_sparse else '密集'} 矩阵。")

    # 循环生成每个伪批量样本
    for i in range(n_pseudo_bulk_samples):
        # 确定当前样本的细胞数
        n_cells = random.randint(n_cells_per_sample_min, n_cells_per_sample_max) \
            if n_cells_per_sample_min != n_cells_per_sample_max else n_cells_per_sample_min
        
        # 随机抽样细胞索引 (无放回)
        sampled_indices = random.sample(range(adata_filtered.n_obs), k=n_cells)
        sampled_X = adata_filtered.X[sampled_indices, :] # 提取抽样细胞的表达数据
        
        # 聚合基因表达
        agg_expr = sampled_X.sum(axis=0) if aggregation_method == "sum" else sampled_X.mean(axis=0)
        # 处理稀疏矩阵的输出 (转换为一维 NumPy 数组)
        if hasattr(agg_expr, "A1"): agg_expr = agg_expr.A1 
        elif hasattr(agg_expr, "toarray"): agg_expr = agg_expr.toarray().flatten()
        
        pseudo_bulk_expressions.append(agg_expr) # 添加到列表

        # 计算细胞类型比例
        sampled_cell_types = adata_filtered.obs[cell_type_col].iloc[sampled_indices]
        type_counts = sampled_cell_types.value_counts().reindex(unique_cell_types, fill_value=0.0) # 统计并补全缺失类型
        pseudo_bulk_fractions.append((type_counts / n_cells).values) # 计算比例并添加到列表

        # 打印进度
        if (i + 1) % (n_pseudo_bulk_samples // 10 if n_pseudo_bulk_samples >= 10 else 1) == 0:
            print(f"  已生成 {i + 1}/{n_pseudo_bulk_samples} 个伪批量样本...")

    try:
        pseudo_bulk_X_np = np.array(pseudo_bulk_expressions) # 转换为 NumPy 数组
    except Exception as e:
        print(f"错误: 将表达数据转换为 NumPy 数组失败: {e}")
        return None
    # 创建包含细胞类型比例的 DataFrame
    fractions_df = pd.DataFrame(pseudo_bulk_fractions, columns=unique_cell_types, 
                                index=[f"pseudo_bulk_sample_{i}" for i in range(n_pseudo_bulk_samples)])
    var_df = pd.DataFrame(index=all_gene_names) # 创建基因信息的 DataFrame (仅索引)
    
    try:
        # 创建最终的伪批量 AnnData 对象
        pseudo_bulk_adata = sc.AnnData(X=pseudo_bulk_X_np, obs=fractions_df, var=var_df)
    except Exception as e:
        print(f"错误: 创建伪批量数据的最终 AnnData 对象失败: {e}")
        return None

    print(f"\n伪批量 AnnData 创建完毕: X 形状 {pseudo_bulk_adata.X.shape}, obs 形状 {pseudo_bulk_adata.obs.shape}")
    return pseudo_bulk_adata


# --- 3. 伪批量数据预处理 (来自 preprocess_bulk_for_scgpt.py) ---
# 定义为 scGPT 预处理伪批量数据的函数
def preprocess_pseudo_bulk_for_scgpt(
    pseudo_bulk_adata: sc.AnnData, # 输入的伪批量 AnnData 对象
    target_sum: Optional[float] = 1e4, # 标准化目标总和 (例如 CPM)
    n_bins: int = 51, # 表达值分箱数量
    hvg_col_name: Optional[str] = None # .var 中高变基因列名 (可选)
) -> Optional[sc.AnnData]: # 返回处理后的 AnnData 对象或 None
    print("开始预处理伪批量数据...")
    adata_processed = pseudo_bulk_adata.copy() # 创建副本进行操作
    print(f"  输入形状: {adata_processed.shape}")

    # 可选的高变基因筛选
    if hvg_col_name:
        if hvg_col_name in adata_processed.var.columns and adata_processed.var[hvg_col_name].dtype == bool:
            print(f"  基于 HVG 列 '{hvg_col_name}' 进行筛选。")
            adata_processed = adata_processed[:, adata_processed.var[hvg_col_name]].copy()
            print(f"  HVG 筛选后形状: {adata_processed.shape}")
            if adata_processed.n_vars == 0: print("警告: HVG 筛选后无剩余基因。"); return None
        else: print(f"警告: HVG 列 '{hvg_col_name}' 未找到或非布尔类型。跳过 HVG 筛选。")

    # 标准化
    if target_sum is not None and target_sum > 0:
        if adata_processed.X is None: print("错误: .X 为 None，无法标准化。"); return None
        print(f"  将总计数标准化至 {target_sum}。")
        sc.pp.normalize_total(adata_processed, target_sum=target_sum) # 执行标准化
        adata_processed.layers['normalized'] = adata_processed.X.copy() # 存储标准化后的数据
    else:
        print("  跳过标准化 (target_sum 为 None 或 <=0)。")
        if adata_processed.X is None: print("错误: .X 为 None 且跳过标准化。无法继续。"); return None

    # Log1p 转换
    print("  应用 log1p 转换。")
    if 'normalized' in adata_processed.layers: # 如果已标准化，.X 已更新
        sc.pp.log1p(adata_processed) # 就地修改 .X
        adata_processed.layers['log1p'] = adata_processed.X.copy() # 存储 log1p 转换后的数据
    else: # 如果未标准化，直接对 .X 操作
        adata_processed.layers['log1p'] = np.log1p(adata_processed.X)
        adata_processed.X = adata_processed.layers['log1p'].copy() # 更新 .X

    # 表达值分箱
    print(f"  将对数转换后的值分箱为 {n_bins} 个 bin。")
    data_to_bin = adata_processed.layers['log1p'] # 获取待分箱数据
    if data_to_bin is None: print("错误: layers['log1p'] 为 None。无法分箱。"); return None
    
    finite_data = data_to_bin[np.isfinite(data_to_bin)] # 筛选有限值
    if finite_data.size == 0: print("错误: 无有限数据可供分箱。"); return None
    min_val, max_val = np.min(finite_data), np.max(finite_data) # 计算范围
    print(f"  分箱数据范围: min={min_val:.4f}, max={max_val:.4f}")

    if min_val == max_val: # 如果所有值相同
        print("警告: 所有待分箱值相同。分配到 bin 0。")
        binned_expression = np.zeros(data_to_bin.shape, dtype=int)
    else: # 正常分箱
        # 创建 n_bins-1 个边界点，用于 np.digitize 将数据划分到 n_bins 个 bin 中 (索引 0 到 n_bins-1)
        bin_edges = np.linspace(min_val, max_val, num=n_bins - 1)
        binned_expression = np.digitize(data_to_bin, bin_edges, right=False)
    
    adata_processed.layers['binned_expression'] = binned_expression.astype(np.int32) # 存储分箱结果
    print(f"  分箱完成。最小 bin: {np.min(binned_expression)}, 最大 bin: {np.max(binned_expression)}")
    if np.max(binned_expression) >= n_bins: print("警告: 最大 bin 索引 >= n_bins。请检查分箱逻辑。")
    
    print("预处理完成。")
    return adata_processed


# --- 4. 模型定义 (来自 deconvolution_model_def.py) ---
# 首先，定义 GeneVocab 和 TransformerModel (如果 scgpt 未安装，可以是虚拟类)
try:
    # 尝试导入真实的 scGPT 模型和词汇表类
    from scgpt.model import TransformerModel as scGPTTransformerModel
    from scgpt.tokenizer import GeneVocab as scGPTGeneVocab
    scgpt_available = True # 标记 scGPT 可用
except ImportError:
    # 如果导入失败，打印警告并使用虚拟类定义
    print("警告: 未找到 scGPT 库。将为 TransformerModel 和 GeneVocab 使用虚拟 (DUMMY) 类。")
    scgpt_available = False # 标记 scGPT 不可用
    class scGPTTransformerModel(nn.Module): # 虚拟 TransformerModel
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.d_model = kwargs.get("d_model", 128) # 获取模型维度或使用默认值
            # 定义一个简单的线性层模拟编码器
            self.dummy_encoder = nn.Linear(kwargs.get("ntoken", 100), self.d_model)
            print(f"虚拟 scGPTTransformerModel 已初始化。d_model={self.d_model}")
        def load_state_dict(self, state_dict, strict=True): print("虚拟 load_state_dict 已调用。"); return [],[] # 模拟加载状态字典
        def forward(self, src, values, src_key_padding_mask): # 模拟前向传播
            # 简化逻辑：如果 'values' 维度匹配，则使用它；否则使用 'src'。
            # 这个虚拟模型非常基础，不能反映真实的 scGPT 行为。
            # 假设 'values' 是 [B, SeqLen]，我们想将其投影到 d_model。
            # 对于这个虚拟模型，假设初始化时的 ntoken 是序列长度。
            if values.shape[1] == self.dummy_encoder.in_features:
                 simulated_emb = self.dummy_encoder(values) # [B, d_model]
            else: # 如果维度不匹配，则返回随机张量
                 simulated_emb = torch.randn(values.shape[0], self.d_model, device=values.device)
            return {"cell_emb": simulated_emb} # 返回符合预期的输出结构

    class scGPTGeneVocab: # 虚拟 GeneVocab
        def __init__(self, gene_list, specials=None): # 构造函数
            self.stoi = {} # string-to-index 字典
            idx = 0
            if specials: # 添加特殊标记
                for s in specials: self.stoi[s] = idx; idx+=1
            for g in gene_list: # 添加基因
                if g not in self.stoi: self.stoi[g] = idx; idx+=1
            self.itos = {i:s for s,i in self.stoi.items()} # index-to-string 字典
            self._pad_token, self._cls_token = "<pad>", "<cls>" # 定义填充和分类标记
            print(f"虚拟 scGPTGeneVocab 已初始化。大小: {len(self.stoi)}")
        def __getitem__(self, token): return self.stoi.get(token, -1) # 获取标记索引
        def __len__(self): return len(self.stoi) # 获取词汇表大小
        @property
        def pad_idx(self): return self.stoi.get(self._pad_token, 0) # 获取填充标记索引
        @property
        def cls_idx(self): return self.stoi.get(self._cls_token, 1) # 获取 CLS 标记索引
        @classmethod
        def from_file(cls, fp): # 从文件加载 (虚拟)
            with open(fp, 'r') as f: data = json.load(f)
            if isinstance(data, dict): return cls(list(data.keys())) # 如果是字典，用键创建
            return cls(data, specials=["<pad>", "<cls>"]) # 如果是列表，添加默认特殊标记

# 使 GeneVocab 全局可访问，指向真实或虚拟的实现
GeneVocab = scGPTGeneVocab
TransformerModel = scGPTTransformerModel

# 定义创建反卷积模型的函数
def create_deconvolution_model(
    model_dir_path: str, vocab_file_path: str, n_cell_types: int, # 基本路径和细胞类型数
    scgpt_dropout_rate: Optional[float] = None, head_dropout_rate: float = 0.1, # Dropout率
    freeze_scgpt_base: bool = False, d_model_override: Optional[int] = None, # 模型结构和冻结选项
    nhead_override: Optional[int] = None, nlayers_override: Optional[int] = None,
    pad_value: int = -2 # 填充值
) -> Optional[nn.Module]: # 返回 PyTorch 模型或 None
    print(f"从目录创建反卷积模型: {model_dir_path}, 词汇表: {vocab_file_path}")
    model_config_file = Path(model_dir_path) / "args.json" # 模型配置文件
    model_weights_file = Path(model_dir_path) / "best_model.pt" # 模型权重文件
    
    # 检查所有必需文件是否存在
    if not all([f.exists() for f in [model_config_file, model_weights_file, Path(vocab_file_path)]]):
        print("错误: 一个或多个模型/词汇表文件未找到。")
        # 如果 scGPT 不可用，无法使用真实模型；如果示例继续，将使用虚拟模型
        if not scgpt_available: print("scGPT 不可用，无法使用真实模型。如果示例继续，将使用虚拟模型。"); return None
        # 如果 scGPT 可用但文件缺失，对于真实运行则为确定性错误
        if scgpt_available: return None 
    
    try:
        if scgpt_available: # 仅当 scGPT 可用时加载真实模型配置
            with open(model_config_file, "r") as f: model_configs = json.load(f)
        else: # 虚拟模型的虚拟配置
            model_configs = {"embsize": 128, "nheads": 2, "nlayers": 2, "pad_token":"<pad>", "pad_value":-2,
                             "dropout":0.1, "input_emb_style":"continuous", "cell_emb_style":"cls", "n_cls":1, "nlayers_cls":1}
        vocab = GeneVocab.from_file(Path(vocab_file_path)) # 加载词汇表
        ntoken = len(vocab) # 词汇表大小
        pad_token = model_configs.get("pad_token", "<pad>") # 获取填充标记
        print(f"词汇表大小: {ntoken}, 填充标记: {pad_token}")
    except Exception as e: print(f"加载配置/词汇表时出错: {e}"); return None

    # 获取模型参数，优先使用覆盖值
    d_model = d_model_override if d_model_override else model_configs.get("embsize", 128)
    nhead = nhead_override if nhead_override else model_configs.get("nheads", 2)
    nlayers = nlayers_override if nlayers_override else model_configs.get("nlayers", 2)
    dropout = scgpt_dropout_rate if scgpt_dropout_rate is not None else model_configs.get("dropout", 0.1)
    
    # 实例化 Transformer 模型 (真实或虚拟)
    scgpt_model_instance = TransformerModel(
        ntoken=ntoken, d_model=d_model, nhead=nhead, d_hid=model_configs.get("d_hid", d_model*4),
        nlayers=nlayers, vocab=vocab, dropout=dropout, pad_token=pad_token, 
        pad_value=model_configs.get("pad_value", pad_value),
        # args.json 中常见的其他参数
        input_emb_style=model_configs.get("input_emb_style", "continuous"),
        n_input_bins=model_configs.get("n_input_bins", 0),
        cell_emb_style=model_configs.get("cell_emb_style", "cls"),
        n_cls=model_configs.get("n_cls",1),
        nlayers_cls=model_configs.get("nlayers_cls",1),
        use_fast_transformer=model_configs.get("use_fast_transformer", False) if scgpt_available else False, # 虚拟模型不使用此参数
        pre_norm=model_configs.get("pre_norm", False)
    )
    print(f"TransformerModel 已实例化 (d_model={d_model})。")

    # 如果是真实模型且权重文件存在，则加载预训练权重
    if scgpt_available and model_weights_file.exists():
        try:
            state_dict = torch.load(model_weights_file, map_location=torch.device('cpu')) # 加载权重
            # strict=False 允许模型中的某些键在状态字典中缺失 (例如新添加的头)
            missing, unexpected = scgpt_model_instance.load_state_dict(state_dict, strict=False)
            print(f"已加载预训练权重。缺失键: {len(missing)}, 意外键: {len(unexpected)}")
            if unexpected: print(f"  示例意外键: {unexpected[:3]}") # 打印一些意外的键以供调试
        except Exception as e: print(f"加载权重时出错: {e}")
    elif not scgpt_available: # 如果是虚拟模型
        print("跳过为虚拟 scGPT 模型加载权重。")
    else: # scGPT 可用但权重文件不存在
        print(f"警告: 未找到模型权重文件 {model_weights_file}。将从头开始初始化。")

    # 定义并附加反卷积头
    scgpt_model_instance.deconv_head = nn.Sequential(
        nn.Linear(d_model, d_model // 2), # 线性层
        nn.ReLU(), # ReLU 激活
        nn.Dropout(head_dropout_rate), # Dropout 层
        nn.Linear(d_model // 2, n_cell_types), # 输出层
        nn.Softmax(dim=-1) # Softmax 以获得概率分布
    )
    print(f"已添加反卷积头 (输出维度={n_cell_types})。")

    # 根据参数冻结基础模型
    if freeze_scgpt_base:
        print("正在冻结基础模型参数。")
        for name, param in scgpt_model_instance.named_parameters():
            if 'deconv_head' not in name: param.requires_grad = False # 冻结非头部的参数
            else: param.requires_grad = True # 确保头部参数可训练
    else: # 如果不冻结
        print("所有模型参数都可训练。")
        for param in scgpt_model_instance.parameters(): param.requires_grad = True # 确保所有参数都可训练
            
    return scgpt_model_instance # 返回构建好的模型


# --- 5. 数据加载器 (来自 deconvolution_dataloaders.py) ---
# 定义伪批量数据集类
class PseudoBulkDataset(Dataset):
    def __init__(self, pseudo_bulk_adata: sc.AnnData, vocab: GeneVocab, max_seq_len: int, 
                 expression_layer: str, cls_value_bin: int, pad_value: int):
        self.adata = pseudo_bulk_adata # AnnData 对象
        self.genes = list(pseudo_bulk_adata.var_names) # 基因列表
        self.vocab = vocab # 词汇表对象
        self.max_seq_len = max_seq_len # 最大序列长度
        self.expression_layer = expression_layer # 表达数据层名称
        self.cls_value_bin = cls_value_bin # CLS 标记的表达 bin 值
        self.pad_value = pad_value # 填充值
        if not self.genes: raise ValueError("AnnData var_names 为空。") # 检查基因列表是否为空
        # 将基因名转换为词汇表索引 (不包括 CLS)
        self.gene_ids_no_cls = [self.vocab[g] for g in self.genes]
        # 检查是否有基因未在词汇表中找到
        if any(gid == -1 for gid in self.gene_ids_no_cls):
            print(f"警告: 一些基因未在词汇表中找到: {[self.genes[i] for i,gid in enumerate(self.gene_ids_no_cls) if gid == -1][:5]}")

    def __len__(self): return self.adata.n_obs # 返回样本数量
    def __getitem__(self, idx): # 获取单个样本
        raw_expr = self.adata.layers[self.expression_layer][idx] # 获取原始表达数据
        # 转换为密集 NumPy 数组并展平
        expr_flat = raw_expr.toarray().flatten() if hasattr(raw_expr, "toarray") else np.asarray(raw_expr).flatten()
        
        # 获取细胞类型比例
        fractions_np = self.adata.obs.iloc[idx, :].values.astype(np.float32)
        
        # 构建基因 ID 序列 (CLS + 基因) 和表达值序列 (CLS值 + 基因表达)
        ids_list = [self.vocab.cls_idx] + self.gene_ids_no_cls
        vals_list = np.concatenate((np.array([self.cls_value_bin]), expr_flat))
        
        current_len = len(ids_list) # 当前序列长度
        padding_len = self.max_seq_len - current_len # 需要填充的长度
        
        # 根据需要进行截断或填充
        if padding_len < 0: # 截断
            ids_final = np.array(ids_list[:self.max_seq_len])
            vals_final = np.array(vals_list[:self.max_seq_len])
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool) # 无填充，全为 False
        elif padding_len > 0: # 填充
            ids_final = np.pad(ids_list, (0, padding_len), mode='constant', constant_values=self.vocab.pad_idx)
            vals_final = np.pad(vals_list, (0, padding_len), mode='constant', constant_values=self.pad_value)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[current_len:] = True # 标记填充部分为 True
        else: # 长度正好，无需操作
            ids_final = np.array(ids_list)
            vals_final = np.array(vals_list)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            
        # 返回包含处理后数据的字典
        return {"gene_ids": torch.tensor(ids_final, dtype=torch.long),
                "values": torch.tensor(vals_final, dtype=torch.float32),
                "padding_mask": mask, 
                "fractions": torch.tensor(fractions_np, dtype=torch.float32)}

# 定义创建数据加载器的函数
def create_dataloaders(
    train_adata: sc.AnnData, valid_adata: sc.AnnData, vocab: GeneVocab, batch_size: int, 
    expression_layer: str, cls_value_bin: int, pad_value: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]: # 返回训练和验证 DataLoader
    # 检查训练集和验证集的基因是否一致
    if not train_adata.var_names.equals(valid_adata.var_names):
        raise ValueError("训练和验证 AnnData 必须具有相同的 var_names。")
    max_seq_len = len(train_adata.var_names) + 1 # 计算最大序列长度 (基因数 + CLS)
    print(f"DataLoader 的最大序列长度: {max_seq_len}")

    # 创建训练和验证数据集实例
    train_ds = PseudoBulkDataset(train_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value)
    valid_ds = PseudoBulkDataset(valid_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value)

    # 创建训练数据加载器
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=num_workers > 0, drop_last=True) # drop_last=True 保证批次大小稳定
    # 创建验证数据加载器
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=num_workers > 0, drop_last=False) # drop_last=False 保留所有验证样本
    print(f"DataLoaders 已创建。训练批次数: {len(train_loader)}, 验证批次数: {len(valid_loader)}")
    return train_loader, valid_loader


# --- 6. 训练工具 (来自 training_utils.py) ---
# 定义获取损失函数的函数
def get_loss_function(loss_type: str = "L1") -> nn.Module:
    if loss_type == "L1": print("使用 L1 损失。"); return nn.L1Loss()
    if loss_type == "MSE": print("使用 MSE 损失。"); return nn.MSELoss()
    if loss_type == "KLDivergence": 
        print("使用 KL 散度损失。确保模型输出为对数概率，目标为概率。")
        return nn.KLDivLoss(reduction='batchmean') # batchmean 表示对批次内的损失取平均
    raise ValueError(f"不支持的损失函数类型: {loss_type}")

# 定义获取优化器的函数
def get_optimizer(model: nn.Module, lr: float, opt_type: str, weight_decay: float) -> optim.Optimizer:
    # 仅优化需要梯度的参数 (未冻结的参数)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    if opt_type == "AdamW": print(f"使用 AdamW (学习率={lr}, 权重衰减={weight_decay})。"); return optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    if opt_type == "Adam": print(f"使用 Adam (学习率={lr}, 权重衰减={weight_decay})。"); return optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"不支持的优化器类型: {opt_type}")

# 定义获取学习率调度器的函数
def get_scheduler(optimizer: optim.Optimizer, sched_type: Optional[str], total_steps: Optional[int], warmup_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if sched_type is None or sched_type.lower() == "none": print("不使用学习率调度器。"); return None
    if sched_type == "OneCycleLR": # OneCycleLR 调度器
        if total_steps is None: raise ValueError("OneCycleLR 需要 total_steps 参数。")
        # 计算预热阶段占总步数的比例
        pct_start = float(warmup_steps) / float(total_steps) if warmup_steps > 0 and total_steps > 0 else 0.3
        print(f"使用 OneCycleLR (最大学习率={optimizer.defaults['lr']}, 总步数={total_steps}, 预热比例={pct_start:.3f})。")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.defaults['lr'], total_steps=total_steps, pct_start=pct_start)
    if sched_type == "LinearWarmup": # 线性预热调度器 (使用 LambdaLR 实现)
        if warmup_steps <= 0: print("线性预热步数为0，实际为恒定学习率。"); return optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        # 定义学习率乘子函数
        def lr_lambda(step): return float(step+1)/float(warmup_steps) if step < warmup_steps else 1.0
        print(f"使用 LambdaLR 实现的线性预热，共 {warmup_steps} 步，之后为恒定学习率。"); return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if sched_type == "ReduceLROnPlateau": # ReduceLROnPlateau 调度器 (基于验证损失降低学习率)
        print("使用 ReduceLROnPlateau。"); return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
    raise ValueError(f"不支持的调度器类型: {sched_type}")


# --- 7. 微调循环 (来自 fine_tuning_loop.py) ---
# 定义运行微调的函数
def run_fine_tuning(
    model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: optim.Optimizer,
    loss_fn: nn.Module, n_epochs: int, device: torch.device, loss_type_str: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], checkpoint_dir: str, best_model_name: str
): # 参数类型提示
    print(f"开始微调: 在 {device} 上进行 {n_epochs} 轮训练")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True) # 创建检查点目录
    best_val_loss = float('inf') # 初始化最佳验证损失
    history = {"train_loss": [], "val_loss": [], "lr": []} # 存储训练历史

    # 训练循环
    for epoch in range(n_epochs):
        epoch_start_time = time.time() # 记录 epoch 开始时间
        model.train() # 设置模型为训练模式
        total_train_loss = 0.0 # 初始化总训练损失
        # 遍历训练数据
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移至设备
            gene_ids, values, padding_mask, true_fractions = \
                batch["gene_ids"].to(device), batch["values"].to(device), \
                batch["padding_mask"].to(device), batch["fractions"].to(device)
            
            optimizer.zero_grad() # 清零梯度
            # 模型前向传播
            scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
            if "cell_emb" not in scgpt_output: raise KeyError("模型输出缺少 'cell_emb'。") # 检查输出
            
            cell_embedding = scgpt_output["cell_emb"] # 获取细胞嵌入
            if not hasattr(model, 'deconv_head'): raise AttributeError("模型缺少 'deconv_head'。") # 检查是否有反卷积头
            predicted_fractions = model.deconv_head(cell_embedding) # 通过反卷积头获取预测比例

            # 计算损失
            if loss_type_str == "KLDivergence": # KL散度损失的特殊处理
                loss = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions) # 输入为对数概率
            else: # 其他损失
                loss = loss_fn(predicted_fractions, true_fractions)
            
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            # 更新学习率 (如果调度器是按步更新的)
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            total_train_loss += loss.item() # 累加训练损失
            # 打印批次训练信息
            if len(train_loader) > 4 and batch_idx % (len(train_loader)//4) == 0 and batch_idx > 0:
                 print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.3e}")

        avg_train_loss = total_train_loss / len(train_loader) # 计算平均训练损失
        history["train_loss"].append(avg_train_loss) # 记录
        history["lr"].append(optimizer.param_groups[0]['lr']) # 记录学习率

        # 验证阶段
        model.eval() # 设置模型为评估模式
        total_val_loss = 0.0 # 初始化总验证损失
        with torch.no_grad(): # 不计算梯度
            for batch in valid_loader: # 遍历验证数据
                # 数据移至设备和前向传播 (与训练阶段类似)
                gene_ids, values, padding_mask, true_fractions = \
                    batch["gene_ids"].to(device), batch["values"].to(device), \
                    batch["padding_mask"].to(device), batch["fractions"].to(device)
                scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
                cell_embedding = scgpt_output["cell_emb"]
                predicted_fractions = model.deconv_head(cell_embedding)
                # 计算验证损失
                if loss_type_str == "KLDivergence":
                    val_loss_batch = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
                else:
                    val_loss_batch = loss_fn(predicted_fractions, true_fractions)
                total_val_loss += val_loss_batch.item() # 累加验证损失
        avg_val_loss = total_val_loss / len(valid_loader) # 计算平均验证损失
        history["val_loss"].append(avg_val_loss) # 记录
        
        # 打印 epoch 摘要
        print(f"Epoch {epoch+1}/{n_epochs} Summary: Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.3e} | Time: {time.time()-epoch_start_time:.2f}s")

        # 更新学习率 (如果调度器是按轮更新的，例如 ReduceLROnPlateau)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(checkpoint_dir) / best_model_name) # 保存模型状态字典
            print(f"  最佳模型已保存 (验证损失: {best_val_loss:.4f})")
            
    print(f"\n微调完成。最佳验证损失: {best_val_loss:.4f}")
    return history # 返回训练历史

# --- 主执行块 ---
if __name__ == '__main__':
    # --- 参数解析 ---
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="scGPT 反卷积微调脚本")
    # 数据路径参数
    parser.add_argument("--adata_file_path", type=str, required=True, help="输入的单细胞 AnnData (.h5ad) 文件路径")
    parser.add_argument("--model_dir_path", type=str, required=True, help="预训练 scGPT 模型目录路径")
    parser.add_argument("--vocab_file_path", type=str, required=True, help="vocab.json 文件路径")
    parser.add_argument("--output_dir", type=str, default="./deconv_finetune_output", help="检查点和结果的输出目录")
    
    # 伪批量样本生成参数
    parser.add_argument("--cell_type_col", type=str, required=True, help="adata.obs 中细胞类型注释的列名")
    parser.add_argument("--n_pseudo_bulk_samples", type=int, default=1000, help="生成的伪批量样本数量")
    parser.add_argument("--min_cells_per_sample", type=int, default=50, help="每个伪批量样本的最小细胞数")
    parser.add_argument("--max_cells_per_sample", type=int, default=200, help="每个伪批量样本的最大细胞数")
    parser.add_argument("--aggregation_method", type=str, default="sum", choices=["sum", "mean"], help="基因表达聚合方法")
    
    # 预处理参数
    parser.add_argument("--norm_target_sum", type=float, default=1e4, help="标准化目标总和 (0 或负数则跳过)")
    parser.add_argument("--n_bins", type=int, default=51, help="表达值离散化的 bin 数量")
    parser.add_argument("--hvg_col", type=str, default=None, help="可选: .var 中标记 HVG 的列名，用于筛选伪批量数据")

    # 模型参数
    parser.add_argument("--freeze_scgpt_base", action='store_true', help="冻结基础 scGPT 模型参数")
    parser.add_argument("--head_dropout_rate", type=float, default=0.1, help="反卷积头的 Dropout 率")
    parser.add_argument("--scgpt_dropout_rate", type=float, default=None, help="覆盖 scGPT 内部 Dropout 率 (None 表示使用原始值)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--n_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--loss_type", type=str, default="L1", choices=["L1", "MSE", "KLDivergence"], help="损失函数类型")
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "Adam"], help="优化器类型")
    parser.add_argument("--scheduler_type", type=str, default="OneCycleLR", choices=["OneCycleLR", "LinearWarmup", "ReduceLROnPlateau", "None"], help="学习率调度器类型")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="预热阶段占总步数的比例")
    
    # 数据集/数据加载器参数
    parser.add_argument("--cls_value_bin", type=int, default=0, help="数据集中 CLS 标记的 bin 索引值")
    parser.add_argument("--expression_pad_value", type=int, default=-2, help="数据集中表达量填充值")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 使用的子进程数")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="用于可复现性的随机种子")
    
    # 解析命令行参数
    args = parser.parse_args()

    # --- 打印配置信息 ---
    print("--- 配置信息 ---")
    for arg, value in vars(args).items(): # 遍历所有参数并打印
        print(f"  {arg}: {value}")
    print("---------------------\n")

    # --- 设置随机种子和设备 ---
    set_seed(args.seed) # 设置随机种子
    Path(args.output_dir).mkdir(parents=True, exist_ok=True) # 创建输出目录
    # 选择设备 (GPU 或 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载和准备单细胞数据 ---
    print("\n--- 步骤 1: 加载和准备单细胞数据 ---")
    # 调用函数加载并初步处理单细胞数据，获取筛选后的 AnnData 对象、词汇表对象和共同基因列表
    filtered_adata, vocab_obj, common_genes = load_and_prepare_data(args.adata_file_path, args.vocab_file_path)
    if filtered_adata is None or vocab_obj is None: # 检查是否成功
        print("数据加载/准备出错，脚本退出。")
        sys.exit(1) # 退出脚本

    # --- 2. 生成伪批量样本 ---
    print("\n--- 步骤 2: 生成伪批量样本 ---")
    # 调用函数从筛选后的单细胞数据生成伪批量样本
    pseudo_bulk_adata = generate_pseudo_bulk_samples(
        filtered_adata, args.cell_type_col, args.n_pseudo_bulk_samples,
        args.min_cells_per_sample, args.max_cells_per_sample, args.aggregation_method
    )
    if pseudo_bulk_adata is None: # 检查是否成功
        print("伪批量样本生成出错，脚本退出。")
        sys.exit(1)

    # --- 3. 划分和预处理伪批量数据 ---
    print("\n--- 步骤 3: 划分和预处理伪批量数据 ---")
    # 简单划分训练集和验证集 (可以改进为使用 sklearn 的 train_test_split 以进行分层抽样)
    n_total_samples = pseudo_bulk_adata.n_obs # 总样本数
    val_frac = 0.2 # 验证集比例 (例如 20%)
    n_val_samples = int(n_total_samples * val_frac) # 验证集样本数
    n_train_samples = n_total_samples - n_val_samples # 训练集样本数
    
    # 在划分前打乱索引
    shuffled_indices = np.random.permutation(n_total_samples)
    train_indices = shuffled_indices[:n_train_samples] # 训练集索引
    valid_indices = shuffled_indices[n_train_samples:] # 验证集索引

    # 创建训练集和验证集的 AnnData 对象副本
    train_pb_adata_raw = pseudo_bulk_adata[train_indices, :].copy()
    valid_pb_adata_raw = pseudo_bulk_adata[valid_indices, :].copy()
    print(f"已划分伪批量数据: 训练集 {train_pb_adata_raw.n_obs}, 验证集 {valid_pb_adata_raw.n_obs}")

    # 分别对训练集和验证集进行预处理
    train_processed_adata = preprocess_pseudo_bulk_for_scgpt(
        train_pb_adata_raw, args.norm_target_sum if args.norm_target_sum > 0 else None, # 条件标准化
        args.n_bins, args.hvg_col
    )
    valid_processed_adata = preprocess_pseudo_bulk_for_scgpt(
        valid_pb_adata_raw, args.norm_target_sum if args.norm_target_sum > 0 else None, 
        args.n_bins, args.hvg_col
    )
    if train_processed_adata is None or valid_processed_adata is None: # 检查是否成功
        print("伪批量数据预处理出错，脚本退出。")
        sys.exit(1)
    
    # 确保 HVG 筛选后 var_names 一致 (如果使用了 HVG 筛选)
    # HVG 筛选应在划分前基于完整伪批量数据进行，或基于训练集的 HVG 一致地应用于训练集和验证集。
    # 此处为简化，如果使用了 hvg_col，则假设已应用，然后取共同基因。
    if args.hvg_col:
        # 获取 HVG 筛选后训练集和验证集的共同基因
        common_vars_after_hvg = list(train_processed_adata.var_names.intersection(valid_processed_adata.var_names))
        train_processed_adata = train_processed_adata[:, common_vars_after_hvg].copy() # 再次筛选
        valid_processed_adata = valid_processed_adata[:, common_vars_after_hvg].copy()
        print(f"HVG 筛选后确保共同基因: {len(common_vars_after_hvg)} 个基因。")
        if not common_vars_after_hvg: print("错误: HVG 筛选后无共同基因。"); sys.exit(1)
        # 注意：重新分配 vocab_obj 以仅使用这些共同基因是复杂的；
        # 从原始数据加载的 vocab_obj 应该被使用。
        # PseudoBulkDataset 将使用 processed_adata 中的 var_names。
        # vocab_obj 仍应包含所有原始共同基因。

    # --- 4. 模型设置 ---
    print("\n--- 步骤 4: 设置反卷积模型 ---")
    # 细胞类型数量由伪批量数据 .obs 的列数 (即细胞类型比例的列数) 决定
    n_cell_types = train_processed_adata.obs.shape[1]
    print(f"用于反卷积头的细胞类型数量: {n_cell_types}")
    
    # 调用函数创建反卷积模型
    model = create_deconvolution_model(
        model_dir_path=args.model_dir_path,
        vocab_file_path=args.vocab_file_path,
        n_cell_types=n_cell_types,
        scgpt_dropout_rate=args.scgpt_dropout_rate,
        head_dropout_rate=args.head_dropout_rate,
        freeze_scgpt_base=args.freeze_scgpt_base
        # 如果需要，可以添加 d_model_override, nhead_override, nlayers_override
    )
    if model is None: # 检查模型是否成功创建
        print("模型创建出错，脚本退出。")
        sys.exit(1)
    model.to(device) # 将模型移至选定设备

    # --- 5. 创建数据加载器 ---
    print("\n--- 步骤 5: 创建数据加载器 ---")
    # 调用函数为训练集和验证集创建 DataLoader
    train_loader, valid_loader = create_dataloaders(
        train_processed_adata, valid_processed_adata, vocab_obj, args.batch_size,
        expression_layer='binned_expression', # 这是 preprocess_pseudo_bulk_for_scgpt 存储分箱数据的位置
        cls_value_bin=args.cls_value_bin,
        pad_value=args.expression_pad_value,
        num_workers=args.num_workers
    )

    # --- 6. 设置训练组件 ---
    print("\n--- 步骤 6: 设置训练组件 ---")
    loss_fn = get_loss_function(args.loss_type) # 获取损失函数
    optimizer = get_optimizer(model, args.learning_rate, args.optimizer_type, args.weight_decay) # 获取优化器
    
    # 计算总训练步数和预热步数 (用于某些学习率调度器)
    total_training_steps = args.n_epochs * len(train_loader)
    warmup_training_steps = int(args.warmup_frac * total_training_steps)
    print(f"总训练步数: {total_training_steps}, 预热步数: {warmup_training_steps}")
    
    scheduler = get_scheduler(optimizer, args.scheduler_type, total_training_steps, warmup_training_steps) # 获取学习率调度器

    # --- 7. 运行微调 ---
    print("\n--- 步骤 7: 开始微调 ---")
    # 定义最佳模型的保存名称
    best_model_save_name = f"best_model_deconv_epochs{args.n_epochs}_lr{args.learning_rate}_{args.loss_type}.pt"
    
    # 调用核心函数执行微调循环
    history = run_fine_tuning(
        model, train_loader, valid_loader, optimizer, loss_fn, args.n_epochs, device,
        args.loss_type, scheduler, args.output_dir, best_model_save_name
    )

    # --- 8. 保存训练历史 (可选) ---
    # 构建训练历史文件的保存路径
    history_file = Path(args.output_dir) / f"training_history_epochs{args.n_epochs}_lr{args.learning_rate}_{args.loss_type}.json"
    try:
        with open(history_file, 'w') as f: # 以写入模式打开文件
            json.dump(history, f, indent=4) # 将历史记录字典以 JSON 格式写入文件，indent=4 使其更易读
        print(f"\n训练历史已保存至 {history_file}")
    except Exception as e: # 捕获保存错误
        print(f"\n保存训练历史时出错: {e}")

    print("\n--- 微调脚本执行完毕 ---")

```
