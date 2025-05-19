#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局参数
sc.settings.verbosity = 3  # 显示详细信息

# 加载数据
print("加载数据...")
raw_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_raw.h5ad"
annotated_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated.h5ad"

raw = sc.read_h5ad(raw_path)
annotated = sc.read_h5ad(annotated_path)

print("\n========= 原始数据(raw)信息 =========")
print(raw)
print("\n原始数据形状:", raw.shape)
print("原始数据观测数量:", raw.n_obs)
print("原始数据基因数量:", raw.n_vars)
print("\n原始数据中的观测注释信息:")
print(raw.obs.head())
print("\n原始数据中的变量注释信息:")
print(raw.var.head())
print("\n原始数据中的层(layers):")
print(list(raw.layers) if hasattr(raw, 'layers') and raw.layers else "无")
print("\n原始数据中的未处理矩阵:")
if raw.raw is not None:
    print("包含raw属性")
else:
    print("不包含raw属性")

print("\n========= 注释后数据(annotated)信息 =========")
print(annotated)
print("\n注释后数据形状:", annotated.shape)
print("注释后数据观测数量:", annotated.n_obs)
print("注释后数据基因数量:", annotated.n_vars)
print("\n注释后数据中的观测注释信息:")
print(annotated.obs.head())
print("\n注释后数据变量注释信息:")
print(annotated.var.head())
print("\n注释后数据中的层(layers):")
print(list(annotated.layers) if hasattr(annotated, 'layers') and annotated.layers else "无")
print("\n注释后数据中的未处理矩阵:")
if annotated.raw is not None:
    print("包含raw属性")
else:
    print("不包含raw属性")

# 比较观测(细胞)注释
print("\n========= 观测(细胞)注释比较 =========")
raw_cols = set(raw.obs.columns)
annotated_cols = set(annotated.obs.columns)
print("原始数据特有的观测注释列:", raw_cols - annotated_cols)
print("注释后数据特有的观测注释列:", annotated_cols - raw_cols)
print("两者共有的观测注释列:", raw_cols & annotated_cols)

# 比较变量(基因)注释
print("\n========= 变量(基因)注释比较 =========")
raw_var_cols = set(raw.var.columns)
annotated_var_cols = set(annotated.var.columns)
print("原始数据特有的变量注释列:", raw_var_cols - annotated_var_cols)
print("注释后数据特有的变量注释列:", annotated_var_cols - raw_var_cols)
print("两者共有的变量注释列:", raw_var_cols & annotated_var_cols)

# 检查预处理状态
print("\n========= 预处理状态比较 =========")
if hasattr(raw, 'uns') and 'log1p' in raw.uns:
    print("原始数据已进行log1p归一化")
else:
    print("原始数据未进行log1p归一化")

if hasattr(annotated, 'uns') and 'log1p' in annotated.uns:
    print("注释后数据已进行log1p归一化")
else:
    print("注释后数据未进行log1p归一化")

# 如果两者大小不同，检查两者之间的细胞和基因重叠
print("\n========= 细胞和基因重叠检查 =========")
cells_in_both = len(set(raw.obs_names).intersection(set(annotated.obs_names)))
genes_in_both = len(set(raw.var_names).intersection(set(annotated.var_names)))

print(f"两个数据集中共有的细胞数: {cells_in_both}")
print(f"两个数据集中共有的基因数: {genes_in_both}")

print("\n完成比较！") 