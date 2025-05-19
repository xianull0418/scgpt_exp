#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 设置全局参数
sc.settings.verbosity = 3  # 显示详细信息
sc.settings.set_figure_params(dpi=100, facecolor='white')

# 定义路径
raw_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_raw.h5ad"
output_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated_new.h5ad"
fig_dir = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/figures"

# 确保输出和图像目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# 加载原始数据
print("加载原始数据...")
adata = sc.read_h5ad(raw_path)
print(f"原始数据形状: {adata.shape}")

# 1. 基础预处理
print("\n执行基础预处理...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 计算质控指标
print("计算质控指标...")
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 线粒体基因
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# 可视化QC指标
print("生成QC指标图...")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=axs[0])
sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=axs[1])
sc.pl.violin(adata, 'pct_counts_mt', jitter=0.4, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{fig_dir}/qc_metrics.png")
plt.close()

# 绘制QC散点图
print("生成QC散点图...")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axs[0])
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axs[1])
plt.tight_layout()
plt.savefig(f"{fig_dir}/qc_scatter.png")
plt.close()

# 根据质控指标过滤
print("根据质控指标过滤...")
sc.pp.filter_cells(adata, max_counts=2500)
sc.pp.filter_cells(adata, max_genes=700)

# 2. 归一化处理
print("\n执行归一化处理...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 3. 选择高变基因
print("\n选择高变基因...")
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2100)

# 可视化高变基因
print("生成高变基因图...")
plt.figure(figsize=(10, 7))
sc.pl.highly_variable_genes(adata)
plt.savefig(f"{fig_dir}/highly_variable_genes.png")
plt.close()

adata = adata[:, adata.var.highly_variable]
print(f"高变基因筛选后形状: {adata.shape}")

# 4. 线性回归去除技术效应 (可选)
# sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# 5. 缩放数据
print("\n缩放数据...")
sc.pp.scale(adata, max_value=10)

# 6. 维度规约 (PCA)
print("\n执行PCA降维...")
sc.tl.pca(adata, svd_solver='arpack')

# 可视化PCA结果
print("生成PCA图...")
plt.figure(figsize=(10, 8))
sc.pl.pca(adata, color=adata.var_names[0:3])
plt.savefig(f"{fig_dir}/pca.png")
plt.close()

# PCA碎石图
print("生成PCA碎石图...")
plt.figure(figsize=(8, 6))
sc.pl.pca_variance_ratio(adata, n_pcs=30)
plt.savefig(f"{fig_dir}/pca_variance_ratio.png")
plt.close()

# 7. 构建邻居图
print("\n构建邻居图...")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

# 8. 非线性降维 (UMAP)
print("\n执行UMAP降维...")
sc.tl.umap(adata)

# 可视化UMAP结果
print("生成UMAP图...")
plt.figure(figsize=(8, 6))
sc.pl.umap(adata, color=adata.var_names[0:3])
plt.savefig(f"{fig_dir}/umap_genes.png")
plt.close()

# 9. 聚类
print("\n执行leiden聚类...")
sc.tl.leiden(adata, resolution=0.5)

# 可视化聚类结果
print("生成聚类图...")
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='leiden')
plt.savefig(f"{fig_dir}/umap_leiden.png")
plt.close()

# 10. 将leiden聚类结果复制为CellType注释
print("\n添加CellType注释...")
adata.obs['CellType'] = adata.obs['leiden']

# 可视化CellType注释
print("生成CellType注释图...")
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='CellType')
plt.savefig(f"{fig_dir}/umap_celltype.png")
plt.close()

# 11. 保存注释后的数据
print("\n保存注释后的数据...")
adata.write(output_path)

# 12. 生成聚类标记基因（可选）
print("\n查找每个聚类的标记基因...")
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
plt.figure(figsize=(12, 10))
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
plt.savefig(f"{fig_dir}/marker_genes.png")
plt.close()

# 保存标记基因到CSV
print("保存标记基因...")
marker_genes = pd.DataFrame()
for cluster in adata.obs['leiden'].unique():
    genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'][cluster])
    scores = pd.DataFrame(adata.uns['rank_genes_groups']['scores'][cluster])
    pvals = pd.DataFrame(adata.uns['rank_genes_groups']['pvals'][cluster])
    logfoldchanges = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'][cluster])
    
    df = pd.concat([genes, scores, pvals, logfoldchanges], axis=1)
    df.columns = ['gene', 'score', 'pval', 'logfoldchange']
    df['cluster'] = cluster
    marker_genes = pd.concat([marker_genes, df])

marker_genes.to_csv(f"{fig_dir}/marker_genes.csv", index=False)

print(f"\n处理完成！结果已保存到: {output_path}")
print(f"图形已保存到: {fig_dir}")
print(f"最终数据形状: {adata.shape}") 