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
output_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated_matched.h5ad"
fig_dir = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/figures_matched"

# 确保输出和图像目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# 加载原始数据和参考注释数据
print("加载原始数据...")
adata = sc.read_h5ad(raw_path)
print(f"原始数据形状: {adata.shape}")

print("加载参考注释数据...")
annotated = sc.read_h5ad("/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated.h5ad")
print(f"参考注释数据形状: {annotated.shape}")

# 基础预处理 - 但保留更多细胞，接近原始annotated文件
print("\n执行基础预处理...")
# 保留原始文件中的所有2700个细胞
# 轻微过滤，确保基本的数据质量
sc.pp.filter_cells(adata, min_genes=50)  # 设置非常低的阈值
sc.pp.filter_genes(adata, min_cells=3)   # 基本的基因过滤

# 计算质控指标，但不用于严格过滤
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

# 移除严格的细胞过滤 - 不使用max_counts和max_genes过滤
# 移除或最小化这些过滤，以保持细胞数量为2700
# 只有在数据质量非常差的情况下才过滤
print("执行最小化过滤，保留细胞数量...")
sc.pp.filter_cells(adata, min_genes=100)  # 非常宽松的过滤

# 归一化处理
print("\n执行归一化处理...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 选择高变基因 - 确保数量与annotated文件一致
print("\n选择高变基因...")
# 设置n_top_genes=2078，与参考文件一致
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2078)

# 可视化高变基因
print("生成高变基因图...")
plt.figure(figsize=(10, 7))
sc.pl.highly_variable_genes(adata)
plt.savefig(f"{fig_dir}/highly_variable_genes.png")
plt.close()

# 过滤数据，只保留高变基因
adata = adata[:, adata.var.highly_variable]
print(f"高变基因筛选后形状: {adata.shape}")

# 缩放数据
print("\n缩放数据...")
sc.pp.scale(adata, max_value=10)

# 维度规约 (PCA)
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

# 构建邻居图
print("\n构建邻居图...")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

# UMAP降维
print("\n执行UMAP降维...")
sc.tl.umap(adata)

# 可视化UMAP结果
print("生成UMAP图...")
plt.figure(figsize=(8, 6))
sc.pl.umap(adata, color=adata.var_names[0:3])
plt.savefig(f"{fig_dir}/umap_genes.png")
plt.close()

# 执行leiden聚类，尝试获得与参考文件相似的聚类结果
print("\n执行leiden聚类...")
# 检查参考文件中的聚类数量
n_clusters_reference = len(annotated.obs['leiden'].cat.categories)
print(f"参考文件中的聚类数量: {n_clusters_reference}")

# 调整resolution参数以尝试获得相同数量的聚类
resolution = 0.5  # 默认值
for res in np.linspace(0.1, 2.0, 20):  # 尝试多个分辨率值
    sc.tl.leiden(adata, resolution=res)
    n_clusters = len(adata.obs['leiden'].cat.categories)
    print(f"分辨率 {res:.2f} 产生 {n_clusters} 个聚类")
    if n_clusters == n_clusters_reference:
        resolution = res
        print(f"找到匹配的分辨率: {resolution}")
        break
    
# 使用找到的最佳分辨率重新执行聚类
sc.tl.leiden(adata, resolution=resolution)
print(f"最终使用分辨率: {resolution}, 产生聚类数: {len(adata.obs['leiden'].cat.categories)}")

# 可视化聚类结果
print("生成聚类图...")
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='leiden')
plt.savefig(f"{fig_dir}/umap_leiden.png")
plt.close()

# 将leiden聚类结果复制为CellType注释，与原始文件一致
print("\n添加CellType注释...")
adata.obs['CellType'] = adata.obs['leiden']

# 可视化CellType注释
print("生成CellType注释图...")
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='CellType')
plt.savefig(f"{fig_dir}/umap_celltype.png")
plt.close()

# 减少.obs和.var中不必要的列，使其更接近原始annotated文件
print("\n清理元数据以匹配参考文件...")
# 保留与参考文件相似的obs列
keep_obs_columns = ['leiden', 'CellType']
for col in list(adata.obs.columns):
    if col not in keep_obs_columns:
        del adata.obs[col]

# 保留与参考文件相似的var列
keep_var_columns = ['gene_ids', 'highly_variable', 'means', 'dispersions', 'dispersions_norm', 'mean', 'std']
for col in list(adata.var.columns):
    if col not in keep_var_columns and col != 'mt':  # 保留mt列用于内部处理
        del adata.var[col]

# 保存注释后的数据
print("\n保存匹配的注释数据...")
adata.write(output_path)

# 生成聚类标记基因
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

# 比较结果与参考文件的统计信息
print("\n比较生成结果与参考文件...")
print(f"参考文件形状: {annotated.shape}")
print(f"生成文件形状: {adata.shape}")

print("\n处理完成！结果已保存到: {output_path}")
print(f"图形已保存到: {fig_dir}")
print(f"最终数据形状: {adata.shape}")

# 可选：显示两个文件的UMAP对比图
try:
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    sc.pl.umap(annotated, color='leiden', show=False, title='参考文件UMAP')
    plt.subplot(1, 2, 2)
    sc.pl.umap(adata, color='leiden', show=False, title='生成文件UMAP')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/umap_comparison.png")
    plt.close()
    print("生成UMAP比较图")
except Exception as e:
    print(f"生成比较图失败: {e}") 