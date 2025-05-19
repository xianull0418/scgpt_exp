#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置全局参数
sc.settings.verbosity = 3  # 显示详细信息
sc.settings.set_figure_params(dpi=100, facecolor='white')

# 定义路径
raw_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_raw.h5ad"
annotated_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated.h5ad"
fig_dir = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/analysis_figures"

# 确保输出和图像目录存在
os.makedirs(fig_dir, exist_ok=True)

# 加载数据
print("加载数据...")
raw = sc.read_h5ad(raw_path)
annotated = sc.read_h5ad(annotated_path)

print(f"原始数据形状: {raw.shape}")
print(f"注释数据形状: {annotated.shape}")

# 1. 基因重叠分析
print("\n1. 分析基因重叠...")
genes_raw = set(raw.var_names)
genes_annotated = set(annotated.var_names)
genes_common = genes_raw.intersection(genes_annotated)

print(f"原始数据中的基因数: {len(genes_raw)}")
print(f"注释数据中的基因数: {len(genes_annotated)}")
print(f"两者共有的基因数: {len(genes_common)}")
print(f"筛选掉的基因数: {len(genes_raw) - len(genes_annotated)}")

# 绘制基因重叠图
print("绘制基因重叠图...")
plt.figure(figsize=(10, 6))
venn_labels = {'001': len(genes_annotated - genes_raw), 
               '010': len(genes_raw - genes_annotated), 
               '011': len(genes_common)}
from matplotlib_venn import venn2
v = venn2(subsets=venn_labels, set_labels=('Annotated Genes', 'Raw Genes'))
plt.title('Gene Overlap Between Raw and Annotated Data')
plt.savefig(f"{fig_dir}/gene_overlap.png")
plt.close()

# 2. 聚类和细胞类型分析
print("\n2. 分析聚类和细胞类型...")
cluster_counts = annotated.obs['leiden'].value_counts().sort_index()
celltype_counts = annotated.obs['CellType'].value_counts().sort_index()

print("聚类分布:")
print(cluster_counts)
print("\n细胞类型分布:")
print(celltype_counts)

# 绘制聚类和细胞类型分布图
print("绘制聚类和细胞类型分布图...")
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax[0])
ax[0].set_title('Leiden Cluster Distribution')
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Count')

sns.barplot(x=celltype_counts.index, y=celltype_counts.values, ax=ax[1])
ax[1].set_title('Cell Type Distribution')
ax[1].set_xlabel('Cell Type')
ax[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f"{fig_dir}/cluster_celltype_distribution.png")
plt.close()

# 3. 可视化UMAP聚类和标记基因
print("\n3. 可视化UMAP聚类和标记基因...")
plt.figure(figsize=(12, 10))
sc.pl.umap(annotated, color='leiden', legend_loc='on data', 
           legend_fontsize=10, title='Clustering on UMAP')
plt.savefig(f"{fig_dir}/umap_leiden_detailed.png")
plt.close()

# 获取每个聚类的top标记基因
print("获取每个聚类的标记基因...")
sc.tl.rank_genes_groups(annotated, 'leiden', method='wilcoxon')
top_markers = {}
for i in range(len(annotated.obs['leiden'].cat.categories)):
    cluster = str(i)
    genes = pd.DataFrame(annotated.uns['rank_genes_groups']['names'][cluster][0:5])
    top_markers[cluster] = genes[0].tolist()

print("每个聚类的Top 5标记基因:")
for cluster, genes in top_markers.items():
    print(f"Cluster {cluster}: {', '.join(genes)}")

# 4. 可视化一些重要标记基因的表达
print("\n4. 可视化标记基因表达...")
marker_genes_to_plot = [gene for genes in list(top_markers.values())[:5] for gene in genes[:2]]
if len(marker_genes_to_plot) > 0:
    marker_genes_to_plot = list(set(marker_genes_to_plot))[:9]  # 去重后最多取9个
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    for i, gene in enumerate(marker_genes_to_plot):
        if i < len(axs) and gene in annotated.var_names:
            sc.pl.umap(annotated, color=gene, ax=axs[i], show=False, 
                       title=f'{gene} Expression')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/marker_genes_expression.png")
    plt.close()

# 5. 分析高变基因
print("\n5. 分析高变基因...")
if 'highly_variable' in annotated.var.columns:
    n_hvg = annotated.var['highly_variable'].sum()
    print(f"高变基因数量: {n_hvg}")
    
    plt.figure(figsize=(12, 8))
    sc.pl.highly_variable_genes(annotated)
    plt.savefig(f"{fig_dir}/highly_variable_genes_analysis.png")
    plt.close()

# 6. 预处理前后基因表达分布比较
print("\n6. 基因表达分布比较...")
# 选择两个数据集中都存在的基因
common_genes = list(genes_common)[:5]  # 取前几个进行示例

if len(common_genes) > 0:
    # 为了便于比较，我们获取相同的细胞
    common_cells = list(set(raw.obs_names).intersection(set(annotated.obs_names)))
    
    if len(common_cells) > 0:
        # 筛选原始和注释数据中相同的细胞和基因
        raw_subset = raw[common_cells, common_genes].X
        annotated_subset = annotated[common_cells, common_genes].X
        
        # 转换为密集数组以便绘图
        if hasattr(raw_subset, 'toarray'):
            raw_subset = raw_subset.toarray()
        if hasattr(annotated_subset, 'toarray'):
            annotated_subset = annotated_subset.toarray()
        
        # 绘制基因表达分布比较
        fig, axs = plt.subplots(len(common_genes), 2, figsize=(12, 4 * len(common_genes)))
        
        for i, gene in enumerate(common_genes):
            # 原始数据表达分布
            sns.histplot(raw_subset[:, i], kde=True, ax=axs[i, 0])
            axs[i, 0].set_title(f'{gene} Expression (Raw)')
            axs[i, 0].set_xlabel('Expression')
            axs[i, 0].set_ylabel('Frequency')
            
            # 处理后数据表达分布
            sns.histplot(annotated_subset[:, i], kde=True, ax=axs[i, 1])
            axs[i, 1].set_title(f'{gene} Expression (Annotated)')
            axs[i, 1].set_xlabel('Expression')
            axs[i, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/gene_expression_comparison.png")
        plt.close()

# 7. 细胞间相似性分析
print("\n7. 细胞间相似性分析...")
# 使用PCA和UMAP结果
if 'X_pca' in annotated.obsm and 'X_umap' in annotated.obsm:
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    # 使用PCA的前两个分量
    scatter1 = axs[0].scatter(annotated.obsm['X_pca'][:, 0], annotated.obsm['X_pca'][:, 1], 
                             c=annotated.obs['leiden'].astype('category').cat.codes, 
                             s=10, alpha=0.7, cmap='tab20')
    axs[0].set_title('PCA Projection')
    axs[0].set_xlabel('PC1')
    axs[0].set_ylabel('PC2')
    
    # 使用UMAP结果
    scatter2 = axs[1].scatter(annotated.obsm['X_umap'][:, 0], annotated.obsm['X_umap'][:, 1], 
                             c=annotated.obs['leiden'].astype('category').cat.codes, 
                             s=10, alpha=0.7, cmap='tab20')
    axs[1].set_title('UMAP Projection')
    axs[1].set_xlabel('UMAP1')
    axs[1].set_ylabel('UMAP2')
    
    # 添加聚类标签
    for cluster in annotated.obs['leiden'].cat.categories:
        cluster_idx = annotated.obs['leiden'] == cluster
        center_x_pca = np.mean(annotated.obsm['X_pca'][cluster_idx, 0])
        center_y_pca = np.mean(annotated.obsm['X_pca'][cluster_idx, 1])
        axs[0].text(center_x_pca, center_y_pca, cluster, fontsize=12, weight='bold')
        
        center_x_umap = np.mean(annotated.obsm['X_umap'][cluster_idx, 0])
        center_y_umap = np.mean(annotated.obsm['X_umap'][cluster_idx, 1])
        axs[1].text(center_x_umap, center_y_umap, cluster, fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/cell_similarity.png")
    plt.close()

print(f"\n分析完成！图形已保存到: {fig_dir}") 