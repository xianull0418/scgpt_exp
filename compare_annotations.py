#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 设置全局参数
sc.settings.verbosity = 3  # 显示详细信息
sc.settings.set_figure_params(dpi=100, facecolor='white')

# 定义路径
original_annotated_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated.h5ad"
matched_annotated_path = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/pbmc3k_annotated_matched.h5ad"
fig_dir = "/ai/home/jcw/scGPT_new/finetune/data/TAPE/comparison_figures"

# 确保输出图像目录存在
os.makedirs(fig_dir, exist_ok=True)

# 加载数据
print("加载数据...")
original = sc.read_h5ad(original_annotated_path)
matched = sc.read_h5ad(matched_annotated_path)

print(f"原始注释数据形状: {original.shape}")
print(f"匹配注释数据形状: {matched.shape}")

# 1. 基本结构比较
print("\n1. 基本结构比较")
print("原始注释数据:")
print(original)
print("\n匹配注释数据:")
print(matched)

# 2. 比较细胞和基因重叠
print("\n2. 细胞和基因重叠比较")
cells_original = set(original.obs_names)
cells_matched = set(matched.obs_names)
genes_original = set(original.var_names)
genes_matched = set(matched.var_names)

cells_common = cells_original.intersection(cells_matched)
genes_common = genes_original.intersection(genes_matched)

print(f"原始注释数据细胞数: {len(cells_original)}")
print(f"匹配注释数据细胞数: {len(cells_matched)}")
print(f"共有细胞数: {len(cells_common)}")
print(f"细胞重叠比例: {len(cells_common)/max(len(cells_original), len(cells_matched)):.2%}")

print(f"原始注释数据基因数: {len(genes_original)}")
print(f"匹配注释数据基因数: {len(genes_matched)}")
print(f"共有基因数: {len(genes_common)}")
print(f"基因重叠比例: {len(genes_common)/max(len(genes_original), len(genes_matched)):.2%}")

# 3. 聚类结果比较
print("\n3. 聚类结果比较")
if len(cells_common) > 0:
    # 提取共有细胞的聚类结果
    common_cells = list(cells_common)
    original_clusters = original[common_cells].obs['leiden'].astype('category').cat.codes.values
    matched_clusters = matched[common_cells].obs['leiden'].astype('category').cat.codes.values
    
    # 计算聚类一致性指标
    ari = adjusted_rand_score(original_clusters, matched_clusters)
    nmi = normalized_mutual_info_score(original_clusters, matched_clusters)
    
    print(f"聚类调整兰德指数 (ARI): {ari:.4f} (范围: -1到1, 1表示完全匹配)")
    print(f"聚类标准化互信息 (NMI): {nmi:.4f} (范围: 0到1, 1表示完全匹配)")
    
    # 绘制混淆矩阵
    confusion_mat = pd.crosstab(
        pd.Series(original[common_cells].obs['leiden'], name='原始注释'),
        pd.Series(matched[common_cells].obs['leiden'], name='匹配注释')
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('聚类结果混淆矩阵')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/cluster_confusion_matrix.png")
    plt.close()
    
    # 绘制每个数据集的聚类分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始注释数据聚类分布
    original_counts = original.obs['leiden'].value_counts().sort_index()
    sns.barplot(x=original_counts.index, y=original_counts.values, ax=ax1)
    ax1.set_title('原始注释数据聚类分布')
    ax1.set_xlabel('聚类')
    ax1.set_ylabel('细胞数量')
    
    # 匹配注释数据聚类分布
    matched_counts = matched.obs['leiden'].value_counts().sort_index()
    sns.barplot(x=matched_counts.index, y=matched_counts.values, ax=ax2)
    ax2.set_title('匹配注释数据聚类分布')
    ax2.set_xlabel('聚类')
    ax2.set_ylabel('细胞数量')
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/cluster_distributions.png")
    plt.close()

# 4. UMAP可视化比较
print("\n4. UMAP可视化比较")
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 原始注释数据UMAP
    sc.pl.umap(original, color='leiden', ax=ax1, show=False, title='原始注释数据UMAP')
    
    # 匹配注释数据UMAP
    sc.pl.umap(matched, color='leiden', ax=ax2, show=False, title='匹配注释数据UMAP')
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/umap_comparison.png")
    plt.close()
    print("生成UMAP比较图")
except Exception as e:
    print(f"生成UMAP比较图失败: {e}")

# 5. 基因表达比较 (对共有细胞和基因)
print("\n5. 基因表达比较")
if len(cells_common) > 0 and len(genes_common) > 0:
    # 抽样一些共有基因进行比较
    common_genes = sorted(list(genes_common))
    sample_genes = common_genes[:min(5, len(common_genes))]  # 最多取前5个基因
    
    # 使用共有细胞的数据
    common_cells = sorted(list(cells_common))
    sample_cells = common_cells[:min(100, len(common_cells))]  # 最多取前100个细胞
    
    # 提取原始和匹配数据的子集
    original_subset = original[sample_cells, sample_genes].X
    matched_subset = matched[sample_cells, sample_genes].X
    
    # 转换为密集数组以便绘图
    if hasattr(original_subset, 'toarray'):
        original_subset = original_subset.toarray()
    if hasattr(matched_subset, 'toarray'):
        matched_subset = matched_subset.toarray()
    
    # 计算每个基因表达的相关系数
    correlations = {}
    for i, gene in enumerate(sample_genes):
        corr = np.corrcoef(original_subset[:, i], matched_subset[:, i])[0, 1]
        correlations[gene] = corr
    
    print("基因表达相关系数:")
    for gene, corr in correlations.items():
        print(f"{gene}: {corr:.4f}")
    
    # 绘制基因表达的散点图比较
    fig, axes = plt.subplots(len(sample_genes), 1, figsize=(10, 5*len(sample_genes)))
    if len(sample_genes) == 1:
        axes = [axes]
    
    for i, gene in enumerate(sample_genes):
        ax = axes[i]
        ax.scatter(original_subset[:, i], matched_subset[:, i], alpha=0.5)
        ax.set_title(f'基因 {gene} 表达比较 (相关系数: {correlations[gene]:.4f})')
        ax.set_xlabel('原始注释数据表达')
        ax.set_ylabel('匹配注释数据表达')
        # 添加对角线
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gene_expression_comparison.png")
    plt.close()

# 6. 标记基因比较
print("\n6. 标记基因比较")
try:
    # 对原始注释数据计算标记基因
    sc.tl.rank_genes_groups(original, 'leiden', method='wilcoxon')
    
    # 已经在匹配注释数据中计算了标记基因
    
    # 比较每个聚类的前5个标记基因
    cluster_markers_original = {}
    cluster_markers_matched = {}
    
    for cluster in original.obs['leiden'].cat.categories:
        try:
            markers_original = pd.DataFrame(original.uns['rank_genes_groups']['names'][cluster])
            cluster_markers_original[cluster] = set(markers_original[0][:5].tolist())  # 前5个标记基因
        except:
            cluster_markers_original[cluster] = set()
    
    for cluster in matched.obs['leiden'].cat.categories:
        try:
            markers_matched = pd.DataFrame(matched.uns['rank_genes_groups']['names'][cluster])
            cluster_markers_matched[cluster] = set(markers_matched[0][:5].tolist())  # 前5个标记基因
        except:
            cluster_markers_matched[cluster] = set()
    
    # 计算每个聚类标记基因的重叠
    print("聚类标记基因重叠:")
    cluster_overlaps = {}
    for cluster in set(cluster_markers_original.keys()) & set(cluster_markers_matched.keys()):
        markers_original = cluster_markers_original[cluster]
        markers_matched = cluster_markers_matched[cluster]
        overlap = markers_original & markers_matched
        overlap_percent = len(overlap) / max(len(markers_original), len(markers_matched)) if max(len(markers_original), len(markers_matched)) > 0 else 0
        cluster_overlaps[cluster] = overlap_percent
        print(f"聚类 {cluster}: 重叠率 {overlap_percent:.2%}, 共有标记基因: {', '.join(overlap) if overlap else '无'}")
    
    # 绘制标记基因重叠条形图
    plt.figure(figsize=(12, 6))
    clusters = sorted(cluster_overlaps.keys())
    overlaps = [cluster_overlaps[c] for c in clusters]
    plt.bar(clusters, overlaps)
    plt.axhline(y=0.6, color='r', linestyle='-', alpha=0.3)  # 60%的重叠率参考线
    plt.ylim(0, 1.0)
    plt.title('各聚类标记基因重叠率')
    plt.xlabel('聚类')
    plt.ylabel('标记基因重叠率')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/marker_gene_overlap.png")
    plt.close()
except Exception as e:
    print(f"标记基因比较失败: {e}")

print(f"\n比较分析完成！图形已保存到: {fig_dir}") 