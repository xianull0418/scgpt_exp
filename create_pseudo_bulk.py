# 导入必要的库
import numpy as np # NumPy 用于数值计算，例如创建和操作数组
import pandas as pd # Pandas 用于数据处理和分析，例如创建和操作 DataFrame
import scanpy as sc # Scanpy 用于单细胞数据分析
import random # Random 用于生成随机数，例如随机抽样细胞
from typing import Optional, List, Union # Typing 用于类型提示，增强代码可读性和健壮性

# 定义生成伪批量样本的函数
def generate_pseudo_bulk_samples(
    adata_filtered: sc.AnnData, # 经过筛选的 AnnData 对象，包含单细胞基因表达数据
    cell_type_col: str, # adata_filtered.obs 中细胞类型注释的列名
    n_pseudo_bulk_samples: int = 1000, # 要生成的伪批量样本数量
    n_cells_per_sample_min: int = 50, # 每个伪批量样本的最小细胞数
    n_cells_per_sample_max: int = 200, # 每个伪批量样本的最大细胞数
    aggregation_method: str = "sum" # 聚合基因表达的方法 ("sum" 或 "mean")
) -> Optional[sc.AnnData]: # 返回一个新的 AnnData 对象，包含伪批量数据；如果出错则返回 None
    """
    从单细胞 AnnData 对象生成伪批量样本。

    Args:
        adata_filtered (sc.AnnData): 经过筛选的 AnnData 对象，包含单细胞基因表达数据。
        cell_type_col (str): adata_filtered.obs 中细胞类型注释的列名。
        n_pseudo_bulk_samples (int): 要生成的伪批量样本数量。
        n_cells_per_sample_min (int): 每个伪批量样本的最小细胞数。
        n_cells_per_sample_max (int): 每个伪批量样本的最大细胞数。
        aggregation_method (str): 聚合基因表达的方法，可选 "sum"（求和）或 "mean"（均值）。

    Returns:
        Optional[sc.AnnData]: 一个新的 AnnData 对象，其中 .X 存储伪批量基因表达，
                              .obs 存储每个样本中细胞类型的真实比例。如果发生错误，则返回 None。
    """
    # 打印开始信息和参数
    print(f"开始生成伪批量样本，共 {n_pseudo_bulk_samples} 个样本...")
    print(f"参数: min_cells={n_cells_per_sample_min}, max_cells={n_cells_per_sample_max}, aggregation='{aggregation_method}'")

    # 验证 cell_type_col 是否存在于 adata_filtered.obs 中
    if cell_type_col not in adata_filtered.obs.columns:
        print(f"错误: 在 adata_filtered.obs 中未找到细胞类型列 '{cell_type_col}'。可用列为: {list(adata_filtered.obs.columns)}")
        return None

    # 验证 aggregation_method 是否有效
    if aggregation_method not in ["sum", "mean"]:
        print(f"错误: 未知的 aggregation_method '{aggregation_method}'。请选择 'sum' 或 'mean'。")
        return None

    # 确保 n_cells_per_sample_max 不小于 n_cells_per_sample_min
    if n_cells_per_sample_max < n_cells_per_sample_min:
        print(f"错误: n_cells_per_sample_max ({n_cells_per_sample_max}) 不能小于 n_cells_per_sample_min ({n_cells_per_sample_min})。")
        return None
        
    # 确保有足够的细胞可供采样
    if adata_filtered.n_obs < n_cells_per_sample_min:
        print(f"错误: adata_filtered 中的细胞数 ({adata_filtered.n_obs}) 不足以采样 {n_cells_per_sample_min} 个细胞/样本。")
        return None

    # 获取所有基因名称列表
    all_gene_names: List[str] = list(adata_filtered.var_names)
    # 获取所有唯一的细胞类型，并排序
    unique_cell_types: List[str] = sorted(list(adata_filtered.obs[cell_type_col].astype('category').cat.categories))
    
    # 如果未找到唯一的细胞类型，则报错返回
    if not unique_cell_types:
        print(f"错误: 在列 '{cell_type_col}' 中未找到唯一的细胞类型。请检查列内容。")
        return None
    print(f"找到 {len(unique_cell_types)} 种独特细胞类型: {unique_cell_types}")

    # 初始化用于存储伪批量表达数据的列表
    pseudo_bulk_expressions: List[np.ndarray] = []
    # 初始化用于存储伪批量细胞类型比例的列表
    pseudo_bulk_fractions: List[np.ndarray] = []

    # 判断输入数据是否为稀疏矩阵
    is_sparse = isinstance(adata_filtered.X, (sc.sparse.csr_matrix, sc.sparse.csc_matrix))
    if is_sparse:
        print("输入 AnnData.X 是稀疏矩阵。将使用稀疏矩阵操作。")
    else:
        print("输入 AnnData.X 是密集矩阵。")


    # 循环生成每个伪批量样本
    for i in range(n_pseudo_bulk_samples):
        # 确定当前样本的细胞数量
        if n_cells_per_sample_min == n_cells_per_sample_max:
            n_cells_current_sample = n_cells_per_sample_min # 如果最大最小相等，则固定数量
        else:
            # 否则在最小和最大细胞数之间随机选择一个整数
            n_cells_current_sample = random.randint(n_cells_per_sample_min, n_cells_per_sample_max)
        
        # 从所有细胞中随机抽取指定数量的细胞索引（无放回抽样）
        sampled_indices = random.sample(range(adata_filtered.n_obs), k=n_cells_current_sample)
        
        # 提取抽样细胞的基因表达数据
        # 这将是原始矩阵（稀疏或密集）的一个切片
        sampled_X = adata_filtered.X[sampled_indices, :]
        
        # 根据指定的聚合方法聚合基因表达
        if aggregation_method == "sum":
            aggregated_expression = sampled_X.sum(axis=0) # 按列求和
        else: # aggregation_method == "mean"
            aggregated_expression = sampled_X.mean(axis=0) # 按列求均值
        
        # 对于稀疏矩阵，sum/mean 可能返回一个二维矩阵 (1, n_genes)，需要转换为一维数组
        if hasattr(aggregated_expression, "A1"): # 处理 numpy.matrix 对象
            aggregated_expression = aggregated_expression.A1
        elif hasattr(aggregated_expression, "toarray"): # 处理某些 A1 属性无法覆盖的稀疏结果
             aggregated_expression = aggregated_expression.toarray().flatten()
        
        # 将聚合后的表达数据添加到列表中
        pseudo_bulk_expressions.append(aggregated_expression)

        # 计算细胞类型比例
        # 获取抽样细胞的细胞类型注释
        sampled_cell_types_series = adata_filtered.obs[cell_type_col].iloc[sampled_indices]
        # 计算每种细胞类型的数量，并确保所有 unique_cell_types 都包含在内（缺失的填充为0）
        type_counts = sampled_cell_types_series.value_counts().reindex(unique_cell_types, fill_value=0.0)
        # 计算每种细胞类型的比例
        fractions = type_counts / n_cells_current_sample
        # 将比例（NumPy 数组）添加到列表中
        pseudo_bulk_fractions.append(fractions.values)

        # 打印进度，大约打印10次
        if (i + 1) % (n_pseudo_bulk_samples // 10 if n_pseudo_bulk_samples >=10 else 1) == 0 :
            print(f"已生成 {i + 1}/{n_pseudo_bulk_samples} 个伪批量样本...")

    # 将列表转换为最终格式
    try:
        # 将伪批量表达数据列表转换为 NumPy 数组
        pseudo_bulk_X_np = np.array(pseudo_bulk_expressions)
    except Exception as e:
        print(f"错误：将 pseudo_bulk_expressions 转换为 NumPy 数组失败: {e}")
        # 如果基因数量不一致（理论上不应发生），可以取消注释以下代码进行检查
        # for idx, arr in enumerate(pseudo_bulk_expressions): print(f"表达样本 {idx} 的形状: {arr.shape}")
        return None

    # 将细胞类型比例列表转换为 Pandas DataFrame
    fractions_df = pd.DataFrame(pseudo_bulk_fractions, columns=unique_cell_types)

    # 为伪批量数据创建新的 AnnData 对象
    try:
        # X 存储聚合表达，obs 存储细胞类型比例，var 存储基因名称
        pseudo_bulk_adata = sc.AnnData(X=pseudo_bulk_X_np, obs=fractions_df, var=pd.DataFrame(index=all_gene_names))
        # pseudo_bulk_adata.var_names = all_gene_names # 如果构造函数中未设置 var，这是另一种设置基因名的方法
    except Exception as e:
        print(f"错误：创建最终伪批量数据 AnnData 对象失败: {e}")
        return None

    # 打印成功信息和结果摘要
    print("\n伪批量 AnnData 对象创建成功。")
    print(f"伪批量 X 的形状 (样本数 x 基因数): {pseudo_bulk_adata.X.shape}")
    print(f"伪批量 obs (细胞类型比例) 的形状: {pseudo_bulk_adata.obs.shape}")
    print(f"伪批量 obs 中的列 (细胞类型): {list(pseudo_bulk_adata.obs.columns)}")
    
    # 返回创建的伪批量 AnnData 对象
    return pseudo_bulk_adata

# 当脚本作为主程序执行时运行以下代码
if __name__ == '__main__':
    # 打印示例用法开始的提示信息
    print("--- generate_pseudo_bulk_samples 函数使用示例 ---")

    # 这是一个如何使用该函数的示例。
    # 首先，您需要加载并准备您的单细胞数据。
    # 例如，使用先前脚本中的 'load_and_prepare_data' 函数。
    
    # --- 占位符：创建一个用于演示的虚拟 filtered_adata ---
    print("\n正在创建用于演示的占位符 filtered_adata...")
    # 定义示例细胞数和基因数
    n_example_cells, n_example_genes = 1000, 500
    # 生成符合泊松分布的模拟基因表达计数数据（更接近真实情况）
    example_X_data = np.random.poisson(2, size=(n_example_cells, n_example_genes)).astype(np.float32)
    
    # 确保细胞类型是字符串并且是类别类型，以便正确处理
    cell_type_categories = ['TypeA', 'TypeB', 'TypeC', 'TypeD'] # 定义示例细胞类型
    # 创建包含细胞类型注释的 Pandas DataFrame
    example_obs_data = pd.DataFrame({
        'celltype': pd.Categorical(np.random.choice(cell_type_categories, size=n_example_cells)) # 随机分配细胞类型
    })
    # 创建包含基因名称的 Pandas DataFrame作为 .var
    example_var_data = pd.DataFrame(index=[f'gene_{j}' for j in range(n_example_genes)])
    
    # 使用模拟数据创建 AnnData 对象
    filtered_adata_placeholder = sc.AnnData(X=example_X_data, obs=example_obs_data, var=example_var_data)
    # 打印占位符 AnnData 的信息
    print(f"占位符 filtered_adata 创建完毕，形状为: ({filtered_adata_placeholder.n_obs}, {filtered_adata_placeholder.n_vars})")
    print(f"占位符中的细胞类型分布: {filtered_adata_placeholder.obs['celltype'].value_counts().to_dict()}")
    # --- 占位符结束 ---

    # 检查占位符 filtered_adata_placeholder 是否已成功创建
    if 'filtered_adata_placeholder' in locals():
        # --- 定义伪批量样本生成的参数 ---
        cell_type_column = "celltype"      # .obs 中包含细胞类型标签的列名
        num_pseudo_bulk_samples = 50       # 要创建的伪批量样本数量
        min_cells_per_sample = 20          # 每个样本的最小细胞数
        max_cells_per_sample = 100         # 每个样本的最大细胞数
        agg_method = "sum"                 # 聚合方法: "sum" 或 "mean"
        # --- 参数定义结束 ---
        
        print("\n开始使用占位符数据生成伪批量样本...")
        # 调用核心函数生成伪批量数据集
        pseudo_bulk_dataset = generate_pseudo_bulk_samples(
            adata_filtered=filtered_adata_placeholder,  # 输入的单细胞数据
            cell_type_col=cell_type_column,            # 细胞类型列名
            n_pseudo_bulk_samples=num_pseudo_bulk_samples, # 伪批量样本数量
            n_cells_per_sample_min=min_cells_per_sample, # 每样本最小细胞数
            n_cells_per_sample_max=max_cells_per_sample, # 每样本最大细胞数
            aggregation_method=agg_method              # 表达聚合方法
        )

        # 检查伪批量数据集是否成功生成
        if pseudo_bulk_dataset:
            print("\n使用占位符数据成功生成伪批量样本！")
            print(f"生成的伪批量 AnnData 形状: {pseudo_bulk_dataset.shape}")
            print("细胞类型比例数据 (pseudo_bulk_dataset.obs) 的前5行:")
            print(pseudo_bulk_dataset.obs.head())
            print("\n前3个伪批量样本的前5个基因的表达值 (pseudo_bulk_dataset.X):")
            print(pseudo_bulk_dataset.X[:3, :5])
        else:
            print("\n使用占位符数据生成伪批量样本失败。")
    else:
        # 如果占位符数据未加载，则打印跳过信息
        print("\n跳过伪批量生成示例，因为占位符数据未加载。")
        print("要运行此示例，请确保占位符数据加载部分已激活，或与实际数据集成。")

    # 打印示例用法结束的提示信息
    print("\n--- 示例用法结束 ---")
    # 提示用户如何有效使用此脚本
    print("\n提醒：要有效地使用此脚本，请将其与您的实际数据加载和准备流程集成。")
    print("您将需要提供一个加载了真实单细胞数据的实际 'filtered_adata' 对象。")
```
