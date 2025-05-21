# 导入 scanpy 库，用于处理 AnnData 对象
import scanpy as sc
# 导入 numpy 库，用于数值计算，特别是数组操作
import numpy as np
# 导入 pandas 库，主要用于示例数据生成
import pandas as pd 
# 从 typing 模块导入 Optional，用于类型提示，表示参数可以是可选的
from typing import Optional

# 定义为 scGPT 预处理伪批量数据的函数
def preprocess_pseudo_bulk_for_scgpt(
    pseudo_bulk_adata: sc.AnnData, # 包含伪批量基因表达的 AnnData 对象
    target_sum: Optional[float] = 1e4, # 计数标准化的目标总和 (例如，1e4 表示 CPM)。如果为 None，则跳过标准化。
    n_bins: int = 51, # 用于离散化表达值的 bin 的数量 (例如，scGPT 使用 51)。
    hvg_col_name: Optional[str] = None # .var 中标记高变基因 (HVG) 的列名。如果提供，则筛选这些基因。
) -> Optional[sc.AnnData]: # 返回处理后的 AnnData 对象 (包含分箱表达) 或在发生严重错误时返回 None。
    """
    为 scGPT 输入预处理伪批量 AnnData。
    执行可选的 HVG 筛选、标准化、log转换和基因表达值分箱。

    Args:
        pseudo_bulk_adata (sc.AnnData): 包含伪批量基因表达的 AnnData 对象。
        target_sum (Optional[float]): 计数标准化的目标总和 (例如，1e4 代表 CPM)。
                                      如果为 None，则跳过标准化。
        n_bins (int): 用于离散化表达值的 bin 的数量 (例如，scGPT 为 51)。
        hvg_col_name (Optional[str]): .var 中标记高变基因的列的名称。
                                      如果提供，则筛选这些基因。

    Returns:
        Optional[sc.AnnData]: 处理后的 AnnData 对象，其中包含分箱后的表达数据。
                              如果发生严重错误，则返回 None。
    """
    print("开始预处理伪批量数据...")

    # 创建一个副本，以避免意外修改原始 AnnData 对象
    adata_processed = pseudo_bulk_adata.copy()
    print(f"已复制输入 AnnData。原始形状: {pseudo_bulk_adata.shape}, 复制后形状: {adata_processed.shape}")

    # 可选：高变基因 (HVG) 筛选
    if hvg_col_name: # 如果提供了 HVG 列名
        if hvg_col_name in adata_processed.var.columns: # 检查列是否存在于 .var 中
            if adata_processed.var[hvg_col_name].dtype == bool: # 检查 HVG 列是否为布尔类型
                print(f"基于布尔列 '{hvg_col_name}' 筛选基因。")
                # 根据 HVG 列筛选 AnnData 对象 (仅保留 HVG 为 True 的基因)
                adata_processed = adata_processed[:, adata_processed.var[hvg_col_name]].copy()
                print(f"HVG 筛选后的基因数量: {adata_processed.n_vars}")
                if adata_processed.n_vars == 0: # 如果筛选后没有基因剩下
                    print(f"警告: 使用列 '{hvg_col_name}' 进行 HVG 筛选后没有剩余基因。请检查您的 HVG 数据。")
                    return None # 或进行适当处理
            else: # 如果 HVG 列不是布尔类型
                print(f"警告: HVG 列 '{hvg_col_name}' 不是布尔类型。跳过 HVG 筛选。")
        else: # 如果 HVG 列不存在
            print(f"警告: 在 .var 中未找到 HVG 列 '{hvg_col_name}'。跳过 HVG 筛选。")

    # 标准化 (Normalization)
    if target_sum is not None: # 如果指定了目标总和
        if adata_processed.X is None: # 检查 .X 是否存在
            print("错误: adata_processed.X 为 None。无法执行标准化。")
            return None
        print(f"将每个样本的总计数标准化为 {target_sum}...")
        # 使用 scanpy 的 normalize_total 函数进行标准化
        sc.pp.normalize_total(adata_processed, target_sum=target_sum)
        # 如果需要进一步直接使用，可以将标准化后的数据存储在一个层中，
        # 尽管 sc.pp.log1p 将对 adata_processed.X (现在是标准化的) 进行操作。
        adata_processed.layers['normalized'] = adata_processed.X.copy() 
    else: # 如果未指定目标总和
        print("由于 target_sum 为 None，跳过总计数标准化。")
        # 如果不进行标准化，请确保在 log1p 之前 .X 不为 None
        if adata_processed.X is None:
            print("错误: adata_processed.X 为 None 且 target_sum 为 None。无法继续。")
            return None

    # Log1p 转换
    print("正在应用 log1p 转换...")
    if target_sum is not None: # 如果进行了标准化
        # 如果已标准化，.X 已被 normalize_total 更新。sc.pp.log1p 会就地修改 .X
        sc.pp.log1p(adata_processed) 
        # 将 log1p 转换后的数据存储在 .layers['log1p'] 中
        adata_processed.layers['log1p'] = adata_processed.X.copy()
    else: # 如果未进行标准化
        # 如果未标准化，则对原始 .X 应用 log1p 并存储在 layers['log1p'] 中
        # 同时更新 .X 为此 log1p 数据以保持一致性
        adata_processed.layers['log1p'] = np.log1p(adata_processed.X)
        adata_processed.X = adata_processed.layers['log1p'].copy()
    print("Log1p 转换完成。数据存储在 adata_processed.layers['log1p'] 和 adata_processed.X 中")

    # 对数转换后的表达值分箱 (Binning)
    print(f"将对数转换后的表达值分箱为 {n_bins} 个 bin...")
    # 获取用于分箱的数据 (来自 .layers['log1p'])
    data_to_bin = adata_processed.layers['log1p']
    
    if data_to_bin is None: # 检查数据是否存在
        print("错误: data_to_bin (来自 layers['log1p']) 为 None。无法执行分箱。")
        return None

    # 从数据本身确定用于分箱的最小值和最大值
    # 筛选出非有限值 (例如 NaN, inf) 以进行 min/max 计算
    finite_data = data_to_bin[np.isfinite(data_to_bin)]
    if finite_data.size == 0: # 如果没有有限数据
        print("错误: 没有可用于分箱的有限数据。请检查输入数据。")
        return None

    min_val = np.min(finite_data) # 计算最小值
    max_val = np.max(finite_data) # 计算最大值
    print(f"用于分箱的数据范围 (有限值): min={min_val:.4f}, max={max_val:.4f}")

    if min_val == max_val: # 处理所有值都相同的情况
        print("警告: 在筛选有限值后，data_to_bin 中的所有值都相同。将所有值分配到 bin 0。")
        # 所有值都将落入第一个 bin (索引 0)
        binned_expression = np.zeros(data_to_bin.shape, dtype=int)
    else:
        # 创建 n_bins 个区间 (对于 np.linspace 需要 n_bins+1 个边界点)
        # np.digitize 需要 n_bins 个边界点来产生 n_bins+1 个可能的输出值 (0 到 n_bins)
        # 或者，需要 n_bins-1 个边界点来产生 n_bins 个值 (如果值 < edges[0] 映射到 0，则为 0 到 n_bins-1)
        # 示例代码使用 bins[:-1] 和 np.digitize，表示 n_bins-1 个边界点。
        # bins = np.linspace(min_val, max_val, num=n_bins) # 这会创建 n_bins 个点，即 n_bins-1 个区间。
                                                        # 如果用作 np.digitize 的边界，则会产生 n_bins 个类别。
        
        # 我们将遵循提示中示例的逻辑：
        # bins = np.linspace(min_val, max_val, num=n_bins) # 这些是 n_bins 个 bin 的中心点/代表点
        # 如果 n_bins=51，则给出 51 个点。
        # 对于 np.digitize，我们需要边界。如果我们想要 n_bins 个不同的整数输出 (0 到 n_bins-1)，
        # 我们需要 n_bins-1 个边界。
        # 示例：bins = [0,1,2,3,4]，n_bins=5。边界 = [0.5, 1.5, 2.5, 3.5] (n_bins-1 个边界)
        # 值 < 0.5 -> 0
        # 0.5 <= 值 < 1.5 -> 1
        # ...
        # 值 >= 3.5 -> 4
        
        # 示例中 np.digitize(data_to_bin, bins[:-1], right=False) 及后续的 clip 逻辑：
        # 来自 np.linspace(min_val, max_val, num=n_bins) 的 `bins` 有 `n_bins` 个元素。
        # `bins[:-1]` 有 `n_bins-1` 个元素 (边界)。
        # 具有 `n_bins-1` 个边界的 `np.digitize` 将返回从 `0` 到 `n_bins-1` 的值。
        # 值 < edges[0] -> 0
        # edges[0] <= 值 < edges[1] -> 1
        # ...
        # 值 >= edges[n_bins-2] (最后一个边界) -> n_bins-1
        # 这对于 0 到 n_bins-1 的输出似乎是正确的。
        # 示例中后续的 `clip(binned_expression - 1, 0, n_bins - 1)` 暗示 digitize 可能输出 1 到 n_bins。
        # 让我们重新验证 np.digitize：
        # 如果 bins = [b1, b2, b3]，digitize(x, bins) 给出：
        # x < b1 -> 0
        # b1 <= x < b2 -> 1
        # b2 <= x < b3 -> 2
        # x >= b3 -> 3
        # 因此，对于 N 个边界，它给出 N+1 个结果。
        # 如果我们想要 n_bins 个结果 (0 到 n_bins-1)，我们需要 n_bins-1 个边界。
        # 所以，`edges = np.linspace(min_val, max_val, num=n_bins -1)` 是错误的。
        # `edges = np.linspace(min_val, max_val, num=n_bins)` 会给出 n_bins 个点。
        # 如果这些是 np.digitize 的边界，`np.digitize(data, edges)` 会给出 0 到 n_bins。
        
        # 为安全起见，让我们使用提示中 np.digitize 部分的精确表述。
        # 它似乎意图让 linspace 生成的 `bins` 作为 bin 的上边界。
        # bin_edges = np.linspace(min_val, max_val, num=n_bins) # 这是 n_bins 个点。
                                                            # 如果用作边界，它们定义了 n_bins-1 个区间。
                                                            # 使用这些边界的 np.digitize 对于 值 < bin_edges[-1] 会给出索引 0 到 n_bins-1，
                                                            # 对于 值 >= bin_edges[-1] 会给出 n_bins。
        
        # 如果 bin_edges 有 K 个元素，np.digitize(X, bin_edges) 返回索引 i，使得 bin_edges[i-1] <= x < bin_edges[i]
        # 如果我们想要 n_bins 个类别 (0 到 n_bins-1)：
        # 我们需要 n_bins-1 个实际阈值来将空间划分为 n_bins 个区域。
        # 示例：n_bins=3。值 0, 1, 2。
        # 边界：e1, e2。 x < e1 -> 0； e1 <= x < e2 -> 1； x >= e2 -> 2。
        # 因此，np.linspace(min_val, max_val, num=n_bins -1) 会给出错误的边界数量。
        # 让我们使用 `num=n_bins` 进行 linspace 来定义 bin 的*上*边界，然后进行调整。
        
        # 创建 n_bins + 1 个边界来定义 n_bins 个区间。
        # bin_edges_for_digitize = np.linspace(min_val, max_val, num=n_bins + 1)
        
        # Digitize：如果值落在这些边界内，则将它们分配给 bin 索引 1 到 n_bins。
        # 值 < bin_edges_for_digitize[0] 为 0。值 >= bin_edges_for_digitize[-1] 为 n_bins+1。
        # 这不是示例中使用的方法。
        
        # 恢复到示例中分箱部分的直接逻辑：
        # `bins = np.linspace(min_val, max_val, num=n_bins)` -> 这是 n_bins 个点。
        # `binned_expression = np.digitize(data_to_bin, bins[:-1], right=False)`
        # `bins[:-1]` 有 `n_bins-1` 个元素。
        # 具有 `N` 个边界 (bins[:-1]) 的 `np.digitize` 返回从 `0` 到 `N` 的值。
        # 因此，`binned_expression` 的范围将是 `0` 到 `n_bins-1`。
        # 示例：n_bins=51。`bins` 有 51 个元素。`bins[:-1]` 有 50 个元素 (边界)。
        # `np.digitize` 将返回从 0 到 50 的值。这似乎是正确的。
        # 示例中后续的 clip `np.clip(binned_expression - 1, 0, n_bins - 1)` 可能是安全措施，
        # 或者最初基于对 np.digitize 输出的不同解释。
        # 如果 np.digitize(data, edges_len_N) 给出 0 到 N，那么它已经是 0 到 n_bins-1。
        
        # 让我们仔细测试 np.digitize 的行为：
        # x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
        # edges = np.array([1, 2, 3]) # 3 个边界
        # np.digitize(x, edges) -> array([0, 0, 1, 1, 2, 2, 3]) # 输出范围 0 到 len(edges)，即 0 到 3。
        # 因此，如果 `bins[:-1]` 有 `n_bins-1` 个边界，输出是 `0` 到 `n_bins-1`。这是期望的。
        # 示例中的 `clip(binned_expression -1 ...)` 会将其移位到 -1 到 n_bins-2。这是不正确的。
        # 提示中 clip 的代码是：`np.clip(binned_expression -1, 0, n_bins - 1)`
        # 如果 binned_expression 已经是 0 到 n_bins-1，那么 `binned_expression - 1` 是 -1 到 n_bins-2。
        # 将其裁剪到 (0, n_bins-1) 使其变为 0 到 n_bins-2。这实际上丢失了最后一个 bin。
        # 这一定是我对示例 clip 逻辑的误解。

        # 让我们假设目标是：结果在 [0, n_bins-1] 范围内。
        # 使用 `np.linspace(min_val, max_val, num=n_bins)` 创建 `n_bins` 个点。
        # 如果这些是 bin 的*上*边界 (不包括，对于 right=False)，那么 `bins[:-1]` 是前 `n_bins-1` 个上边界。
        # `np.digitize(data, bins[:-1], right=False)`：
        #   - 值 < `bins[0]` 映射到 bin 0。
        #   - `bins[0]` <= 值 < `bins[1]` 映射到 bin 1。
        #   ...
        #   - `bins[n_bins-2]` <= 值 映射到 bin `n_bins-1`。(因为 `bins[:-1]` 表示最后一个边界是 `bins[n_bins-2]`)
        # 这直接给出了 `0, ..., n_bins-1` 范围内的值。
        
        # 因此，如果使用 `bins[:-1]`，示例中的 `binned_expression - 1` 的 clip 可能是错误的。
        # 让我们使用更简单的解释：具有 `k` 个边界的 `np.digitize` 给出 `k+1` 个 bin (0 到 `k`)。
        # 如果 `bins = np.linspace(min_val, max_val, n_bins-1)`，这些是 `n_bins-1` 个边界。
        # `np.digitize(data_to_bin, bins)` 将产生从 0 到 `n_bins-1` 的值。这似乎是最简单的。

        # 创建 n_bins-1 个边界，这些边界定义了 n_bins 个区间
        bin_edges = np.linspace(min_val, max_val, num=n_bins - 1) 
        # 使用 np.digitize 进行分箱。right=False 表示区间是 [左闭, 右开)
        binned_expression = np.digitize(data_to_bin, bin_edges, right=False)
        # 具有 N 个边界的 np.digitize 返回从 0 到 N 的值。
        # 这里，N = n_bins-1。所以输出是 0 到 n_bins-1。这正是我们想要的。
        # 这种表述方式应该不需要裁剪。

    # 将分箱后的表达数据存储在 AnnData 对象的 .layers['binned_expression'] 中，并确保为整数类型
    adata_processed.layers['binned_expression'] = binned_expression.astype(np.int32)
    print(f"分箱完成。分箱数据存储在 adata_processed.layers['binned_expression'] 中")
    
    # 验证分箱数据的最小值/最大值
    if binned_expression.size > 0:
        print(f"最小 bin 索引: {np.min(binned_expression)}, 最大 bin 索引: {np.max(binned_expression)}")
        # 检查最大 bin 索引是否超出预期范围
        if np.max(binned_expression) >= n_bins:
            print(f"警告: 最大 bin 索引 ({np.max(binned_expression)}) >= n_bins ({n_bins})。请检查分箱逻辑。")
    else:
        print("警告: 分箱后的表达数据为空。")


    # 主表达矩阵 X 可以保持为 log1p 转换后的数据。
    # adata_processed.X 已经设置为 log1p 数据。

    print("预处理完成。")
    # 返回处理后的 AnnData 对象
    return adata_processed

# 当脚本作为主程序执行时
if __name__ == '__main__':
    print("--- preprocess_pseudo_bulk_for_scgpt 函数使用示例 ---")
    
    print("\n正在创建用于演示的占位符 pseudo_bulk_adata...")
    n_samples, n_genes_ps = 100, 50 # 定义样本数和基因数
    # 模拟伪批量的原始计数数据 (非负整数)
    example_X_ps = np.random.randint(0, 1000, size=(n_samples, n_genes_ps)).astype(float) 
    # 创建示例 .obs (样本元数据)
    example_obs_ps = pd.DataFrame({f'meta_{k}': np.random.rand(n_samples) for k in range(2)},
                                  index=[f"sample_{i}" for i in range(n_samples)])
    # 创建示例 .var (基因元数据)
    example_var_ps = pd.DataFrame(index=[f'gene_{j}' for j in range(n_genes_ps)])
    
    # 创建占位符 AnnData 对象
    pseudo_bulk_adata_placeholder = sc.AnnData(X=example_X_ps, obs=example_obs_ps, var=example_var_ps)
    
    # 可选：添加一些虚假的 HVG 信息以测试该路径
    # pseudo_bulk_adata_placeholder.var['highly_variable_genes'] = np.random.choice([True, False], size=n_genes_ps)
    # pseudo_bulk_adata_placeholder.var['highly_variable_genes'].iloc[:n_genes_ps//2] = True # 确保有一些为 True

    print(f"占位符 pseudo_bulk_adata 创建完毕，形状为: {pseudo_bulk_adata_placeholder.shape}")
    print(f"示例原始数据 (前 3x5 个样本):\n{pseudo_bulk_adata_placeholder.X[:3, :5]}")

    # --- 定义预处理参数 ---
    norm_target_sum = 10000.0  # 例如，类似 CPM 的标准化
    num_bins_for_scgpt = 51    # scGPT 模型的分箱数
    # hvg_filter_col = "highly_variable_genes" # 示例："highly_variable_genes" 或 None
    hvg_filter_col = None # 当前示例不使用 HVG 筛选
    # --- 参数定义结束 ---

    print("\n开始使用占位符数据进行伪批量数据预处理...")
    # 调用核心预处理函数
    processed_adata = preprocess_pseudo_bulk_for_scgpt(
        pseudo_bulk_adata_placeholder,
        target_sum=norm_target_sum,
        n_bins=num_bins_for_scgpt,
        hvg_col_name=hvg_filter_col
    )

    if processed_adata: # 如果预处理成功
        print("\n伪批量数据预处理成功！")
        print(f"生成的 AnnData 形状: {processed_adata.shape}")
        
        # 检查标准化数据 (如果执行了标准化)
        if 'normalized' in processed_adata.layers:
            print(f"标准化数据总和 (第一个样本，如果执行了标准化): {processed_adata.layers['normalized'][0].sum():.2f}")
        
        # 检查 log1p 转换后的数据
        if 'log1p' in processed_adata.layers:
            print(f"Log1p 数据 (前 3x5 个样本):\n{processed_adata.layers['log1p'][:3, :5]}")
        
        # 检查分箱后的表达数据
        if 'binned_expression' in processed_adata.layers:
            print(f"分箱后的表达数据 (前 3x5 个样本):\n{processed_adata.layers['binned_expression'][:3, :5]}")
            unique_bins = np.unique(processed_adata.layers['binned_expression']) # 获取唯一的 bin 值
            print(f"观察到的唯一 bin 值: {unique_bins}")
            print(f"最小 bin: {np.min(unique_bins)}, 最大 bin: {np.max(unique_bins)}")
            # 检查最大 bin 值是否在预期范围内
            if np.max(unique_bins) >= num_bins_for_scgpt:
                 print(f"警告: 最大 bin {np.max(unique_bins)} 超出预期范围 [0, {num_bins_for_scgpt-1}]")

        # 检查 .X 的最终内容 (应为 log1p 数据)
        # print(f"最终 .X 数据 (前 3x5 个样本，应为 log1p):\n{processed_adata.X[:3, :5]}")

    else: # 如果预处理失败
        print("\n伪批量数据预处理失败。请检查日志中的错误。")
        
    print("\n--- 示例用法结束 ---")
    print("\n提醒：要使用此脚本，请将其与您的实际伪批量数据生成流程集成。")

```
