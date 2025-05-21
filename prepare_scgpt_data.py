# 导入 scanpy 库，用于处理 AnnData 对象 (单细胞数据的事实标准格式)
import scanpy as sc
# 导入 json 库，用于加载 JSON 文件 (例如 scGPT 模型的词汇表)
import json
# 从 pathlib 库导入 Path 对象，用于以面向对象的方式处理文件和目录路径
from pathlib import Path

# --- 用户定义的占位符路径 ---
# 请将这些占位符替换为您的实际文件路径
# AnnData 文件 (.h5ad) 的路径
adata_file_path = "YOUR_PATH_TO_ANNDATA.h5ad"
# 包含 vocab.json (scGPT 模型词汇表) 的预训练模型目录路径
model_dir_path = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/"
# --- 用户定义占位符路径结束 ---

# 定义加载和准备数据的函数
def load_and_prepare_data(adata_path_str: str, model_dir_str: str):
    """
    加载单细胞 AnnData 对象和 scGPT 词汇表，然后筛选 AnnData 对象，
    使其仅包含两者共有的基因。

    Args:
        adata_path_str (str): .h5ad AnnData 文件的路径字符串。
        model_dir_str (str): 包含 vocab.json 文件的目录的路径字符串。

    Returns:
        tuple: 一个包含以下元素的元组：
            - filtered_adata (sc.AnnData | None): 筛选后仅包含共同基因的 AnnData 对象。
                                                 如果发生错误，则返回 None。
            - model_gene_vocab (list[str] | None): 从 vocab.json 加载的基因词汇表列表。
                                                 如果发生错误，则返回 None。
    """
    # 打印正在加载的 AnnData 文件路径
    print(f"正在从以下路径加载 AnnData: {adata_path_str}")
    # 将字符串路径转换为 Path 对象
    adata_path = Path(adata_path_str)
    # 检查 AnnData 文件是否存在
    if not adata_path.exists():
        print(f"错误: 在 {adata_path_str} 未找到 AnnData 文件")
        print("请确保占位符路径 'adata_file_path' 已正确设置。")
        return None, None # 如果文件不存在，返回 None
    
    # 尝试读取 AnnData 文件
    try:
        adata = sc.read_h5ad(adata_path)
        # 打印成功加载 AnnData 对象的信息，包括细胞数和基因数
        print(f"成功加载 AnnData 对象，包含 {adata.n_obs} 个细胞和 {adata.n_vars} 个基因。")
    except Exception as e: # 捕获读取过程中可能发生的任何异常
        print(f"读取 AnnData 文件 {adata_path_str} 时出错: {e}")
        return None, None # 如果读取失败，返回 None

    # 构建词汇表文件的完整路径 (假设词汇表文件名为 vocab.json，位于模型目录中)
    vocab_file = Path(model_dir_str) / "vocab.json"
    # 打印正在加载的词汇表文件路径
    print(f"正在从以下路径加载词汇表: {vocab_file}")

    # 检查词汇表文件是否存在
    if not vocab_file.exists():
        print(f"错误: 在 {vocab_file} 未找到词汇表文件 (vocab.json)")
        print("请确保 'model_dir_path' 正确，并且 'vocab.json' 存在于该目录中。")
        return None, None # 如果文件不存在，返回 None

    # 尝试加载和解析词汇表文件
    try:
        # 打开词汇表文件进行读取
        with open(vocab_file, "r") as f:
            vocab_content = json.load(f) # 解析 JSON 内容
        
        # 处理不同格式的词汇表 (scGPT 可能输出列表或字典)
        if isinstance(vocab_content, list): # 如果词汇表是基因列表
            model_gene_vocab = vocab_content
        elif isinstance(vocab_content, dict): # 如果词汇表是字典 (例如 {gene: index})
            model_gene_vocab = list(vocab_content.keys()) # 提取基因名称作为列表
        else: # 如果格式不符合预期
            print(f"错误: {vocab_file} 中的词汇表格式非预期。期望是一个基因列表或字典。")
            return None, None
        
        # 验证词汇表中的所有基因名称是否为字符串
        if not all(isinstance(gene, str) for gene in model_gene_vocab):
            print(f"错误: {vocab_file} 中的词汇表包含非字符串元素。基因名称必须是字符串。")
            return None, None
            
        # 打印成功加载词汇表的信息，包括词汇表中的基因数量
        print(f"成功加载词汇表，包含 {len(model_gene_vocab)} 个基因。")

    except json.JSONDecodeError: # 捕获 JSON 解析错误
        print(f"错误: 解析词汇表文件 {vocab_file} 中的 JSON 时出错")
        return None, None
    except Exception as e: # 捕获其他加载词汇表文件时可能发生的异常
        print(f"错误: 加载词汇表文件 {vocab_file} 时出错: {e}")
        return None, None

    # 获取原始 AnnData 对象中的基因列表
    original_adata_genes = list(adata.var_names)
    # 打印原始 AnnData 对象中的基因数量
    print(f"原始 AnnData 中的基因数量: {len(original_adata_genes)}")

    # 识别 AnnData 和模型词汇表之间的共同基因
    # 使用集合 (set) 的交集操作 (&) 找到共同基因，然后转换为排序列表
    common_genes = sorted(list(set(original_adata_genes) & set(model_gene_vocab)))
    
    # 检查是否找到了共同基因
    if not common_genes:
        print("错误: 在 AnnData 和词汇表之间未找到共同基因。")
        print("请检查您的 AnnData 基因名称和词汇表内容 (例如，基因符号与 Ensemble ID 是否匹配)。")
        # 根据示例逻辑的隐含行为，即使没有共同基因也返回词汇表
        return None, model_gene_vocab 
        
    # 打印找到的共同基因数量
    print(f"找到的共同基因数量: {len(common_genes)}")

    # 根据共同基因筛选 AnnData 对象
    # 使用 .copy() 以确保我们得到的是一个新的 AnnData 对象切片，而不是视图
    adata_filtered = adata[:, common_genes].copy()
    # 打印筛选后 AnnData 对象中的基因数量
    print(f"已将 AnnData 筛选至 {adata_filtered.n_vars} 个共同基因。")

    # 返回筛选后的 AnnData 对象和完整的模型基因词汇表 (根据示例)
    return adata_filtered, model_gene_vocab

# 当脚本作为主程序执行时 (即直接运行此文件，而不是作为模块导入)
if __name__ == '__main__':
    # 打印数据准备过程开始的提示信息
    print("开始数据准备过程...")
    
    # 检查用户是否已修改占位符路径
    if adata_file_path == "YOUR_PATH_TO_ANNDATA.h5ad" or \
       model_dir_path == "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/":
        print("警告: 占位符路径仍为默认值。")
        print("请编辑脚本，将 'adata_file_path' 和 'model_dir_path' 设置为您的实际路径。")
    else: # 如果路径已修改，则使用用户提供的路径
        print(f"使用 AnnData 路径: {adata_file_path}")
        print(f"使用模型目录: {model_dir_path}")
        
        # 调用核心函数加载和准备数据
        filtered_adata, vocabulary = load_and_prepare_data(adata_file_path, model_dir_path)

        # 检查 AnnData 是否成功返回 (词汇表可能在 common_gene 失败时仍返回)
        if filtered_adata is not None: 
            print("----------------------------------------------------")
            print("数据准备成功！")
            # 打印筛选后 AnnData 的形状 (细胞数 x 基因数)
            print(f"筛选后的 AnnData 形状: {filtered_adata.shape}")
            # 如果词汇表也成功加载 (在灾难性文件加载失败时可能为 None)
            if vocabulary:
                 print(f"词汇表大小: {len(vocabulary)}")
            # 现在可以使用这个 filtered_adata 进行 scGPT 的微调
            print("您现在可以使用此 filtered_adata 进行 scGPT 的微调。")
        else: # 如果数据准备失败
            print("----------------------------------------------------")
            print("数据准备失败。请检查上面的错误信息。")
            print("确保您的路径正确，并且文件格式符合预期。")
    
    # 打印数据准备脚本结束的提示信息
    print("数据准备脚本已完成。")
```
