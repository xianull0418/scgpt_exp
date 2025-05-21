# 导入 PyTorch 相关库
import torch # PyTorch 深度学习框架
from torch.utils.data import Dataset, DataLoader # 用于创建自定义数据集和数据加载器
# 导入其他常用库
import numpy as np # NumPy 用于数值计算
import pandas as pd # Pandas 用于数据处理
import scanpy as sc # Scanpy 用于处理 AnnData 对象
import sys # Sys 用于与 Python 解释器交互，例如修改模块搜索路径
from typing import Dict, Any, Optional # Typing 用于类型提示

# 假设 scgpt 模块在 Python 路径中
# 如果 scgpt 未安装，可能需要取消注释下一行并指定 scGPT 仓库的路径
# sys.path.append("path_to_scGPT_repository")
try:
    from scgpt.tokenizer import GeneVocab # 尝试从 scgpt.tokenizer 导入 GeneVocab 类，用于加载和使用基因词汇表
except ImportError:
    # 如果导入失败，打印警告信息并定义一个虚拟的 GeneVocab 类
    # 这主要用于脚本生成或测试，当实际的 scGPT 环境不可用时
    print("警告: 未找到 scGPT.tokenizer.GeneVocab。将使用虚拟类进行脚本生成。")
    # 定义一个虚拟的 GeneVocab 类，模拟真实 GeneVocab 的基本功能
    class GeneVocab:
        # 构造函数
        def __init__(self, gene_list, specials=None):
            self.stoi = {} # string-to-index 字典，存储基因/特殊标记到索引的映射
            # 首先添加基因
            for i, gene in enumerate(gene_list):
                self.stoi[gene] = i
            
            # 添加特殊标记 (special tokens)
            # 确保它们不会覆盖现有的基因索引（如果名称冲突）
            # 并确保它们在不冲突的情况下获得唯一的索引
            current_max_idx = len(gene_list) -1 # 当前最大索引
            if specials: # 如果提供了特殊标记列表
                for special in specials:
                    if special not in self.stoi: # 如果特殊标记不在字典中
                        current_max_idx += 1
                        self.stoi[special] = current_max_idx # 添加新的特殊标记
            
            self.itos = {i: gene for gene, i in self.stoi.items()} # index-to-string 字典
            
            # 定义常用的特殊标记属性，如果输入的 specials 中没有，则使用默认值
            self._pad_token = "<pad>" # 填充标记
            self._cls_token = "<cls>" # 分类标记 (class token)
            self._mask_token = "<mask>" # 掩码标记 (mask token)，此脚本中不直接使用
            self._eos_token = "<eos>"   # 序列结束标记 (end-of-sequence token)
            self._eoc_token = "<eoc>"   # 表达结束标记 (end-of-cell expression token)

            # 如果默认的填充和分类标记不在 stoi 中，则添加它们
            if self._pad_token not in self.stoi: self.stoi[self._pad_token] = len(self.stoi)
            if self._cls_token not in self.stoi: self.stoi[self._cls_token] = len(self.stoi)
            # 可能添加了新标记后，重建 itos
            self.itos = {i: gene for gene, i in self.stoi.items()}

        # 通过标记获取索引
        def __getitem__(self, token: str) -> int:
            return self.stoi.get(token, -1) # 如果标记不存在，返回 -1 (或适当处理未知标记)

        # 返回词汇表大小
        def __len__(self) -> int:
            return len(self.stoi)
        
        # 获取填充标记的索引 (property)
        @property
        def pad_idx(self) -> int:
            idx = self.stoi.get(self._pad_token)
            if idx is None: raise ValueError(f"填充标记 '{self._pad_token}' 在词汇表中未找到。")
            return idx

        # 获取 CLS 标记的索引 (property)
        @property
        def cls_idx(self) -> int:
            idx = self.stoi.get(self._cls_token)
            if idx is None: raise ValueError(f"CLS 标记 '{self._cls_token}' 在词汇表中未找到。")
            return idx
        
        # 获取掩码标记的索引 (property, 可选)
        @property
        def mask_idx(self) -> Optional[int]:
            return self.stoi.get(self._mask_token)

        # 获取 EOS 标记的索引 (property, 可选)
        @property
        def eos_idx(self) -> Optional[int]:
            return self.stoi.get(self._eos_token)
        
        # 获取 EOC 标记的索引 (property, 可选)
        @property
        def eoc_idx(self) -> Optional[int]:
            return self.stoi.get(self._eoc_token)

        # 从文件加载词汇表的虚拟类方法
        @classmethod
        def from_file(cls, filepath: str): # 虚拟方法
            print(f"虚拟 GeneVocab.from_file 被调用，路径: {filepath}")
            import json # 动态导入 json，因为这只是一个虚拟方法
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f) # 加载 JSON 文件
                if isinstance(data, dict): # 假设是 stoi 格式，如 {"<pad>":0, "gene1":1}
                    # 基于典型模式提取基因和特殊标记
                    genes = [k for k,v in data.items() if not (k.startswith("<") and k.endswith(">"))]
                    specials = [k for k,v in data.items() if (k.startswith("<") and k.endswith(">"))]
                    return cls(genes, specials=specials)
                elif isinstance(data, list): # 假设是基因列表
                    return cls(data, specials=["<pad>", "<cls>", "<eoc>", "<mask>", "<eos>"]) # 添加默认特殊标记
            except Exception as e:
                print(f"虚拟 GeneVocab.from_file 中出错: {e}。将使用默认的小型词汇表。")
            # 如果文件加载失败或未实现，则回退到一个非常简单的词汇表
            return cls(["gene1", "gene2", "gene3"], specials=["<pad>", "<cls>"])

# 定义伪批量数据集类，继承自 PyTorch 的 Dataset
class PseudoBulkDataset(Dataset):
    # 构造函数
    def __init__(self, 
                 pseudo_bulk_adata: sc.AnnData, # 预处理过的 AnnData 对象，包含分箱后的表达数据和细胞比例
                 vocab: GeneVocab, # scGPT 词汇表对象 (GeneVocab)
                 max_seq_len: int, # 最大序列长度 (基因数 + CLS 标记)
                 expression_layer: str = 'binned_expression', # AnnData 中存储分箱表达值的层名称
                 cls_value_bin: int = 0, # 分配给 CLS 标记表达值的分箱值
                 pad_value: int = -2): # 用于填充表达值的数值
        """
        用于伪批量反卷积的自定义 PyTorch 数据集。

        Args:
            pseudo_bulk_adata: 预处理过的 AnnData 对象，包含分箱后的表达数据和细胞组分。
            vocab: scGPT 词汇表对象 (GeneVocab)。
            max_seq_len: 最大序列长度（基因 + CLS 标记）。
            expression_layer: AnnData 中包含分箱表达值的层的名称。
            cls_value_bin: 分配给 CLS 标记的表达值的 bin 值。
            pad_value: 用于填充表达数据的值。
        """
        self.adata = pseudo_bulk_adata # AnnData 对象
        self.genes = list(pseudo_bulk_adata.var_names) # 基因名称列表
        self.vocab = vocab # 词汇表对象
        self.max_seq_len = max_seq_len # 最大序列长度
        self.expression_layer = expression_layer # 表达数据层名称
        self.cls_value_bin = cls_value_bin # CLS 标记的表达值
        self.pad_value = pad_value # 填充值

        # 检查基因列表是否为空
        if not self.genes:
            raise ValueError("pseudo_bulk_adata.var_names 为空。无法创建数据集。")

        # 预先计算基因 ID (目前不包括 CLS 标记)
        self.gene_ids_no_cls = [self.vocab[gene] for gene in self.genes]
        # 检查是否有基因未在词汇表中找到
        if any(gid == -1 for gid in self.gene_ids_no_cls):
            unknown_genes = [self.genes[i] for i, gid in enumerate(self.gene_ids_no_cls) if gid == -1]
            print(f"警告: 一些基因未在词汇表中找到，将被映射为 -1: {unknown_genes[:5]}...")
            # 根据模型不同，如果嵌入层的 padding_idx 未处理 -1，可能会导致问题
        
        # 确保 CLS 和 PAD 标记在词汇表中 (如果不在，属性访问会引发错误)
        _ = self.vocab.cls_idx 
        _ = self.vocab.pad_idx

    # 返回数据集中样本的数量
    def __len__(self) -> int:
        return self.adata.n_obs

    # 根据索引获取单个样本
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. 获取样本的分箱表达数据
        # 确保是密集的 numpy 数组
        raw_expression_vector = self.adata.layers[self.expression_layer][idx]
        if hasattr(raw_expression_vector, "toarray"): # 检查是否为稀疏矩阵
            sample_binned_expression = raw_expression_vector.toarray().flatten() # 转换为密集数组并展平
        else:
            sample_binned_expression = np.asarray(raw_expression_vector).flatten() # 转换为数组并展平

        # 2. 获取细胞类型比例
        # 假设比例数据在 .obs 中，并且已经是数值类型
        sample_fractions_np = self.adata.obs.iloc[idx, :].values.astype(np.float32) # 获取行数据，转换为 float32
        sample_fractions = torch.tensor(sample_fractions_np, dtype=torch.float32) # 转换为 PyTorch 张量

        # 3. 准备包含 CLS 标记的基因 ID 和分箱值
        # 基因 ID 序列: [CLS_ID, gene1_ID, gene2_ID, ...]
        current_gene_ids_list = [self.vocab.cls_idx] + self.gene_ids_no_cls
        # 分箱值序列: [CLS_VALUE_BIN, bin_val1, bin_val2, ...]
        current_values_list = np.concatenate((np.array([self.cls_value_bin]), sample_binned_expression))
        
        current_seq_len = len(current_gene_ids_list) # 当前序列长度

        # 4. 填充 (Padding)
        padding_len = self.max_seq_len - current_seq_len # 需要填充的长度
        
        if padding_len < 0: # 如果当前序列长度超过最大长度，则截断
            # 理想情况下，如果 max_seq_len 基于 adata.var 正确计算，则不应发生这种情况
            # print(f"警告: current_seq_len ({current_seq_len}) > max_seq_len ({self.max_seq_len})。将进行截断。")
            padded_gene_ids_np = np.array(current_gene_ids_list[:self.max_seq_len])
            padded_values_np = np.array(current_values_list[:self.max_seq_len])
            # 对于截断的序列，掩码应全为 False (没有填充标记)
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        elif padding_len > 0: # 如果需要填充
            # 填充基因 ID 序列
            padded_gene_ids_np = np.pad(current_gene_ids_list, (0, padding_len), 
                                        mode='constant', constant_values=self.vocab.pad_idx)
            # 填充表达值序列
            padded_values_np = np.pad(current_values_list, (0, padding_len), 
                                      mode='constant', constant_values=self.pad_value)
            # 创建填充掩码，填充位置为 True
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            padding_mask[current_seq_len:] = True # 将填充部分标记为 True
        else: # 无需填充，当前序列长度等于最大序列长度
            padded_gene_ids_np = np.array(current_gene_ids_list)
            padded_values_np = np.array(current_values_list)
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool) # 全为 False

        # 返回一个包含处理后数据的字典
        return {
            "gene_ids": torch.tensor(padded_gene_ids_np, dtype=torch.long), # 基因 ID 张量
            "values": torch.tensor(padded_values_np, dtype=torch.float32), # 分箱表达值张量 (作为嵌入层输入或浮点特征)
            "padding_mask": padding_mask, # 填充掩码张量 (True 表示填充位置)
            "fractions": sample_fractions, # 细胞类型比例张量 (反卷积的目标)
        }

# 定义创建数据加载器的函数
def create_dataloaders(
    train_adata: sc.AnnData, # 预处理后的训练集 AnnData 对象
    valid_adata: sc.AnnData, # 预处理后的验证集 AnnData 对象
    vocab: GeneVocab, # scGPT 词汇表 (GeneVocab)
    batch_size: int, # 每个批次的样本数
    expression_layer: str = 'binned_expression', # AnnData 中分箱表达数据所在的层
    cls_value_bin: int = 0, # CLS 标记的 bin 值
    pad_value: int = -2, # 表达数据的填充值
    num_workers: int = 0 # 用于数据加载的子进程数
) -> tuple[DataLoader, DataLoader]: # 返回包含训练和验证 DataLoader 的元组
    """
    为训练和验证创建 PyTorch DataLoader。

    Args:
        train_adata: 预处理的训练 AnnData 对象。
        valid_adata: 预处理的验证 AnnData 对象。
        vocab: scGPT 词汇表 (GeneVocab)。
        batch_size: 每个批次的样本数。
        expression_layer: AnnData 中用于分箱表达的层。
        cls_value_bin: CLS 标记的 bin 值。
        pad_value: 表达数据的填充值。
        num_workers: 数据加载的子进程数。

    Returns:
        一个包含 (train_loader, valid_loader) 的元组。
    """
    # 检查训练集和验证集的基因名称是否一致且顺序相同
    if not train_adata.var_names.equals(valid_adata.var_names):
        raise ValueError("训练和验证 AnnData 对象必须具有相同顺序的相同 var_names (基因)。")
    
    # max_seq_len 包括 CLS 标记
    max_seq_len = len(train_adata.var_names) + 1 
    print(f"DataLoader 的最大序列长度 (基因数 + CLS 标记): {max_seq_len}")

    # 创建训练数据集实例
    train_dataset = PseudoBulkDataset(
        train_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value
    )
    # 创建验证数据集实例
    valid_dataset = PseudoBulkDataset(
        valid_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset, # 训练数据集
        batch_size=batch_size, # 批量大小
        shuffle=True, # 每个 epoch 开始时打乱数据
        num_workers=num_workers, # 数据加载子进程数
        pin_memory=True if num_workers > 0 else False, # 如果使用多进程加载，则将数据复制到 CUDA 固定内存中以加速传输
        drop_last=True # 丢弃最后一个不完整的批次 (通常在训练时为 True，以保持批次大小稳定)
    )
    # 创建验证数据加载器
    valid_loader = DataLoader(
        valid_dataset, # 验证数据集
        batch_size=batch_size, # 批量大小
        shuffle=False, # 验证时通常不打乱数据
        num_workers=num_workers, # 数据加载子进程数
        pin_memory=True if num_workers > 0 else False,
        drop_last=False # 验证时通常不丢弃最后一个批次
    )
    print(f"已创建 DataLoaders: 训练批次数={len(train_loader)}, 验证批次数={len(valid_loader)}")
    return train_loader, valid_loader # 返回数据加载器

# 当脚本作为主程序执行时
if __name__ == '__main__':
    print("--- 用于反卷积的 PyTorch Dataset 和 DataLoader 示例 ---")

    # --- 虚拟数据配置 ---
    N_TRAIN_SAMPLES, N_VALID_SAMPLES = 120, 40 # 减少样本数以便快速示例
    N_GENES = 50  # 减少基因数量以缩小序列长度
    N_CELL_TYPES = 8 # 细胞类型数量
    BATCH_SIZE_EXAMPLE = 16 # 较小的批量大小
    EXPRESSION_LAYER_NAME = 'binned_expression' # 表达数据层名称
    # CLS 标记的表达值 (它落入哪个 bin 或一个特殊值)
    CLS_TOKEN_VALUE_BIN_EXAMPLE = 0 
    # 用于填充表达数据点 (基因) 的值
    # 这理想情况下应该是模型的表达值嵌入层可以忽略的值 (例如 padding_idx)
    # 或者一个超出分箱数据典型范围的值。
    EXPRESSION_PAD_VALUE_EXAMPLE = -2 
    # --- 配置结束 ---

    print("\n1. 创建虚拟词汇表...")
    gene_names_example = [f"gene_{i}" for i in range(N_GENES)] # 示例基因名称
    # 确保标准特殊标记是虚拟 GeneVocab 的一部分
    special_tokens_example = ["<pad>", "<cls>", "<eoc>", "<mask>", "<eos>"] 
    example_vocab_instance = GeneVocab(gene_names_example, specials=special_tokens_example) # 创建词汇表实例
    print(f"虚拟词汇表已创建。大小: {len(example_vocab_instance)}。CLS ID: {example_vocab_instance.cls_idx}, PAD ID: {example_vocab_instance.pad_idx}")

    print("\n2. 创建虚拟的预处理 AnnData 对象...")
    # 定义一个函数来创建虚拟 AnnData 对象
    def create_dummy_anndata(n_samples, var_names, n_cell_types, layer_name):
        # 模拟分箱表达数据 (例如，对于 51 个 bin，值为 0 到 50)
        binned_data = np.random.randint(0, 51, size=(n_samples, len(var_names))).astype(np.float32)
        # 模拟细胞类型比例 (总和为 1)
        fractions_data = np.random.rand(n_samples, n_cell_types)
        fractions_data = fractions_data / fractions_data.sum(axis=1, keepdims=True) # 归一化
        
        # 创建 AnnData 对象
        adata = sc.AnnData(
            X=np.random.rand(n_samples, len(var_names)).astype(np.float32), # .X 中存储类似原始数据 (Dataset 不使用)
            layers={layer_name: binned_data}, # .layers 中存储分箱数据
            obs=pd.DataFrame(fractions_data, 
                             index=[f"sample_s{i}" for i in range(n_samples)], # 样本名称
                             columns=[f"CellType_{ct}" for ct in range(n_cell_types)]), # 细胞类型名称
            var=pd.DataFrame(index=var_names) # 基因名称
        )
        return adata

    # 创建虚拟训练和验证 AnnData 对象
    train_adata_dummy = create_dummy_anndata(N_TRAIN_SAMPLES, gene_names_example, N_CELL_TYPES, EXPRESSION_LAYER_NAME)
    valid_adata_dummy = create_dummy_anndata(N_VALID_SAMPLES, gene_names_example, N_CELL_TYPES, EXPRESSION_LAYER_NAME)
    print(f"虚拟训练 AnnData: {train_adata_dummy.shape}, 层 '{EXPRESSION_LAYER_NAME}' 已找到。")
    print(f"虚拟验证 AnnData: {valid_adata_dummy.shape}, 层 '{EXPRESSION_LAYER_NAME}' 已找到。")
    print(f"示例比例 (第一个样本): {train_adata_dummy.obs.iloc[0].values}")

    print("\n3. 创建 DataLoaders...")
    try:
        # 调用 create_dataloaders 函数
        train_loader_example, valid_loader_example = create_dataloaders(
            train_adata_dummy,
            valid_adata_dummy,
            example_vocab_instance,
            batch_size=BATCH_SIZE_EXAMPLE,
            expression_layer=EXPRESSION_LAYER_NAME,
            cls_value_bin=CLS_TOKEN_VALUE_BIN_EXAMPLE,
            pad_value=EXPRESSION_PAD_VALUE_EXAMPLE,
            num_workers=0 # 在主进程中加载数据
        )
        print("\nDataLoaders 创建成功。")

        print("\n4. 检查来自 train_loader 的一个样本批次...")
        sample_batch = next(iter(train_loader_example)) # 获取第一个批次
        # 打印批次中每个键的形状和数据类型
        for key, value in sample_batch.items():
            print(f"  键: '{key}', 形状: {value.shape}, 数据类型: {value.dtype}")
        
        # 详细检查批次中的第一个样本
        print("\n  详细检查批次中的第一个样本:")
        first_sample_gene_ids = sample_batch['gene_ids'][0] # 第一个样本的基因 ID
        first_sample_values = sample_batch['values'][0] # 第一个样本的表达值
        first_sample_padding_mask = sample_batch['padding_mask'][0] # 第一个样本的填充掩码
        
        print(f"    CLS 标记 ID (预期 {example_vocab_instance.cls_idx}): {first_sample_gene_ids[0].item()}")
        print(f"    CLS 标记值 (预期 {CLS_TOKEN_VALUE_BIN_EXAMPLE}): {first_sample_values[0].item()}")
        
        # 查找第一个填充位置以验证填充（如果存在）
        # max_s_len 是基因数 + CLS
        actual_seq_len_plus_cls = N_GENES + 1 
        if actual_seq_len_plus_cls < first_sample_gene_ids.shape[0]: # 检查是否存在填充
            print(f"    第一个填充位置的 PAD 标记 ID (预期 {example_vocab_instance.pad_idx}): {first_sample_gene_ids[actual_seq_len_plus_cls].item()}")
            print(f"    第一个填充位置的 PAD 值 (预期 {EXPRESSION_PAD_VALUE_EXAMPLE}): {first_sample_values[actual_seq_len_plus_cls].item()}")
            print(f"    第一个填充位置的填充掩码 (预期 True): {first_sample_padding_mask[actual_seq_len_plus_cls].item()}")
        else:
            print("    序列未填充 (max_seq_len 等于 actual_seq_len)。")
            print(f"    序列中最后一个基因 ID: {first_sample_gene_ids[-1].item()}")
            print(f"    序列中最后一个值: {first_sample_values[-1].item()}")
            print(f"    最后一个位置的填充掩码 (预期 False): {first_sample_padding_mask[-1].item()}")


    except Exception as e:
        print(f"\nDataLoader 示例期间发生错误: {e}")
        import traceback # 导入 traceback 模块以打印详细错误信息
        traceback.print_exc() # 打印异常的堆栈跟踪

    print("\n--- DataLoader 示例结束 ---")
    print("提醒: 实际使用时，请用真实的、预处理过的 AnnData 对象和有效的 scGPT 词汇表替换虚拟数据。")
```
