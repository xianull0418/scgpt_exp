# 导入 PyTorch 库
import torch
# 导入 PyTorch 神经网络模块，用于构建神经网络层
import torch.nn as nn
# 导入 json 库，用于读取和写入 JSON 文件（例如模型配置文件）
import json
# 从 pathlib 导入 Path 对象，用于处理文件和目录路径
from pathlib import Path
# 导入 sys 库，用于与 Python 解释器交互（例如修改模块搜索路径）
import sys
# 从 typing 导入 Optional, Dict, Any，用于类型提示，增强代码可读性和健壮性
from typing import Optional, Dict, Any

# 假设 scgpt 模块在 Python 的搜索路径中
# 如果 scgpt 未作为包安装，可能需要取消注释下一行并提供 scGPT 仓库的路径
# sys.path.append("path_to_scGPT_repository")
try:
    # 尝试从 scgpt.model 模块导入 TransformerModel 类
    from scgpt.model import TransformerModel
    # 尝试从 scgpt.tokenizer 模块导入 GeneVocab 类，用于加载基因词汇表
    from scgpt.tokenizer import GeneVocab
except ImportError:
    # 如果导入失败（例如 scgpt 未安装或路径配置不正确），则执行以下回退逻辑
    # 这允许脚本在没有完整 scGPT 环境的情况下至少能够被创建（但无法实际运行模型）
    print("警告: 未找到 scGPT 库。请确保已安装或已将其路径添加到 PYTHONPATH 以获得完整功能。")
    # 定义虚拟的 TransformerModel 类，以便脚本在语法上有效
    class TransformerModel(nn.Module):
        # 虚拟 TransformerModel 的构造函数
        def __init__(self, *args, **kwargs):
            super().__init__() # 调用父类 nn.Module 的构造函数
            print("虚拟 TransformerModel 已初始化，因为未找到 scGPT。")
            # 添加一个基础层，以便在虚拟示例中允许状态字典的加载/保存
            self.dummy_layer = nn.Linear(10,10)
        # 虚拟 TransformerModel 的 load_state_dict 方法
        def load_state_dict(self, state_dict, strict=True):
            print("虚拟 TransformerModel.load_state_dict 已调用。")
            # 处理虚拟示例的最小化状态字典
            if 'dummy_layer.weight' in state_dict and 'dummy_layer.bias' in state_dict:
                 super().load_state_dict({'dummy_layer.weight': state_dict['dummy_layer.weight'], 
                                          'dummy_layer.bias': state_dict['dummy_layer.bias']}, strict=False)
            return [], [] # 返回空的 missing_keys 和 unexpected_keys
        # 虚拟 TransformerModel 的 forward 方法
        def forward(self, *args, **kwargs):
            print("虚拟 TransformerModel.forward 已调用。")
            # 返回一个符合预期的输出结构，包含随机的 cell_emb
            return {'cell_emb': torch.randn(1, kwargs.get("d_model", 128))} # d_model 来自虚拟参数

    # 定义虚拟的 GeneVocab 类
    class GeneVocab:
        # 虚拟 GeneVocab 的构造函数
        def __init__(self, genes_list, specials=None):
            # 创建字符串到索引的映射 (stoi)
            self.stoi = {gene: i for i, gene in enumerate(genes_list)}
            if specials: # 如果提供了特殊标记
                for i, special_token in enumerate(specials):
                    if special_token not in self.stoi: # 如果特殊标记尚未存在，则添加
                        self.stoi[special_token] = len(self.stoi)
            # 创建索引到字符串的映射 (itos)
            self.itos = {i: gene for gene, i in self.stoi.items()}
            self._default_index = -1 # 占位符，表示默认索引
            print("虚拟 GeneVocab 已初始化。")

        # 从文件加载词汇表的虚拟类方法
        @classmethod
        def from_file(cls, file_path: Path):
            print(f"虚拟 GeneVocab.from_file 已调用，路径: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f) # 加载 JSON 数据
            if isinstance(data, dict): # 假设是 {"<pad>":0, "gene1":1} 这样的 stoi 格式
                return cls(list(data.keys())) # 为虚拟目的简化处理
            elif isinstance(data, list): # 假设是基因列表格式
                return cls(data)
            raise ValueError("不支持的虚拟词汇表格式")
        
        # 获取默认索引的方法
        def get_default_index(self):
            return self._default_index

        # 返回词汇表大小的方法
        def __len__(self):
            return len(self.itos)
        
        # 通过标记获取索引的方法
        def __getitem__(self, token):
            return self.stoi.get(token, self._default_index)


# --- 用户定义的占位符路径 ---
# 这些路径应由用户替换为实际的预训练模型目录和词汇表文件路径
# model_dir_placeholder = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/" # 应包含 args.json, best_model.pt 文件
# vocab_file_placeholder = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/vocab.json" # 如果词汇表不在模型目录中，则为单独路径
# --- 用户定义占位符路径结束 ---

# 定义创建反卷积模型的函数
def create_deconvolution_model(
    model_dir_path: str, # 预训练 scGPT 模型所在目录的路径
    vocab_file_path: str, # 基因词汇表文件 (vocab.json) 的路径
    n_cell_types: int, # 要反卷积的目标细胞类型的数量
    scgpt_dropout_rate: Optional[float] = None, # 可选，用于覆盖 scGPT 基础模型中的 dropout 率
    head_dropout_rate: float = 0.1, # 反卷积头的 dropout 率，默认为 0.1
    freeze_scgpt_base: bool = False, # 是否冻结 scGPT 基础模型的参数，默认为 False (即微调基础模型)
    d_model_override: Optional[int] = None, # 可选，用于覆盖模型配置中的 d_model (嵌入维度)
    nhead_override: Optional[int] = None, # 可选，用于覆盖模型配置中的 nhead (多头注意力头数)
    nlayers_override: Optional[int] = None, # 可选，用于覆盖模型配置中的 nlayers (Transformer 层数)
    pad_value: int = -2 # 某些 scGPT 设置中使用的默认填充值
) -> Optional[nn.Module]: # 返回创建的模型 (nn.Module) 或在出错时返回 None
    '''
    通过加载预训练的 scGPT 模型并在其上添加一个针对特定任务的反卷积头来创建反卷积模型。
    '''
    print(f"正在创建反卷积模型...")
    print(f"从目录加载预训练模型: {model_dir_path}")
    print(f"从文件加载词汇表: {vocab_file_path}")

    # 构建模型配置文件、权重文件和词汇表文件的完整路径
    model_config_file = Path(model_dir_path) / "args.json" # 模型配置文件路径
    model_weights_file = Path(model_dir_path) / "best_model.pt" # 模型权重文件路径
    vocab_path = Path(vocab_file_path) # 词汇表文件路径

    # 检查所需文件是否存在
    if not model_config_file.exists():
        print(f"错误: 在 {model_config_file} 未找到模型配置文件 'args.json'")
        return None
    if not model_weights_file.exists():
        print(f"错误: 在 {model_weights_file} 未找到模型权重文件 'best_model.pt'")
        return None
    if not vocab_path.exists():
        print(f"错误: 在 {vocab_path} 未找到词汇表文件 'vocab.json'")
        return None

    # 加载模型配置文件 (args.json)
    try:
        with open(model_config_file, "r") as f:
            model_configs: Dict[str, Any] = json.load(f) # 将 JSON 内容加载为字典
        print(f"已从 {model_config_file} 加载模型配置")
    except Exception as e:
        print(f"加载模型配置 {model_config_file} 时出错: {e}")
        return None
        
    # 加载词汇表 (vocab.json)
    try:
        vocab = GeneVocab.from_file(vocab_path) # 从文件创建 GeneVocab 对象
        ntoken = len(vocab) # 词汇表中的标记数量 (包括基因和特殊标记)
        # 从词汇表或 args.json 中确保正确识别填充标记 (pad_token)
        pad_token = model_configs.get("pad_token", "<pad>") # 获取 pad_token，默认为 "<pad>"
        if pad_token not in vocab.stoi: # 检查 pad_token 是否在词汇表的 string-to-index 映射中
             print(f"警告: 在 args.json 中定义的填充标记 '{pad_token}' (或默认值) 未在词汇表中找到。尝试使用词汇表的默认值或第一个特殊标记。")
             # 如果未明确定义填充标记或默认值缺失，则尝试查找填充标记
             if "<pad>" in vocab.stoi: pad_token = "<pad>"
             elif vocab.stoi: pad_token = vocab.itos[0] # 回退方案，不理想
             else:
                 print("错误: 无法从词汇表中确定填充标记。")
                 return None
        print(f"已从 {vocab_path} 加载词汇表。词汇表大小 (ntoken): {ntoken}, 填充标记: {pad_token}")
    except Exception as e:
        print(f"加载词汇表 {vocab_path} 时出错: {e}")
        return None

    # 从模型配置中提取参数，应用覆盖值和默认值
    d_model = d_model_override if d_model_override is not None else model_configs.get("embsize", 512) # 嵌入维度
    nhead = nhead_override if nhead_override is not None else model_configs.get("nheads", 4) # 多头注意力头数
    nlayers = nlayers_override if nlayers_override is not None else model_configs.get("nlayers", 4) # Transformer 层数
    
    d_hid = model_configs.get("d_hid", d_model * 4) # Transformer 前馈网络隐藏层维度，默认为 d_model 的 4 倍
    
    current_dropout = model_configs.get("dropout", 0.1) # scGPT 的默认 dropout 率
    if scgpt_dropout_rate is not None: # 如果指定了覆盖值
        current_dropout = scgpt_dropout_rate # 使用覆盖值
        
    # TransformerModel 可能需要的 args.json 中的其他参数
    nlayers_cls = model_configs.get("nlayers_cls", 3) # CLS 标记处理层的数量
    n_cls = model_configs.get("n_cls", 1) # CLS 标记的数量 (通常为 1，用于使用 CLS 标记的任务)
    input_emb_style = model_configs.get("input_emb_style", "continuous") # 输入嵌入风格 ("continuous", "binned")
    n_input_bins = model_configs.get("n_input_bins", 0) # 如果 input_emb_style 是 "binned"，则相关
    cell_emb_style = model_configs.get("cell_emb_style", "cls") # 细胞嵌入的生成方式
    use_fast_transformer = model_configs.get("use_fast_transformer", True) # 是否使用快速 Transformer 实现 (检查 args.json)
    pre_norm = model_configs.get("pre_norm", False) # 是否使用 Pre-Normalization (检查 args.json)
    
    # pad_value 也可以在 args.json 中定义，覆盖函数默认值
    final_pad_value = model_configs.get("pad_value", pad_value) # 最终使用的填充值

    print(f"最终模型参数: ntoken={ntoken}, d_model={d_model}, nhead={nhead}, d_hid={d_hid}, nlayers={nlayers}, dropout={current_dropout}")
    print(f"填充标记: {pad_token}, 填充值: {final_pad_value}, 输入分箱数: {n_input_bins}")

    # 实例化 scGPT 模型
    try:
        scgpt_model = TransformerModel(
            ntoken=ntoken, # 词汇表大小
            d_model=d_model, # 嵌入维度
            nhead=nhead, # 多头注意力头数
            d_hid=d_hid, # 前馈网络隐藏层维度
            nlayers=nlayers, # Transformer 主体层数
            nlayers_cls=nlayers_cls, # CLS 特征提取层数
            n_cls=n_cls, # CLS 标记数量
            vocab=vocab, # 传入加载的词汇表对象
            dropout=current_dropout, # dropout 率
            pad_token=pad_token, # 填充标记字符串
            pad_value=final_pad_value, # 填充值 (用于输入数据)
            input_emb_style=input_emb_style, # 输入嵌入风格
            n_input_bins=n_input_bins, # 输入分箱数
            cell_emb_style=cell_emb_style, # 细胞嵌入风格
            use_fast_transformer=use_fast_transformer, # 是否使用快速 Transformer
            pre_norm=pre_norm # 是否使用 Pre-Normalization
            # 如果 TransformerModel 期望其他来自 model_configs 的相关参数，在此处添加
        )
        print("scGPT TransformerModel 实例化成功。")
    except Exception as e:
        print(f"实例化 TransformerModel 时出错: {e}")
        return None

    # 加载预训练权重
    try:
        # 从 .pt 文件加载状态字典，映射到 CPU (后续可以移到 GPU)
        state_dict = torch.load(model_weights_file, map_location=torch.device('cpu'))
        # 使用 strict=False 允许新的反卷积头不在检查点中，
        # 并忽略其他潜在的不匹配 (例如，如果模型保存时带有不同的头)。
        missing_keys, unexpected_keys = scgpt_model.load_state_dict(state_dict, strict=False)
        print(f"已从 {model_weights_file} 加载预训练权重。")
        if missing_keys: # 如果有缺失的键
            print(f"  状态字典中缺失的键: {missing_keys}")
        if unexpected_keys: # 如果有意外的键
            # 过滤掉可能属于检查点中预先存在的头的键
            filtered_unexpected_keys = [k for k in unexpected_keys if not k.startswith("deconv_head") and not k.startswith("value_head")]
            if filtered_unexpected_keys: # 如果过滤后仍有意外的键
                 print(f"  警告: 状态字典中发现意外的键 (不包括典型的头名称): {filtered_unexpected_keys}")

    except Exception as e:
        print(f"从 {model_weights_file} 加载模型权重时出错: {e}")
        return None

    # 定义反卷积头 (deconvolution head)
    # 头的输入通常是来自 scGPT 的 CLS 标记嵌入 (维度为 d_model)
    # 输出是 n_cell_types 个细胞类型的概率
    scgpt_model.deconv_head = nn.Sequential( # 使用 nn.Sequential 容器定义序列模块
        nn.Linear(d_model, d_model // 2), # 第一个线性层，将维度从 d_model 降到 d_model // 2
        nn.ReLU(), # ReLU 激活函数
        nn.Dropout(head_dropout_rate), # Dropout 层，防止过拟合
        nn.Linear(d_model // 2, n_cell_types), # 第二个线性层，输出维度为 n_cell_types
        nn.Softmax(dim=-1)  # Softmax 层，沿最后一个维度计算概率分布
    )
    print(f"已添加反卷积头: Linear({d_model}, {d_model//2}) -> ReLU -> Dropout({head_dropout_rate}) -> Linear({d_model//2}, {n_cell_types}) -> Softmax")

    # 如果请求，则冻结基础模型参数
    if freeze_scgpt_base:
        print("正在冻结预训练的 scGPT 基础模型参数...")
        frozen_count = 0 # 冻结参数计数
        trainable_count = 0 # 可训练参数计数
        # 遍历模型的所有命名参数
        for name, param in scgpt_model.named_parameters():
            if 'deconv_head' not in name: # 如果参数不属于反卷积头
                param.requires_grad = False # 设置为不需要梯度，即冻结
                frozen_count += 1
            else: # 如果参数属于反卷积头
                param.requires_grad = True # 确保头参数是可训练的
                trainable_count +=1
        print(f"参数已冻结: {frozen_count} 个参数。可训练 (头): {trainable_count} 个参数。")
    else: # 如果不冻结基础模型
        print("所有模型参数 (scGPT 基础模型 + 反卷积头) 都将是可训练的。")
        for param in scgpt_model.parameters(): # 确保所有参数都可训练
            param.requires_grad = True
            
    return scgpt_model # 返回配置好的模型

# 当脚本作为主程序执行时的示例代码
if __name__ == '__main__':
    print("--- 反卷积模型创建示例 ---")

    # 定义占位符路径 (这些应由用户替换为实际路径)
    # 对于此示例，我们检查这些是默认占位符还是已被更改。
    user_model_dir = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/" # 用户指定的模型目录
    user_vocab_file = "YOUR_PATH_TO_VOCAB_JSON_FILE" # 用户指定的词汇表文件 (可以是 model_dir + "/vocab.json")

    # 如果占位符未更改，则回退到使用虚拟文件
    if user_model_dir == "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/" or \
       user_vocab_file == "YOUR_PATH_TO_VOCAB_JSON_FILE":
        
        print("\n模型/词汇表路径的占位符未设置。将使用虚拟文件进行演示。")
        # 定义虚拟模型目录路径
        dummy_model_dir = Path("./dummy_scgpt_model_dir")
        dummy_model_dir.mkdir(exist_ok=True) # 创建目录，如果已存在则不报错
        
        # 定义虚拟文件路径
        dummy_vocab_file = dummy_model_dir / "vocab.json"
        dummy_args_file = dummy_model_dir / "args.json"
        dummy_weights_file = dummy_model_dir / "best_model.pt"

        # 创建虚拟 args.json 文件内容
        dummy_args_content = {
            "embsize": 64, "nheads": 2, "nlayers": 1, "dropout": 0.1, "d_hid": 128,
            "n_input_bins": 0, "input_emb_style": "continuous", "cell_emb_style": "cls",
            "pad_token": "<pad>", "pad_value": -2, "n_cls": 1, "nlayers_cls":1,
            "use_fast_transformer": False, "pre_norm": False
        }
        # 将虚拟配置写入 args.json
        with open(dummy_args_file, "w") as f:
            json.dump(dummy_args_content, f)

        # 创建虚拟 vocab.json 文件内容
        dummy_vocab_content = {"<pad>": 0, "geneA": 1, "geneB": 2, "<cls>": 3, "<eoc>": 4}
        # 将虚拟词汇表写入 vocab.json
        with open(dummy_vocab_file, "w") as f:
            json.dump(dummy_vocab_content, f)

        # 创建一个虚拟的 best_model.pt (最小化的状态字典)
        try:
            # 使用 (可能是虚拟的) GeneVocab 和 TransformerModel 来创建一个可保存的状态字典
            temp_vocab = GeneVocab.from_file(dummy_vocab_file) # 加载虚拟词汇表
            # 实例化一个最小化的 scGPT 模型
            minimal_scgpt_model = TransformerModel(
                ntoken=len(temp_vocab), d_model=dummy_args_content["embsize"], 
                nhead=dummy_args_content["nheads"], d_hid=dummy_args_content["d_hid"],
                nlayers=dummy_args_content["nlayers"], vocab=temp_vocab,
                pad_token=dummy_args_content["pad_token"], pad_value=dummy_args_content["pad_value"]
            )
            torch.save(minimal_scgpt_model.state_dict(), dummy_weights_file) # 保存模型状态字典
            print(f"虚拟模型文件 (args, vocab, weights) 已在 {dummy_model_dir} 中创建")
            
            # 更新路径以在示例运行中使用这些虚拟文件
            model_dir_to_use = str(dummy_model_dir)
            vocab_file_to_use = str(dummy_vocab_file)
            can_run_example = True # 标记可以运行示例
        except Exception as e:
            print(f"严重错误: 无法为示例创建虚拟模型文件: {e}")
            print("如果 scGPT 或其依赖项未正确安装/可用，则可能发生这种情况。")
            print("跳过模型创建演示。")
            can_run_example = False # 标记不能运行示例
    else: # 如果用户提供了实际路径
        model_dir_to_use = user_model_dir
        vocab_file_to_use = user_vocab_file
        can_run_example = True # 标记可以运行示例
        print(f"\n使用用户提供的路径: model_dir='{model_dir_to_use}', vocab_file='{vocab_file_to_use}'")


    if can_run_example: # 如果可以运行示例 (无论是用虚拟文件还是真实文件)
        num_cell_types_example = 12  # 示例：用于反卷积的 12 种细胞类型
        
        print(f"\n尝试创建具有 n_cell_types={num_cell_types_example} 的反卷积模型...")
        # 调用核心函数创建反卷积模型实例
        deconv_model_instance = create_deconvolution_model(
            model_dir_path=model_dir_to_use, # 使用的模型目录路径
            vocab_file_path=vocab_file_to_use, # 使用的词汇表文件路径
            n_cell_types=num_cell_types_example, # 细胞类型数量
            freeze_scgpt_base=True, # 示例：冻结基础模型
            head_dropout_rate=0.2, # 示例：设置头部的 dropout 率
            # 覆盖模型参数的示例：
            # d_model_override=dummy_args_content["embsize"] if 'dummy_args_content' in locals() else 128 
        )

        if deconv_model_instance: # 如果模型实例成功创建
            print("\n--- 反卷积模型实例 ---")
            print(deconv_model_instance) # 打印模型结构
            
            print("\n--- 可训练参数检查 ---")
            total_params = 0 # 总参数量
            trainable_params = 0 # 可训练参数量
            # 遍历模型的命名参数
            for name, param in deconv_model_instance.named_parameters():
                total_params += param.numel() # 累加参数数量
                if param.requires_grad: # 如果参数需要梯度 (即可训练)
                    print(f"  可训练: {name} (大小: {param.numel()})")
                    trainable_params += param.numel() # 累加可训练参数数量
            print(f"模型总参数量: {total_params}")
            print(f"模型可训练参数量: {trainable_params}")
            # 根据 freeze_scgpt_base 的设置检查冻结是否符合预期
            if freeze_scgpt_base and trainable_params > 0 and total_params > trainable_params:
                 print("基础模型按预期冻结。")
            elif not freeze_scgpt_base and total_params == trainable_params:
                 print("完整模型按预期可训练。")

        else: # 如果模型创建失败
            print("\n反卷积模型创建失败。请查看日志以获取详细信息。")
            print("如果使用真实路径，请确保它们是正确的并且文件有效。")
            print("如果使用虚拟文件，这可能表明虚拟设置或 scGPT 可用性存在问题。")

    print("\n--- 反卷积模型创建示例结束 ---")
```
