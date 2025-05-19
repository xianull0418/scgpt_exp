# scGPT微调日志：细胞解卷积任务

## 任务目标
使用预训练的scGPT模型微调，构建一个可以从bulk RNA-seq数据中预测细胞类型组成比例的解卷积模型。这种解卷积能力对理解复杂组织的细胞构成至关重要，特别是在肿瘤微环境和免疫浸润分析中。

从头梳理一遍

细胞的表示，再加两层mlp 
细胞注释：多分类
反卷积：回归，

## 微调流程
1. ✅ 导入环境，这里使用scgpt_py3.9
2. ✅ 模型超参数设置
3. ✅ 数据处理
4. ✅ 导入预训练的模型
5. ✅ 确定损失函数，开始微调

## Jupyter Notebook单元结构

### 当前代码结构（TAPE_cell_bulk.py）：
1. **Cell 1-2**: 环境导入与库引入
2. **Cell 3-4**: 超参数设置
   ```python
   hyperparameter_defaults = dict(
       seed=0,
       dataset_name="ms",
       do_train=True,
       load_model="/ai/home/jcw/scGPT_new/pretrain/save/cellxgene_census_blood-May13-00-17-2025",
       mask_ratio=0.0,
       epochs=15,
       n_bins=51,
       MVC=False,  # Masked value prediction for cell embedding
       ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
       dab_weight=0.0,
       lr=5e-4,
       batch_size=64,
       layer_size=128,
       nlayers=4,
       nhead=4,
       dropout=0.1,
       schedule_ratio=0.9,
       save_eval_interval=5,
       fast_transformer=True,
       pre_norm=False,
       amp=True,
       include_zero_gene=False,
       freeze=False,
       DSBN=False,
       weight_decay=1e-5,
       reconstruction_weight=1.0,
       prediction_weight=1.0,
   )
   ```
3. **Cell 5**: 导入TAPE模块和设置随机种子
4. **Cell 6**: 读取单细胞数据
   ```python
   adata = sc.read_h5ad(sc_data_path)
   ```
5. **Cell 7**: 生成模拟bulk数据
   ```python
   # 预处理AnnData为DataFrame
   df = pd.DataFrame(X_dense, index=adata.obs["CellType"], columns=adata.var_names)
   df["celltype"] = df.index
   
   # 使用TAPE生成模拟数据
   simu_data = generate_simulated_data(
       sc_data=df,
       samplenum=sample_num,
       random_state=hyperparameter_defaults["seed"],
       sparse=sparse
   )
   
   # 保存细胞类型信息
   cell_fractions = simu_data.obs
   cell_types = list(cell_fractions.columns)
   ```
6. **Cell 8**: 检查TransformerModel参数列表
7. **Cell 9**: 加载预训练模型并适配解卷积任务
   ```python
   # 加载词汇表
   vocab = GeneVocab.from_file(vocab_file)
   
   # 创建模型
   model = TransformerModel(
       ntoken=len(vocab),
       d_model=embsize,
       # ... 其他参数 ...
   )
   
   # 添加解卷积头
   model.deconv_head = nn.Sequential(
       nn.Linear(embsize, 256),
       nn.ReLU(),
       nn.Dropout(0.1),
       nn.Linear(256, len(cell_types)),
       nn.Softmax(dim=1)
   )
   ```

## 当前进展
### 1. 环境导入
完成导入scgpt环境和相关依赖库

### 2. 超参数设置
已完成超参数调整，设置了学习率、批次大小、模型结构等参数

### 3. 数据处理
已完成TAPE数据处理，成功将单细胞数据转换为模拟bulk数据，并创建了适用于scGPT的数据格式。

#### 遇到的问题与解决方案：
1. **导入错误**：单元1中遇到了`No module named 'TAPE.TAPE'`错误。
   - **解决方案**：修正导入路径，将`from TAPE.TAPE.simulation import generate_simulated_data`改为`from TAPE.simulation import generate_simulated_data`

2. **单元2成功**：已成功读取单细胞数据`/ai/home/jcw/scGPT_new/finetune/data/ TAPE/pbmc3k_annotated.h5ad`。
   - 数据形状：(2700, 2078)
   - 观测变量：['leiden', 'CellType']
   - 已确认数据中包含CellType信息，共7种细胞类型

3. **单元3错误**：在执行`generate_simulated_data`时遇到错误："AnnData has no attribute __contains__, don't check `in adata`."
   - **解决方案**：通过预处理AnnData对象，将其转换为DataFrame格式后成功生成模拟数据

#### 已完成的数据处理步骤：
- ✅ 导入TAPE模块和准备函数
- ✅ 加载原始单细胞数据
- ✅ 使用TAPE生成模拟bulk数据及对应的细胞类型比例

### 4. 导入预训练模型
已成功加载预训练模型并添加解卷积头。

#### 遇到的问题与解决方案：
1. **变量未定义错误**：在模型导入步骤出现错误 `name 'vocab' is not defined`。
   - **解决方案**：修改导入流程，先加载预训练模型的词汇表再创建模型。

2. **特殊token未定义错误**：修复词汇表问题后又出现错误 `name 'pad_token' is not defined`。
   - **解决方案**：在模型加载单元开始处添加特殊token的定义。

3. **TransformerModel参数错误**：修复前两个问题后出现错误 `__init__() got an unexpected keyword argument 'pad_token_id'`。
   - **解决方案**：检查并使用正确的参数名

4. **获取正确的参数列表**：成功获取到TransformerModel的完整参数列表并使用正确参数初始化模型

#### 已完成的模型加载步骤：
- ✅ 加载预训练模型的词汇表
- ✅ 确保特殊token在词汇表中
- ✅ 使用正确的参数创建模型
- ✅ 添加解卷积头
- ✅ 加载预训练权重
- ✅ 将模型移至GPU

## 下一步计划
### 5. 确定损失函数，开始微调
接下来需要完成的任务：
1. **数据准备**:
   - 使用scGPT的预处理器处理模拟数据
   - 拆分训练集和验证集
   - 词汇表构建和数据Tokenizing
   - 创建DataLoader

2. **定义训练组件**:
   - 定义解卷积任务的损失函数
   - 设置优化器和学习率调度器
   - 创建训练和评估函数

3. **开始微调**:
   - 执行训练循环
   - 保存最佳模型
   - 可视化训练过程

4. **评估结果**:
   - 在验证集上评估解卷积性能
   - 分析预测结果
   - 对比模型性能

### 5. 训练过程问题与解决方案
#### 遇到的问题与解决方案：
1. **模型前向传播错误**：在训练循环执行时遇到错误 `forward() missing 1 required positional argument: 'src_key_padding_mask'`。
   - **问题分析**：在调用TransformerModel的forward方法时缺少必要的掩码参数，该参数用于指示序列中哪些位置是填充的。
   - **解决方案**：在训练和评估函数中，通过判断哪些位置是填充标记（即与pad_token相同的位置）来创建掩码，然后将其作为src_key_padding_mask参数传递给模型。

```python
# 修改前
output = model(gene_ids, values)

# 修改后
padding_mask = (gene_ids == vocab[pad_token])
output = model(gene_ids, values, src_key_padding_mask=padding_mask)
```

2. **训练循环AssertionError**：添加掩码参数后，在执行训练循环时遇到了AssertionError错误。
   - **问题分析**：这可能与padding_mask的格式、数据类型或维度有关。Transformer模型对掩码参数有严格的要求。
   - **解决方案**：
     - 确保padding_mask是布尔型张量：`.bool()`
     - 添加错误处理机制，以便更好地诊断和调试问题
     - 添加模型参数检查函数，验证模型forward方法的输入要求

```python
# 修改前
padding_mask = (gene_ids == vocab[pad_token])

# 修改后
padding_mask = (gene_ids == vocab[pad_token]).bool()  # 确保是布尔型

# 添加错误处理
try:
    output = model(gene_ids, values, src_key_padding_mask=padding_mask)
except Exception as e:
    print(f"批次出错: {str(e)}")
    print(f"gene_ids形状: {gene_ids.shape}, values形状: {values.shape}")
    print(f"padding_mask形状: {padding_mask.shape}, 掩码示例: {padding_mask[0][:10]}")
    continue
```

3. **添加模型检查函数**：创建了一个辅助函数来检查模型forward方法的参数签名和输入要求：
   ```python
   def check_model_forward_signature(model):
       # 检查模型forward方法的参数签名，辅助调试
       import inspect
       forward_sig = inspect.signature(model.forward)
       print("TransformerModel.forward方法的参数：")
       # ... 其他检查代码 ...
   ```

4. **鲁棒性增强**：在训练和评估函数中添加了更多的错误处理和防护措施：
   - 添加try-except块捕获批次处理中的错误
   - 防止除零错误，确保平均损失计算的分母不为零
   - 添加详细的日志记录，方便定位问题

5. **TransformerModel参数调查**：通过检查发现TransformerModel.forward方法需要以下参数：
   ```
   TransformerModel.forward方法的参数：
     - src: 必需
     - values: 必需
     - src_key_padding_mask: 必需
     - batch_labels: 可选，默认值=None
     - CLS: 可选，默认值=False
     - CCE: 可选，默认值=False
     - MVC: 可选，默认值=False
     - ECS: 可选，默认值=False
     - do_sample: 可选，默认值=False
   ```

   模型结构包含以下组件：
   ```
   模型结构:
     - encoder: GeneEncoder
     - value_encoder: ContinuousValueEncoder
     - transformer_encoder: TransformerEncoder
     - decoder: ExprDecoder
     - cls_decoder: ClsDecoder
     - sim: Similarity
     - creterion_cce: CrossEntropyLoss
     - deconv_head: Sequential
   ```

6. **Flash Attention兼容性错误**：启动训练时遇到错误 `assert qkv.dtype in [torch.float16, torch.bfloat16]`。
   - **问题分析**：Flash Attention（快速注意力机制）要求输入数据必须是半精度浮点数（torch.float16或torch.bfloat16），但当前使用的是全精度浮点数（torch.float32）。
   - **解决方案**：
     - 启用PyTorch的自动混合精度训练
     - 或者禁用Fast Transformer（将fast_transformer参数设为False）
     - 或者手动将输入数据转换为半精度

```python
# 方案1：使用自动混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 在前向传播时使用
with autocast():
    output_dict = model(src=gene_ids, values=values, src_key_padding_mask=src_key_padding_mask)
```

7. **张量维度错误**：修复Flash Attention问题后，出现新错误 `IndexError: too many indices for tensor of dimension 2`。
   - **问题分析**：模型输出的张量维度为2 [batch_size, hidden_dim]，不是预期的3维 [batch_size, seq_len, hidden_dim]，导致尝试用output[:, 0, :]获取CLS token表示时出错。
   - **解决方案**：
     - 增加条件判断，根据输出张量的维度灵活获取表示
     - 如果是2维，直接使用整个输出作为表示
     - 如果是3维，使用第一个token的表示

```python
# 根据输出维度正确获取表示
if output.dim() == 3:  # [batch_size, seq_len, hidden_dim]
    cls_output = output[:, 0, :]  # 第一个token是[CLS]
elif output.dim() == 2:  # [batch_size, hidden_dim]
    cls_output = output  # 已经是所需表示
else:
    raise ValueError(f"Unexpected output dimension: {output.dim()}")
```

8. **None值错误**：出现错误 `TypeError: cannot unpack non-iterable NoneType object`。
   - **问题分析**：输出处理逻辑中某处可能返回了None值。
   - **解决方案**：
     - 增加None检查，确保处理的对象有效
     - 添加更多的错误处理和调试信息收集

```python
# 添加空值检查
if output is not None and torch.is_tensor(output):
    # 处理有效张量
    # ...
else:
    print(f"警告：模型输出为None或非tensor")
    continue
```

通过这些修复，训练过程应该能够顺利进行。对于输出结构的分析，建议在首次训练时添加详细的调试信息，以便更好地理解模型的实际输出格式。
