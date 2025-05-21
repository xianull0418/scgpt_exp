# 导入 PyTorch 相关的库
import torch
import torch.nn as nn # 用于损失函数示例和虚拟模型
# 导入 NumPy 用于数值运算 (尽管在此脚本中直接使用较少，但通常与 PyTorch 一起使用)
import numpy as np
# 从 pathlib 导入 Path 对象，用于处理文件和目录路径
from pathlib import Path
# 导入 os 模块，用于与操作系统交互 (例如创建目录)
import os
# 导入 time 模块，用于计时 (例如计算每个 epoch 的持续时间)
import time
# from scipy.stats import pearsonr # 用于可选的评估指标，可以稍后添加
from typing import Optional # 用于类型提示，表示参数可以是 None

# 假设模型 (scGPT 基础模型 + 反卷积头)、数据加载器、优化器、损失函数、学习率调度器
# 都在其他地方定义，并作为参数传递给此函数。

# 定义运行微调循环的函数
def run_fine_tuning(
    model: nn.Module, # PyTorch 模型 (nn.Module)，期望包含 scGPT 基础模型和反卷积头
    train_loader: torch.utils.data.DataLoader, # 训练数据加载器
    valid_loader: torch.utils.data.DataLoader, # 验证数据加载器
    optimizer: torch.optim.Optimizer, # 优化器 (例如 AdamW, SGD)
    loss_fn: nn.Module, # 损失函数 (例如 L1Loss, MSELoss, KLDivLoss)
    n_epochs: int, # 要训练的总轮数
    device: torch.device, # 用于训练的设备 (例如 "cuda" 或 "cpu")
    loss_type_str: str, # 损失函数类型的字符串表示 (例如 "L1", "MSE", "KLDivergence")，用于特殊处理 (如 KL散度)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # 可选的学习率调度器
    checkpoint_dir: str = "./checkpoints_finetune", # 保存模型检查点的目录 (更改了默认值以避免冲突)
    best_model_name: str = "best_deconv_model_finetuned.pt" # 最佳模型的名称 (更改了默认值)
):
    """
    运行反卷积模型的微调循环。
    假设 'model' 是一个 nn.Module，其中 model.deconv_head 存在，
    并且主模型 (例如 scGPT 的 TransformerModel) 的前向传播
    返回一个包含 'cell_emb' 的字典。
    """
    print(f"开始在设备 {device} 上进行 {n_epochs} 轮的微调...")
    
    # 创建检查点目录 (如果不存在)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # 初始化最佳验证损失为正无穷大
    best_val_loss = float('inf')
    # 初始化历史记录字典，用于存储每轮的损失和学习率
    history = {"train_loss": [], "val_loss": [], "lr": []}

    # 开始训练循环，遍历指定的轮数
    for epoch in range(n_epochs):
        epoch_start_time = time.time() # 记录当前 epoch 开始时间
        
        # --- 训练阶段 ---
        model.train() # 将模型设置为训练模式 (启用 dropout, batchnorm 更新等)
        total_train_loss = 0.0 # 初始化当前 epoch 的总训练损失
        # 遍历训练数据加载器中的每个批次
        for batch_idx, batch in enumerate(train_loader):
            # 将批次数据移动到指定设备
            gene_ids = batch["gene_ids"].to(device) # 基因 ID
            values = batch["values"].to(device) # 基因表达值 (例如分箱后的值)
            padding_mask = batch["padding_mask"].to(device) # 填充掩码 (标记哪些是填充的基因)
            true_fractions = batch["fractions"].to(device) # 真实的细胞类型比例 (标签)

            optimizer.zero_grad() # 清除先前计算的梯度

            # 通过 scGPT 基础模型进行前向传播
            # model() 调用 scGPT TransformerModel 的 forward 方法。
            # 这应该返回一个字典。
            scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
            
            # 检查 scGPT 输出是否包含 'cell_emb'
            if "cell_emb" not in scgpt_output:
                # 此错误表示模型结构或调用方式不匹配。
                # 'create_deconvolution_model' 脚本假定 scGPT 的 TransformerModel 是基础模型。
                raise KeyError("scGPT 模型的输出不包含 'cell_emb'。"
                               "确保在 scGPT 模型初始化期间 `cell_emb_style` 为 'cls' "
                               "并且模型的前向传播返回它。")
            
            cell_embedding = scgpt_output["cell_emb"] # 获取细胞嵌入 (通常是 CLS 标记的嵌入)

            # 将细胞嵌入传递给反卷积头
            # model.deconv_head 应该在 'create_deconvolution_model' 中附加
            if not hasattr(model, 'deconv_head'):
                 raise AttributeError("模型没有 'deconv_head' 属性。确保它已被附加。")
            predicted_fractions = model.deconv_head(cell_embedding) # 获取预测的细胞类型比例

            # 计算损失
            if loss_type_str == "KLDivergence":
                # KLDivLoss 期望输入为对数概率，目标为概率
                # deconv_head 的输出是 softmax (概率)
                # 为数值稳定性，在 torch.log 中添加一个小的 epsilon
                loss = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
            else: # 其他损失函数 (如 L1Loss, MSELoss) 直接使用预测值和真实值
                loss = loss_fn(predicted_fractions, true_fractions)
            
            loss.backward() # 反向传播，计算梯度
            # 可选：如果需要，进行梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # 更新模型参数

            # 更新学习率 (针对按步更新的调度器，如 OneCycleLR 或自定义的线性预热 LambdaLR)
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            total_train_loss += loss.item() # 累加批次损失 ( .item() 获取 Python 数值)
            
            # 每个 epoch 内记录几次训练进度
            if len(train_loader) > 4 and batch_idx % (len(train_loader) // 4) == 0 and batch_idx > 0 :
                 current_lr_batch = optimizer.param_groups[0]['lr'] # 获取当前学习率
                 print(f"  Epoch {epoch+1}/{n_epochs} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} | LR: {current_lr_batch:.6e}")


        avg_train_loss = total_train_loss / len(train_loader) # 计算平均训练损失
        history["train_loss"].append(avg_train_loss) # 记录当前 epoch 的训练损失
        history["lr"].append(optimizer.param_groups[0]['lr']) # 记录当前 epoch 结束时的学习率


        # --- 验证阶段 ---
        model.eval() # 将模型设置为评估模式 (禁用 dropout, batchnorm 不更新等)
        total_val_loss = 0.0 # 初始化当前 epoch 的总验证损失
        # all_preds_val = [] # 如果需要更详细的指标，可以取消注释以存储预测
        # all_targets_val = [] # 如果需要更详细的指标，可以取消注释以存储目标
        with torch.no_grad(): # 在此块内不计算梯度，以节省内存和计算
            # 遍历验证数据加载器中的每个批次
            for batch in valid_loader:
                # 将批次数据移动到指定设备
                gene_ids = batch["gene_ids"].to(device)
                values = batch["values"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                true_fractions = batch["fractions"].to(device)

                # 前向传播
                scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
                if "cell_emb" not in scgpt_output: # 与训练阶段相同的处理
                     raise KeyError("验证: scGPT 模型的输出不包含 'cell_emb'。")
                
                cell_embedding = scgpt_output["cell_emb"]
                predicted_fractions = model.deconv_head(cell_embedding)
                
                # 计算验证损失
                if loss_type_str == "KLDivergence":
                    val_loss_batch = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
                else:
                    val_loss_batch = loss_fn(predicted_fractions, true_fractions)
                total_val_loss += val_loss_batch.item() # 累加批次验证损失
                
                # 如果要计算更复杂的指标，存储预测和目标
                # all_preds_val.append(predicted_fractions.cpu().numpy())
                # all_targets_val.append(true_fractions.cpu().numpy())

        avg_val_loss = total_val_loss / len(valid_loader) # 计算平均验证损失
        history["val_loss"].append(avg_val_loss) # 记录当前 epoch 的验证损失
        
        epoch_duration = time.time() - epoch_start_time # 计算当前 epoch 的持续时间
        current_lr_epoch_end = optimizer.param_groups[0]['lr'] # 获取 epoch 结束时的学习率
        # 打印当前 epoch 的摘要信息
        print(f"Epoch {epoch+1}/{n_epochs} Summary: "
              f"Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr_epoch_end:.6e} | Duration: {epoch_duration:.2f}s")

        # 更新学习率 (针对按轮更新的调度器，如 ReduceLROnPlateau)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss) # ReduceLROnPlateau 需要验证损失作为参数
            # ReduceLROnPlateau 可能会更改学习率，如果其 verbose=True 则会打印，或手动检查
            # print(f"  ReduceLROnPlateau: Current LR {optimizer.param_groups[0]['lr']:.6e}")


        # 模型检查点 (Model checkpointing)
        # 如果当前验证损失优于历史最佳验证损失，则保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss # 更新最佳验证损失
            checkpoint_path = Path(checkpoint_dir) / best_model_name # 构建保存路径
            try:
                torch.save(model.state_dict(), checkpoint_path) # 保存模型的状态字典
                print(f"  最佳模型已保存至 {checkpoint_path} (验证损失: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  保存模型时出错: {e}")
            
    print(f"\n微调完成。最佳验证损失: {best_val_loss:.4f}")
    return history # 返回包含训练历史的字典


# 当脚本作为主程序执行时的示例代码块
if __name__ == '__main__':
    print("--- 开始微调循环示例 ---")

    # --- 为演示模拟必要的组件 ---
    # 这个 DummyModel 需要与 run_fine_tuning 函数调用它的方式兼容。
    # 具体来说，model(src, values, padding_mask) 应该返回一个包含 "cell_emb" 的字典，
    # 并且 model.deconv_head 应该存在。
    class DummyDeconvolutionModel(nn.Module): # 定义一个虚拟的反卷积模型
        def __init__(self, d_model=64, n_genes_vocab=100, n_cell_types=5, pad_value=-2):
            super().__init__()
            self.d_model = d_model # 模型维度
            self.pad_value = pad_value # 填充值 (在这个简化的虚拟模型的前向传播中未使用)
            
            # 模拟 scGPT 核心 TransformerModel 部分
            # 它应该接受 src, values, src_key_padding_mask
            # 为简单起见，这个虚拟模型将使用 'values' 生成 'cell_emb'。
            # 真实的 TransformerModel 会使用 src (基因ID) 查找嵌入，
            # 然后与 'values' (分箱后的表达) 结合。
            self.mock_transformer_encoder = nn.Linear(n_genes_vocab, d_model) # 输入大小 = 序列长度
            
            # 反卷积头
            self.deconv_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Linear(d_model // 2, n_cell_types), nn.Softmax(dim=-1)
            )
        
        # 虚拟模型的前向传播方法
        def forward(self, src, values, src_key_padding_mask):
            # src: [批次大小, 序列长度] (基因ID)
            # values: [批次大小, 序列长度] (分箱后的表达)
            # src_key_padding_mask: [批次大小, 序列长度] (布尔值, True 表示填充)
            
            # 在真实的 scGPT 模型中，如果 cell_emb_style='cls'，
            # 来自 Transformer 层的 CLS 标记的输出嵌入将是 'cell_emb'。
            # 这个虚拟模型通过处理 'values' 张量来模拟这一点。
            # 我们可以假设 'values' 代表 CLS 标记的特征（如果取平均值），
            # 或者我们可以直接投射整个序列。
            # 对于这个虚拟模型，让我们在传递给线性层之前对序列维度上的 'values' 进行平均，
            # 以使其在某种程度上独立于序列长度。
            # 我们必须小心：src_key_padding_mask 应用于忽略填充值。
            
            # 基于 src_key_padding_mask 创建非填充值的掩码
            # 掩码中 True 表示填充，因此取反以获得有效值
            valid_values_mask = ~src_key_padding_mask.unsqueeze(-1) # [B, S, 1]
            
            # 将掩码应用于值 (将填充值设置为0以便求和/平均)
            masked_values = values.unsqueeze(-1) * valid_values_mask # [B, S, 1]
            
            # 对有效值求和并计数
            sum_values = masked_values.sum(dim=1) # [B, 1]
            num_valid_values = valid_values_mask.sum(dim=1) # [B, 1]
            num_valid_values = torch.clamp(num_valid_values, min=1.0) # 避免除以零
            
            # 有效值的平均值 (序列的一个非常粗略的“嵌入”)
            mean_sequence_value_features = sum_values / num_valid_values # [B,1]
            
            # 要使其成为 [B, d_model]，这个简单的虚拟模型需要更多工作。
            # 为简单起见，我们只在原始 'values' 上使用 mock_transformer_encoder，
            # 假设 'values' 具有正确的维度 (n_genes_vocab)。
            # 这是一个极大的简化。
            if values.shape[1] != self.mock_transformer_encoder.in_features:
                 # 如果数据的 seq_len != 模型定义中的 n_genes_vocab，将会出错。
                 # 这突显了仔细对齐虚拟数据/模型的必要性。
                 # 对于此示例，DataLoader 中的 DUMMY_N_GENES_VOCAB 应与此处的 n_genes_vocab 匹配。
                 raise ValueError(f"虚拟模型输入维度不匹配: values.shape[1]={values.shape[1]}, 期望 {self.mock_transformer_encoder.in_features}")

            mock_cell_emb = self.mock_transformer_encoder(values) # [批次大小, d_model]
            
            return {"cell_emb": mock_cell_emb} # 返回包含模拟细胞嵌入的字典

    class DummyDataLoader: # 定义一个虚拟的数据加载器
        def __init__(self, num_batches, batch_size, n_genes_vocab, n_cell_types, device, is_train=True):
            self.num_batches = num_batches # 批次数量
            self.batch_size = batch_size # 批次大小
            self.n_genes_vocab = n_genes_vocab # 基因词汇表大小 (也用作虚拟模型的特征数)
            self.n_cell_types = n_cell_types # 细胞类型数量
            self.device = device # 设备
            self.is_train = is_train # 是否为训练加载器

        def __len__(self): # 返回批次数量
            return self.num_batches

        def __iter__(self): # 返回一个迭代器
            for _ in range(self.num_batches):
                # 生成虚拟数据
                # gene_ids 在这个特定虚拟模型的前向传播中未使用，但它是规范的一部分
                gene_ids = torch.randint(0, self.n_genes_vocab, (self.batch_size, self.n_genes_vocab), device=self.device, dtype=torch.long)
                # 这个虚拟模型的 values 应该是 [批次大小, n_genes_vocab]
                values_for_dummy = torch.rand(self.batch_size, self.n_genes_vocab, device=self.device, dtype=torch.float)
                padding_mask = torch.zeros(self.batch_size, self.n_genes_vocab, device=self.device, dtype=torch.bool)
                # 示例：为某些样本填充最后10个特征
                if self.n_genes_vocab > 10 and self.is_train: 
                    padding_mask[0, -10:] = True 
                
                fractions = torch.softmax(torch.rand(self.batch_size, self.n_cell_types, device=self.device), dim=-1)
                yield {"gene_ids": gene_ids, "values": values_for_dummy, "padding_mask": padding_mask, "fractions": fractions}
    
    # --- 虚拟运行的参数 ---
    DUMMY_N_EPOCHS = 2 # 减少轮数以便快速测试
    DUMMY_LR = 1e-3 # 稍高的学习率以便更快看到变化
    DUMMY_BATCH_SIZE = 8 
    DUMMY_N_GENES_VOCAB = 100 # 词汇表中的基因数，也用作虚拟模型输入的序列长度
    DUMMY_N_CELL_TYPES = 5 # 虚拟细胞类型数
    DUMMY_D_MODEL = 32 # 虚拟模型的 d_model
    
    # 获取可用的设备 (GPU 或 CPU)
    dummy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化虚拟组件
    example_model = DummyDeconvolutionModel( # 创建虚拟模型实例
        d_model=DUMMY_D_MODEL, 
        n_genes_vocab=DUMMY_N_GENES_VOCAB, 
        n_cell_types=DUMMY_N_CELL_TYPES
    ).to(dummy_device) # 将模型移动到设备
        
    # 创建虚拟训练和验证数据加载器实例
    example_train_loader = DummyDataLoader(10, DUMMY_BATCH_SIZE, DUMMY_N_GENES_VOCAB, DUMMY_N_CELL_TYPES, dummy_device)
    example_valid_loader = DummyDataLoader(5, DUMMY_BATCH_SIZE, DUMMY_N_GENES_VOCAB, DUMMY_N_CELL_TYPES, dummy_device, is_train=False)
    
    # 确保虚拟模型的所有参数都是可训练的
    example_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, example_model.parameters()), lr=DUMMY_LR)
    
    example_loss_fn = nn.L1Loss() # 使用 L1 损失作为示例
    example_loss_type = "L1" # 与 example_loss_fn 匹配
    
    # 使用 OneCycleLR 调度器的示例
    total_steps_for_scheduler = DUMMY_N_EPOCHS * len(example_train_loader) # 计算总步数
    example_scheduler = torch.optim.lr_scheduler.OneCycleLR( # 创建学习率调度器实例
        example_optimizer, 
        max_lr=DUMMY_LR, 
        total_steps=total_steps_for_scheduler
    )
    # example_scheduler = None # 如果不想使用调度器，可以取消注释此行

    print(f"\n已为设备创建虚拟组件: {dummy_device}")
    print(f"  虚拟模型: {type(example_model)}")
    print(f"  虚拟 train_loader: {len(example_train_loader)} 个批次，批次大小 {DUMMY_BATCH_SIZE}")
    print(f"  虚拟 valid_loader: {len(example_valid_loader)} 个批次，批次大小 {DUMMY_BATCH_SIZE}")
    print(f"  优化器: {type(example_optimizer)}")
    print(f"  调度器: {type(example_scheduler)}")
    print(f"  损失函数: {type(example_loss_fn)} (类型: {example_loss_type})")


    # --- 运行微调循环 ---
    print("\n使用虚拟组件调用 run_fine_tuning...")
    try:
        # 调用核心的微调函数
        training_history = run_fine_tuning(
            model=example_model,
            train_loader=example_train_loader,
            valid_loader=example_valid_loader,
            optimizer=example_optimizer,
            loss_fn=example_loss_fn,
            n_epochs=DUMMY_N_EPOCHS,
            device=dummy_device,
            loss_type_str=example_loss_type,
            scheduler=example_scheduler,
            checkpoint_dir="./dummy_checkpoints_finetune", # 为此测试指定唯一的目录
            best_model_name="dummy_best_model_finetuned.pt"
        )
        print("\n微调循环示例成功完成。")
        print(f"训练历史: {training_history}")
    except Exception as e: # 捕获任何异常
        print(f"虚拟微调循环期间出错: {e}")
        import traceback # 导入 traceback 模块以打印详细的错误信息
        traceback.print_exc() # 打印异常的堆栈跟踪
    finally: # 无论是否发生异常，都执行此块
        # 清理虚拟检查点目录 (可选)
        # import shutil
        # dummy_ckpt_path = Path("./dummy_checkpoints_finetune")
        # if dummy_ckpt_path.exists():
        #     shutil.rmtree(dummy_ckpt_path)
        #     print(f"已清理 {dummy_ckpt_path} 目录。")
            
    print("\n提醒: 此示例使用虚拟组件。实际的微调需要真实模型、数据和配置。")

```
