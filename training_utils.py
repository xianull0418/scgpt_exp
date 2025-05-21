# 导入 PyTorch 相关的库
import torch
import torch.nn as nn # PyTorch 神经网络模块，用于定义损失函数和模型层
import torch.optim as optim # PyTorch 优化器模块，包含各种优化算法
from typing import Optional, Union # 从 typing 模块导入 Optional 和 Union，用于类型提示

# 定义获取损失函数的辅助函数
def get_loss_function(loss_type: str = "L1") -> nn.Module:
    """
    根据指定的类型返回一个 PyTorch 损失函数实例。
    支持的类型: 'L1', 'MSE', 'KLDivergence'。

    Args:
        loss_type (str): 损失函数的类型。

    Returns:
        nn.Module: PyTorch 损失函数实例。
    """
    if loss_type == "L1":
        # L1 损失，也称为平均绝对误差 (Mean Absolute Error, MAE)
        print("使用 L1 损失 (平均绝对误差)。")
        return nn.L1Loss()
    elif loss_type == "MSE":
        # MSE 损失，也称为均方误差 (Mean Squared Error)
        print("使用 MSE 损失 (均方误差)。")
        return nn.MSELoss()
    elif loss_type == "KLDivergence":
        # KL 散度损失 (Kullback-Leibler Divergence Loss)
        print("使用 KL 散度损失。")
        print("  注意: 对于 nn.KLDivLoss，模型输出应为对数概率 (例如，LogSoftmax 的输出)")
        print("  目标应为概率 (例如，来自目标组分的 Softmax 输出)。")
        print("  当前反卷积模型的头部以 Softmax 结束。")
        print("  训练循环必须在将模型输出传递给此损失函数之前对其应用 log()。")
        # 'batchmean' 表示损失首先按元素求和，然后除以批次大小。
        # 如果每个样本的损失幅度很重要，则可能首选 'sum'。
        return nn.KLDivLoss(reduction='batchmean') 
    else:
        # 如果损失函数类型不受支持，则抛出 ValueError
        raise ValueError(f"不支持的损失类型: {loss_type}。请从 'L1', 'MSE', 'KLDivergence' 中选择。")

# 定义获取优化器的辅助函数
def get_optimizer(
    model: nn.Module, # 需要优化的模型
    learning_rate: float = 1e-4, # 学习率，默认为 1e-4
    optimizer_type: str = "AdamW", # 优化器类型，默认为 "AdamW"
    weight_decay: float = 1e-5 # 权重衰减 (L2 正则化)，默认为 1e-5
) -> optim.Optimizer:
    """
    返回一个 PyTorch 优化器实例。
    支持的类型: 'AdamW', 'Adam'。

    Args:
        model (nn.Module): 其参数将被优化的模型。
        learning_rate (float): 学习率。
        optimizer_type (str): 优化器的类型。
        weight_decay (float): 权重衰减 (L2 惩罚)。

    Returns:
        optim.Optimizer: PyTorch 优化器实例。
    """
    if optimizer_type == "AdamW":
        # AdamW 是 Adam 优化器的一个变体，它改进了权重衰减的处理方式。
        print(f"使用 AdamW 优化器，学习率={learning_rate}, 权重衰减={weight_decay}。")
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        # Adam (Adaptive Moment Estimation) 是一种常用的优化算法。
        print(f"使用 Adam 优化器，学习率={learning_rate}, 权重衰减={weight_decay}。")
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # 如果优化器类型不受支持，则抛出 ValueError
        raise ValueError(f"不支持的优化器类型: {optimizer_type}。请选择 'AdamW' 或 'Adam'。")

# 定义获取学习率调度器的辅助函数
def get_scheduler(
    optimizer: optim.Optimizer, # 优化器实例
    scheduler_type: Optional[str] = "OneCycleLR", # 调度器类型，默认为 "OneCycleLR"
    total_steps: Optional[int] = None, # 总训练步数，对于 OneCycleLR 和 LinearWarmup 类型是必需的
    warmup_steps: int = 0, # 预热步数，用于 OneCycleLR (通过 pct_start 计算) 和 LinearWarmup
    # num_epochs: Optional[int] = None, # 如果知道每个 epoch 的步数，这是指定 total_steps 的另一种方式
    # initial_lr: Optional[float] = None # 此处未使用，因为 OneCycleLR 从优化器获取 max_lr
) -> Optional[torch.optim.lr_scheduler._LRScheduler]: # 返回调度器实例或 None
    """
    返回一个 PyTorch 学习率调度器实例。
    支持的类型: 'OneCycleLR', 'LinearWarmup', 'ReduceLROnPlateau', None。

    Args:
        optimizer: 优化器实例。
        scheduler_type: 调度器的类型。
        total_steps: 总训练步数。对于 OneCycleLR 和 LinearWarmup 是必需的。
        warmup_steps: 预热的步数。由 OneCycleLR (计算为 pct_start) 和 LinearWarmup 使用。
        
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: 调度器实例或 None。
    """
    if scheduler_type is None or scheduler_type.lower() == "none":
        # 如果 scheduler_type 为 None 或 "none" (不区分大小写)，则不使用调度器
        print("不使用学习率调度器。")
        return None
        
    if scheduler_type == "OneCycleLR":
        # OneCycleLR 调度器根据 1cycle 策略调整学习率。
        if total_steps is None:
            # 对于 OneCycleLR，必须提供 total_steps
            raise ValueError("对于 OneCycleLR，必须提供 'total_steps'。")
        
        # pct_start 是用于预热的 total_steps 的一部分。
        # 如果给定了 warmup_steps，则计算 pct_start。否则，使用 OneCycleLR 的默认值 (0.3)。
        pct_start = float(warmup_steps) / float(total_steps) if warmup_steps > 0 and total_steps > 0 else 0.3
        
        # 确保 pct_start 在 OneCycleLR 的合理范围内 (例如，如果使用预热，则 >0 且 <1)
        if not (0 < pct_start < 1.0) and warmup_steps > 0 :
             print(f"警告: OneCycleLR 的 pct_start ({pct_start:.3f}) 不寻常。请确保 warmup_steps 和 total_steps 合理。")
        if pct_start == 0 and warmup_steps > 0: # 如果 warmup_steps 相对于 total_steps 非常小，避免除以零
            pct_start = 0.001 # 一个非常小的预热比例
        
        print(f"使用 OneCycleLR 调度器，最大学习率 (来自优化器)={optimizer.defaults['lr']}, 总步数={total_steps}, 预热比例={pct_start:.3f}。")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults['lr'], # OneCycleLR 将此用作峰值学习率。
            total_steps=total_steps, # 总步数
            pct_start=pct_start, # 预热阶段所占的比例
            anneal_strategy='cos', # 退火策略，常用 'cos' (余弦退火)，也可以是 'linear'
            div_factor=25,         # 决定初始学习率 = max_lr / div_factor
            final_div_factor=1e4   # 决定最小学习率 = 初始学习率 / final_div_factor
        )
    elif scheduler_type == "LinearWarmup":
        # 此调度器提供从一个较小的学习率线性预热到优化器的初始学习率。
        # 预热后，学习率可以衰减或保持不变。这里设置为保持不变。
        if warmup_steps <= 0:
            print("选择了 LinearWarmup，但 warmup_steps 为 0。调度器实际上将是恒定学习率。")
            # 返回一个什么都不做或保持学习率不变的调度器
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: 1.0)

        # 定义学习率乘子函数
        def lr_lambda(current_step: int):
            if current_step < warmup_steps: # 如果当前步数小于预热步数
                return float(current_step + 1) / float(warmup_steps) # 学习率从接近 0 线性增加到 1
            return 1.0 # 预热后，学习率乘子为 1.0 (即使用优化器的初始学习率)

        print(f"使用 LambdaLR 进行线性预热，共 {warmup_steps} 步，之后为恒定学习率。")
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "ReduceLROnPlateau":
        # ReduceLROnPlateau 调度器在某个指标 (例如验证损失)停止改善时降低学习率。
        print("使用 ReduceLROnPlateau 调度器。监控验证指标 (例如损失)。")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',    # 当监控的量停止减少时降低学习率
            factor=0.1,    # 学习率降低的因子 (new_lr = lr * factor)
            patience=10,   # 在降低学习率之前，没有改善的 epoch/step 数
            verbose=True   # 如果为 True，则在每次更新时向 stdout 输出一条消息
        )
    else:
        # 如果调度器类型不受支持，则抛出 ValueError
        raise ValueError(f"不支持的调度器类型: {scheduler_type}。请选择 'OneCycleLR', 'LinearWarmup', 'ReduceLROnPlateau', 或 None。")

# 当此脚本作为主程序执行时，运行以下示例代码
if __name__ == '__main__':
    print("--- 损失函数、优化器和调度器设置示例 ---")

    # 用于实例化的虚拟模型 (例如一个简单的线性层)
    # 在实际场景中，这应该是您基于 scGPT 的反卷积模型
    dummy_model = nn.Linear(in_features=100, out_features=10) # 示例：100 个输入特征，10 个输出类别/组分
    print(f"\n已创建虚拟模型: {type(dummy_model)}")

    # --- 1. 损失函数示例 ---
    print("\n--- 损失函数示例 ---")
    l1_loss_fn = get_loss_function(loss_type="L1") # 获取 L1 损失函数
    print(f"  已实例化 L1 损失: {type(l1_loss_fn)}")

    mse_loss_fn = get_loss_function(loss_type="MSE") # 获取 MSE 损失函数
    print(f"  已实例化 MSE 损失: {type(mse_loss_fn)}")
    
    kl_loss_fn = get_loss_function(loss_type="KLDivergence") # 获取 KL 散度损失函数
    print(f"  已实例化 KLDiv 损失: {type(kl_loss_fn)}")
    
    # KLDiv 在训练步骤中如何使用的示例：
    # model_output_raw = dummy_model(torch.randn(3, 100)) # 原始 logits 输出
    # model_output_softmax = torch.softmax(model_output_raw, dim=-1) # Softmax 转换为概率
    # model_output_logsoftmax = torch.log(model_output_softmax + 1e-10) # 为稳定性添加 epsilon 后取对数
    # # 目标组分应总和为 1 (概率)
    # target_fractions = torch.softmax(torch.rand(3, 10), dim=-1) 
    # loss_kl_example = kl_loss_fn(model_output_logsoftmax, target_fractions) # 计算损失
    # print(f"  KLDiv 损失计算示例 (概念性): {loss_kl_example.item() if 'loss_kl_example' in locals() else '已跳过'}")


    # --- 2. 优化器示例 ---
    print("\n--- 优化器示例 ---")
    example_learning_rate = 5e-5 # 示例学习率
    # 获取 AdamW 优化器
    adamw_optimizer = get_optimizer(dummy_model, learning_rate=example_learning_rate, optimizer_type="AdamW", weight_decay=0.01)
    print(f"  已实例化 AdamW 优化器: {type(adamw_optimizer)}")
    
    # 获取 Adam 优化器
    adam_optimizer = get_optimizer(dummy_model, learning_rate=1e-3, optimizer_type="Adam", weight_decay=0.0)
    print(f"  已实例化 Adam 优化器: {type(adam_optimizer)}")


    # --- 3. 调度器示例 ---
    print("\n--- 调度器示例 ---")
    # 这些参数通常来自您的训练配置
    total_training_steps_example = 2000 # 示例：20 个 epoch * 100 个批次/epoch
    warmup_training_steps_example = 200  # 示例：总步数的 10% 用于预热
    
    # 将 AdamW 优化器用于调度器示例
    one_cycle_scheduler = get_scheduler(
        adamw_optimizer, 
        scheduler_type="OneCycleLR", 
        total_steps=total_training_steps_example,
        warmup_steps=warmup_training_steps_example 
    )
    print(f"  已实例化 OneCycleLR 调度器: {type(one_cycle_scheduler)}")

    linear_warmup_scheduler = get_scheduler(
        adam_optimizer, # 为多样性使用另一个优化器
        scheduler_type="LinearWarmup",
        total_steps=total_training_steps_example, # 对于这个简单的 LinearWarmup，在预热阶段后 total_steps 未严格使用
        warmup_steps=warmup_training_steps_example
    )
    print(f"  已实例化 LinearWarmup (LambdaLR) 调度器: {type(linear_warmup_scheduler)}")
    
    plateau_scheduler = get_scheduler(adamw_optimizer, scheduler_type="ReduceLROnPlateau")
    print(f"  已实例化 ReduceLROnPlateau 调度器: {type(plateau_scheduler)}")
    
    no_scheduler = get_scheduler(adamw_optimizer, scheduler_type=None) #不使用调度器
    print(f"  无调度器实例: {type(no_scheduler)}")

    print("\n--- 示例组件已成功实例化。 ---")
    print("这些工具函数现在可以集成到主微调脚本中。")
```
