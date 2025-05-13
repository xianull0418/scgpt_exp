# scGPT

这是目前官方的最新版本，但是没有放出`pretrain`代码。

尝试写了一下预训练脚本，成功跑通。发现训练时占用显存显著降低，`batchsize`大小可以调到`128`，但是会有一个问题：出现系统级别的错误`OSError: [Errno 24] Too many open files`。
- 为保证机器性能，目前使用降低了`DataLoader`的参数，牺牲部分时间换取空间

## 新旧分支模型对比

| 特性/方面                   | `model_new.py` (新模型)                                     | `model.py` (旧模型)                                | 对显存影响 (新模型更优原因)                      |
| :---------------------- | :------------------------------------------------------- | :---------------------------------------------- | :----------------------------------- |
| **核心注意力**               | 主要用 FlashAttention (FlashMHA), 实现直接                      | 多种FlashAttention实现/封装, 与特定训练模式耦合                | 新：FlashAttention 应用更纯粹、高效，减少峰值显存。    |
| **Transformer Encoder** | 标准 `TransformerEncoder` + `FlashTransformerEncoderLayer` | 含定制的 `FlashscGPTGenerator` (生成式训练时)             | 新：Encoder 结构更标准统一，避免定制结构额外开销。        |
| **Forward 路径**          | 单一、直接的 `forward`                                         | `perceptual_forward` / `generative_forward` 双路径 | 新：移除高消耗的 `generative_forward`，简化逻辑。  |
| **生成式训练模块**             | **完全移除**                                                 | 含 `FlashscGPTLayer/Generator`, `flag_encoder` 等 | **核心显存优化点**：移除复杂模块，大幅降低参数和计算。        |
| **分类解码器**               | 仅 `ClsDecoder`                                           | 含 `ClsDecoder`，可选 `SimDecoder`                  | 新：简化解码器选项，减少潜在参数。                    |
| **对比细胞嵌入(CCE)**         | **新增** (双次编码)                                            | 无                                               | 新增计算，理论上增显存，但被其他优化项抵消。               |
| **模型参数/体积**             | 预计更少                                                     | 预计更多                                            | 新：模型更小，静态和动态显存占用通常更低。                |
| **代码复杂度**               | 更精简，配置选项少                                                | 更复杂，条件分支和配置标志多                                  | 新：复杂度降低，利于优化和减少潜在显存问题。               |
| **显存占用总结**              | **显著降低**                                                 | 相对较高                                            | **核心**：高效FlashAttention、移除复杂模块、模型简化。 |


## 部分训练参数
```bash
[2025-05-13 19:25:03] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:03] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:03] scGPT - INFO - Start training
[2025-05-13 19:25:03] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:03] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:03] scGPT - INFO - Start training
[2025-05-13 19:25:03] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:03] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:03] scGPT - INFO - Start training
[2025-05-13 19:25:03] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:03] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:03] scGPT - INFO - Start training
[2025-05-13 19:25:03] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:03] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:03] scGPT - INFO - Start training
[2025-05-13 19:25:03] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:03] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:04] scGPT - INFO - Start training
[2025-05-13 19:25:04] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:04] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:04] scGPT - INFO - Start training
[2025-05-13 19:25:04] scGPT - INFO - 启用TF32精度用于矩阵乘法和卷积
[2025-05-13 19:25:04] scGPT - INFO - 混合精度训练: 启用 - BF16
[2025-05-13 19:25:04] scGPT - INFO - Start training
[2025-05-13 20:08:12] scGPT - INFO - | epoch   1 | 2000/95273 batches | lr 0.0000 | ms/batch 1294.76 | loss 599.32 | mse 342.69 | mre 59201.37 |mvc 188.67 |
```


```bash
02:30:00-scGPT-INFO-train: | epoch   2 | 14800/49776 batches | lr 0.0001 | ms/batch 108.14 | loss 271.22 | mse 135.19 | mre 157370.04 |mvc 136.03 |
02:30:22-scGPT-INFO-train: | epoch   2 | 15000/49776 batches | lr 0.0001 | ms/batch 107.76 | loss 270.47 | mse 134.79 | mre 152854.22 |mvc 135.68 |
02:30:44-scGPT-INFO-train: | epoch   2 | 15200/49776 batches | lr 0.0001 | ms/batch 107.95 | loss 271.58 | mse 135.37 | mre 155652.75 |mvc 136.21 |
02:31:05-scGPT-INFO-train: | epoch   2 | 15400/49776 batches | lr 0.0001 | ms/batch 107.97 | loss 270.57 | mse 134.86 | mre 147089.51 |mvc 135.71 |
02:31:27-scGPT-INFO-train: | epoch   2 | 15600/49776 batches | lr 0.0001 | ms/batch 107.80 | loss 271.03 | mse 135.08 | mre 158644.20 |mvc 135.95 |
02:31:48-scGPT-INFO-train: | epoch   2 | 15800/49776 batches | lr 0.0001 | ms/batch 107.92 | loss 270.94 | mse 135.04 | mre 153891.37 |mvc 135.91 |
02:32:10-scGPT-INFO-train: | epoch   2 | 16000/49776 batches | lr 0.0001 | ms/batch 108.22 | loss 270.21 | mse 134.66 | mre 152466.52 |mvc 135.55 |
02:32:32-scGPT-INFO-train: | epoch   2 | 16200/49776 batches | lr 0.0001 | ms/batch 107.94 | loss 269.78 | mse 134.48 | mre 148173.27 |mvc 135.30 |
02:32:53-scGPT-INFO-train: | epoch   2 | 16400/49776 batches | lr 0.0001 | ms/batch 107.57 | loss 271.41 | mse 135.28 | mre 157856.77 |mvc 136.12 |
02:33:15-scGPT-INFO-train: | epoch   2 | 16600/49776 batches | lr 0.0001 | ms/batch 107.71 | loss 269.71 | mse 134.45 | mre 143636.02 |mvc 135.27 |
02:33:36-scGPT-INFO-train: | epoch   2 | 16800/49776 batches | lr 0.0001 | ms/batch 107.79 | loss 271.69 | mse 135.43 | mre 155154.79 |mvc 136.26 |
02:33:58-scGPT-INFO-train: | epoch   2 | 17000/49776 batches | lr 0.0001 | ms/batch 107.52 | loss 271.62 | mse 135.39 | mre 157489.90 |mvc 136.24 |
02:34:19-scGPT-INFO-train: | epoch   2 | 17200/49776 batches | lr 0.0001 | ms/batch 107.76 | loss 271.27 | mse 135.22 | mre 157896.20 |mvc 136.05 |
02:34:41-scGPT-INFO-train: | epoch   2 | 17400/49776 batches | lr 0.0001 | ms/batch 107.93 | loss 271.48 | mse 135.32 | mre 160743.12 |mvc 136.16 |
02:35:02-scGPT-INFO-train: | epoch   2 | 17600/49776 batches | lr 0.0001 | ms/batch 107.86 | loss 271.08 | mse 135.13 | mre 153112.24 |mvc 135.95 |
02:35:24-scGPT-INFO-train: | epoch   2 | 17800/49776 batches | lr 0.0001 | ms/batch 108.76 | loss 270.26 | mse 134.71 | mre 147365.52 |mvc 135.55 |
02:35:46-scGPT-INFO-train: | epoch   2 | 18000/49776 batches | lr 0.0001 | ms/batch 108.37 | loss 270.91 | mse 135.02 | mre 167502.28 |mvc 135.89 |
02:36:28-scGPT-INFO-eval_and_save: valid loss/mse 134.9183 | mre 150044.3281
02:36:28-scGPT-INFO-eval_and_save: Saving the best model to ./save/cellxgene_census_blood-May13-00-17-2025
02:36:54-scGPT-INFO-train: | epoch   2 | 18200/49776 batches | lr 0.0001 | ms/batch 340.95 | loss 271.39 | mse 135.28 | mre 151104.95 |mvc 136.10 |
```