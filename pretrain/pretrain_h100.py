import os
import sys
import argparse
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import scanpy as sc
import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from datasets import Dataset, load_dataset, concatenate_datasets

sys.path.insert(0, "../")
# 导入scGPT模块
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.tokenizer import GeneVocab, random_mask_value
from scgpt.scbank import DataBank
from scgpt.utils import MainProcessOnly, set_seed
from scgpt import logger

# 设置随机种子
set_seed(42)

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data-source",
    type=str,
    required=True,
    help="数据源路径或名称",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="保存模型和结果的目录",
)
parser.add_argument(
    "--load-model",
    type=str,
    default=None,
    help="加载预训练模型的路径",
)

# 数据相关参数
parser.add_argument(
    "--n-hvg",
    type=int,
    default=None,
    help="高变异基因的数量，0表示使用所有基因",
)
parser.add_argument(
    "--valid-size-or-ratio",
    type=float,
    default=0.1,
    help="验证集大小或比例",
)
parser.add_argument(
    "--grad-accu-steps",
    type=int,
    default=1,
    help="梯度累积步数",
)

# tokenizer相关参数
parser.add_argument(
    "--pad-token",
    type=str,
    default="<pad>",
    help="填充标记",
)
parser.add_argument(
    "--input-style",
    type=str,
    choices=["normed_raw", "log1p", "binned"],
    default="binned",
    help="输入数据的风格",
)
parser.add_argument(
    "--input-emb-style",
    type=str,
    choices=["category", "continuous", "scaling"],
    default="continuous",
    help="输入嵌入的风格",
)
parser.add_argument(
    "--n-bins",
    type=int,
    default=51,
    help="binned输入风格的bin数量",
)
parser.add_argument(
    "--max-seq-len",
    type=int,
    default=1536,
    help="序列的最大长度",
)
parser.add_argument(
    "--training-tasks",
    type=str,
    default="both",
    choices=["pcpt", "gen", "both"],
    help="训练任务类型",
)
parser.add_argument(
    "--mask-ratio",
    type=float,
    default=0.40,
    help="遮蔽值的比例",
)
parser.add_argument(
    "--trunc-by-sample",
    action="store_true",
    help="通过采样而不是截断来处理超长序列",
)
parser.add_argument(
    "--vocab-path",
    type=str,
    help="词汇表文件路径",
)

# 分布式训练参数
parser.add_argument(
    "--local-rank",
    type=int,
    default=-1,
    help="分布式训练的本地rank",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,  # 针对H100增加默认批量大小
    help="训练批量大小",
)
parser.add_argument(
    "--eval-batch-size",
    type=int,
    default=256,  # 针对H100增加默认批量大小
    help="评估批量大小",
)

# 训练参数
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="训练周期数",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="学习率",
)
parser.add_argument(
    "--scheduler-interval",
    type=int,
    default=100,
    help="学习率调整间隔",
)
parser.add_argument(
    "--scheduler-factor",
    type=float,
    default=0.99,
    help="学习率调整因子",
)
parser.add_argument(
    "--warmup-ratio-or-step",
    type=float,
    default=0.1,
    help="预热步数比例或固定步数",
)
parser.add_argument(
    "--no-cls",
    action="store_true",
    help="是否停用分类损失",
)
parser.add_argument(
    "--no-cce",
    action="store_true",
    help="是否停用对比细胞嵌入目标",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="是否使用FP16混合精度训练",
)
parser.add_argument(
    "--bf16",
    action="store_true",
    default=True,  # 针对H100默认启用BF16
    help="是否使用BF16混合精度训练（适用于H100）",
)
parser.add_argument(
    "--tf32",
    action="store_true",
    default=True,  # 针对H100默认启用TF32
    help="是否启用TF32精度（适用于H100）",
)
parser.add_argument(
    "--fast-transformer",
    action="store_true",
    default=True,  # 默认启用FlashAttention
    help="是否使用FlashAttention",
)

# 模型参数
parser.add_argument(
    "--nlayers",
    type=int,
    default=12,  # 增加默认层数以充分利用H100
    help="Transformer层数",
)
parser.add_argument(
    "--nheads",
    type=int,
    default=8,  # 增加默认头数
    help="注意力头数",
)
parser.add_argument(
    "--embsize",
    type=int,
    default=512,  # 增加嵌入维度以提高模型容量
    help="嵌入维度",
)
parser.add_argument(
    "--d-hid",
    type=int,
    default=1024,  # 增加前馈网络维度
    help="前馈网络维度",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout率",
)
parser.add_argument(
    "--n-layers-cls",
    type=int,
    default=3,
    help="分类网络层数",
)

# 日志参数
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="日志记录间隔",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=1000,
    help="模型保存间隔",
)

args = parser.parse_args()

# 从环境变量获取LOCAL_RANK
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print(f"从环境变量设置local_rank: {args.local_rank}")

# 验证设置
if args.input_style == "binned":
    if args.input_emb_style == "scaling":
        raise ValueError("binned输入不支持scaling嵌入风格")
elif args.input_style in ["log1p", "normed_raw"]:
    if args.input_emb_style == "category":
        raise ValueError("log1p或normed_raw输入不支持category嵌入风格")

if args.input_emb_style == "category":
    args.mask_value = args.n_bins + 1
    args.pad_value = args.n_bins  # 填充值
    n_input_bins = args.n_bins + 2
else:
    args.mask_value = -1
    args.pad_value = -2
    n_input_bins = args.n_bins

if args.training_tasks in ["gen", "both"]:
    args.mask_ratio = [0.25, 0.50, 0.75]

# 特殊标记和训练设置
special_tokens = [args.pad_token, "<cls>", "<eoc>"]
USE_CLS = not args.no_cls
USE_CCE = not args.no_cce
MVC = True
USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False

# 设置分布式训练
IS_DATA_PARALLEL = args.local_rank != -1
if IS_DATA_PARALLEL:
    torch.distributed.init_process_group(
        backend="nccl",
        rank=args.local_rank,
        timeout=timedelta(hours=10),
    )
    device = torch.device(f"cuda:{args.local_rank}")
    n_gpu = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    logger.info(
        f"设备: {device}, 世界大小: {world_size}, "
        f"可见GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}/{n_gpu}"
    )
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存目录并保存参数
save_dir = Path(args.save_dir)
if args.local_rank in [0, -1]:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
if IS_DATA_PARALLEL:
    torch.distributed.barrier()

# 设置日志
logger.info(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
writer = SummaryWriter(log_dir=save_dir / "tensorboard")
if IS_DATA_PARALLEL:
    writer = MainProcessOnly(writer)

# 数据处理函数
def _map_append_cls(dataset: Dataset) -> Dataset:
    logger.info(f"Rank {args.local_rank}: 向数据集添加<cls>标记")
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [args.pad_value] + example["expressions"],
        },
        num_proc=len(os.sched_getaffinity(0)),
    )
    return dataset

# 先从词汇表文件加载词汇表
if args.vocab_path and Path(args.vocab_path).exists():
    logger.info(f"从 {args.vocab_path} 加载词汇表")
    with open(args.vocab_path, "r") as f:
        vocab_dict = json.load(f)
    vocab = GeneVocab(list(vocab_dict.keys()))
    logger.info(f"成功加载词汇表，包含 {len(vocab)} 个标记")
else:
    logger.error(f"未找到词汇表文件: {args.vocab_path}")
    raise FileNotFoundError(f"词汇表文件不存在: {args.vocab_path}")

# 确保特殊标记存在于词汇表中
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
        logger.info(f"向词汇表添加了特殊标记: {s}")

# 加载数据
if Path(args.data_source).is_dir() and args.data_source.endswith(".scb"):
    # 大规模数据结构
    try:
        db = DataBank.from_path(args.data_source)
        raw_dataset = db.main_data.data
        # 使用已加载的词汇表而非从数据库加载
        logger.info(f"从数据库 {args.data_source} 加载了数据")
        
        if USE_CCE or USE_CLS or MVC:
            # 加载或创建带有<cls>前缀的数据集
            cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
            if not cls_prefix_datatable.exists():
                if args.local_rank in [0, -1]:
                    raw_dataset = _map_append_cls(raw_dataset)
                    raw_dataset.to_parquet(cls_prefix_datatable)
                if IS_DATA_PARALLEL:
                    torch.distributed.barrier()  # 等待映射完成
            raw_dataset = load_dataset(
                "parquet",
                data_files=str(cls_prefix_datatable),
                split="train",
                cache_dir=args.data_source,
            )
            logger.info(f"从{cls_prefix_datatable}加载了{len(raw_dataset)}个样本")
    except Exception as e:
        logger.error(f"加载数据库 {args.data_source} 时出错: {e}")
        raise
elif Path(args.data_source).is_file():
    try:
        adata = sc.read(args.data_source, cache=True)
        # 在首次加载数据时指定所需的列名
        (
            celltype_col,
            str_celltype_col,
            gene_col,
            batch_key,
        ) = scgpt.utils.find_required_colums(
            adata,
            id=args.data_source,
            configs_dir=Path(args.data_source).parent,
        )
        if celltype_col is None:
            celltype_col = "int" + str_celltype_col
            adata.obs[celltype_col] = scgpt.utils.category_str2int(
                adata.obs[str_celltype_col]
            )
        logger.info(f"从文件 {args.data_source} 加载了数据")
    except Exception as e:
        logger.error(f"加载数据文件 {args.data_source} 时出错: {e}")
        raise
else:
    logger.error(f"数据源 {args.data_source} 不存在或格式不支持")
    raise ValueError(f"数据源不存在或格式不支持: {args.data_source}")

# 加载预训练模型
if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    if len(vocab) != len(json.load(open(model_dir / "vocab.json"))):
        raise ValueError(
            f"要加载的模型目录({model_dir})中的词汇表与当前词汇表不匹配。"
        )
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"从{model_file}恢复模型，模型参数将被{model_config_file}中的配置覆盖。"
    )
    args.embsize = model_configs["embsize"]
    args.nheads = model_configs["nheads"]
    args.d_hid = model_configs["d_hid"]
    args.nlayers = model_configs["nlayers"]
    args.n_layers_cls = model_configs["n_layers_cls"]

    # 使用新值重新保存参数
    if args.local_rank in [0, -1]:
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

# 保存词汇表
if args.local_rank in [0, -1]:
    with open(save_dir / "vocab.json", "w") as f:
        json.dump(
            {token: index for token, index in vocab.get_stoi().items()},
            f,
            indent=2,
        )
if IS_DATA_PARALLEL:
    torch.distributed.barrier()  

# 数据处理
raw_dataset = raw_dataset.with_format("torch")

# 分割训练集和验证集
raw_dataset = raw_dataset.train_test_split(
    test_size=args.valid_size_or_ratio, shuffle=True
)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
logger.info(f"训练集样本数: {len(train_dataset)}")
logger.info(f"验证集样本数: {len(valid_dataset)}")

# 数据整理器
collator = scgpt.DataCollator(
    do_padding=True if args.max_seq_len is not None else False,
    pad_token_id=vocab[args.pad_token],
    pad_value=args.pad_value,
    do_mlm=True,
    do_binning=True if args.input_style == "binned" else False,
    mlm_probability=args.mask_ratio,
    mask_value=args.mask_value,
    max_length=args.max_seq_len,
    sampling=args.trunc_by_sample,
    data_style=args.training_tasks,
)

# 数据加载器
train_sampler = (
    DistributedSampler(train_dataset)
    if IS_DATA_PARALLEL
    else RandomSampler(train_dataset)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), args.batch_size),
    pin_memory=True,
    prefetch_factor=4,
)
valid_sampler = (
    DistributedSampler(valid_dataset, shuffle=False)
    if IS_DATA_PARALLEL
    else SequentialSampler(valid_dataset)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.eval_batch_size,
    sampler=valid_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), args.eval_batch_size),
    pin_memory=True,
)

# 创建模型
if USE_CLS:
    celltypes_labels = raw_dataset["celltypes"]
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)
else:
    num_types = 1

ntokens = len(vocab)  # 词汇表大小
model = TransformerModel(
    ntokens,
    d_model=args.embsize,
    nhead=args.nheads,
    d_hid=args.d_hid,
    nlayers=args.nlayers,
    nlayers_cls=args.n_layers_cls,
    n_cls=num_types if USE_CLS else 1,
    dropout=args.dropout,
    pad_token=args.pad_token,
    pad_value=args.pad_value,
    do_mvc=MVC,
    vocab=vocab,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=False,
    input_emb_style=args.input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style="cls",
    mvc_decoder_style="inner product",
    ecs_threshold=0.3,
    explicit_zero_prob=False,
    use_fast_transformer=args.fast_transformer,
    fast_transformer_backend="flash",
    pre_norm=True,  # 使用预归一化以提高训练稳定性
)

if args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
    except:
        # 处理模型键名不匹配的情况
        from collections import OrderedDict
        params = OrderedDict()
        for key, value in torch.load(model_file).items():
            params[key.replace("module.", "")] = value
        model.load_state_dict(params)

model.to(device)
logger.info(model)

if IS_DATA_PARALLEL:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=False,
    )

# 损失函数和优化器
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 学习率调度器
if args.warmup_ratio_or_step > 0:
    total_num_batches = len(train_loader) * args.epochs
    warmup_steps = (
        int(total_num_batches * args.warmup_ratio_or_step)
        if args.warmup_ratio_or_step < 1
        else int(args.warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_batches,
        last_epoch=-1,
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.scheduler_interval, gamma=args.scheduler_factor
    )

# 设置TF32精度
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("已启用TF32精度用于矩阵乘法和卷积")

# 设置混合精度训练
use_amp = args.fp16 or args.bf16
amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
if args.bf16 and args.fp16:
    logger.warning("同时指定了--fp16和--bf16，将优先使用bf16精度")
    
logger.info(f"混合精度训练: {'启用 - ' + ('BF16' if args.bf16 else 'FP16') if use_amp else '禁用'}")

# 初始化梯度缩放器
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # bf16不需要梯度缩放

# 训练函数
def train(model: nn.Module, train_loader: DataLoader, epoch: int) -> None:
    """训练一个epoch"""
    model.train()
    total_loss, total_mse, total_cls, total_gen, total_mvc = 0.0, 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        if USE_GENERATIVE_TRAINING:
            pcpt_gene = data_dict["pcpt_gene"]
            pcpt_expr = data_dict["pcpt_expr"]
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
            gen_gene = data_dict["gen_gene"]
            gen_expr_target = target_values = data_dict["gen_expr_target"]
            gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
        else:
            input_gene_ids = data_dict["gene"]
            input_values = data_dict["masked_expr"]
            target_values = data_dict["expr"]
            src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            if USE_GENERATIVE_TRAINING:
                output_dict = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    CLS=USE_CLS,
                    MVC=MVC,
                    ECS=False,
                )
                gen_expr_preds = output_values = output_dict["mlm_output"]

                positions_to_match = ~gen_key_padding_mask
                loss = loss_mse = criterion(
                    gen_expr_preds, gen_expr_target, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse.item(), global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"],
                        gen_expr_target,
                        positions_to_match,
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc.item(), global_iter)
            else:
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=USE_CLS,
                    CCE=USE_CCE,
                    MVC=MVC,
                    ECS=False,
                )
                output_values = output_dict["mlm_output"]

                positions_to_match = input_values.eq(args.mask_value)
                loss = loss_mse = criterion(
                    output_values, target_values, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse.item(), global_iter)
                if USE_CLS:
                    target_labels = data_dict["celltypes"]
                    loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
                    loss = loss + loss_cls
                    writer.add_scalar("train/cls", loss_cls.item(), global_iter)
                if USE_CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    writer.add_scalar("train/cce", loss_cce.item(), global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, positions_to_match
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc.item(), global_iter)
            
            writer.add_scalar("train/loss", loss.item(), global_iter)

        # 梯度累积
        if args.grad_accu_steps > 1:
            loss = loss / args.grad_accu_steps
            
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if args.grad_accu_steps > 1:
            if batch % args.grad_accu_steps == 0 or batch == num_batches - 1:
                scheduler.step()
                optimizer.zero_grad()
        else:
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            mre = masked_relative_error(
                output_values, target_values, positions_to_match
            )
            writer.add_scalar("train/mre", mre.item(), global_iter)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        # 确保变量存在后再使用
        if USE_CLS and 'loss_cls' in locals():
            total_cls += loss_cls.item()
        if MVC and 'loss_mvc' in locals():
            total_mvc += loss_mvc.item()
        total_error += mre.item()

        # 记录日志
        if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
            # 记录梯度分布
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, global_iter)
                    writer.add_histogram(name + "_param", param, global_iter)

            # 记录标量值
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if USE_CLS else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_error = total_error / log_interval
            
            logger.info(
                f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} 批次 | "
                f"lr {lr:9.7f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f" cls {cur_cls:5.2f} |" if USE_CLS else "")
                + (f" mvc {cur_mvc:5.2f} |" if MVC else "")
            )
            writer.add_scalar("lr", lr, global_iter)

            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_gen = 0
            total_mvc = 0
            total_error = 0
            start_time = time.time()

        # 定期评估和保存
        if batch % args.save_interval == 0 and batch > 0:
            eval_and_save(model, valid_loader, global_iter)
            model.train()  # 重要，恢复训练模式

# 评估函数
def evaluate(model: nn.Module, valid_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    
    with torch.no_grad():
        for data_dict in valid_loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            if USE_GENERATIVE_TRAINING:
                pcpt_gene = data_dict["pcpt_gene"]
                pcpt_expr = data_dict["pcpt_expr"]
                pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
                gen_gene = data_dict["gen_gene"]
                gen_expr_target = target_values = data_dict["gen_expr_target"]
                gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
            else:
                input_gene_ids = data_dict["gene"]
                input_values = data_dict["masked_expr"]
                target_values = data_dict["expr"]
                src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                if USE_GENERATIVE_TRAINING:
                    output_dict = model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        CLS=False,
                        MVC=False,
                        ECS=False,
                    )
                    gen_expr_preds = output_values = output_dict["mlm_output"]
                    positions_to_match = ~gen_key_padding_mask
                else:
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                    )
                    output_values = output_dict["mlm_output"]
                    positions_to_match = input_values.eq(args.mask_value)

                loss = criterion(output_values, target_values, positions_to_match)
                
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, positions_to_match
            ).item()
            
    total_loss = total_loss / len(valid_loader)
    total_error = total_error / len(valid_loader)
    
    return {
        "mse": torch.tensor(total_loss, device=device, dtype=torch.float),
        "mre": torch.tensor(total_error, device=device, dtype=torch.float),
    }

# 评估和保存函数
def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    iter_or_epoch: int,
    is_epoch: bool = False,
    save: bool = True,
) -> None:
    val_loss_dict = evaluate(model, valid_loader)
    val_loss, val_mre = val_loss_dict["mse"], val_loss_dict["mre"]
    
    if IS_DATA_PARALLEL:
        # 收集所有进程的结果
        val_loss_list = [torch.zeros_like(val_loss) for _ in range(world_size)]
        val_mre_list = [torch.zeros_like(val_mre) for _ in range(world_size)]
        torch.distributed.all_gather(val_loss_list, val_loss)
        torch.distributed.all_gather(val_mre_list, val_mre)
        val_loss = torch.mean(torch.stack(val_loss_list))
        val_mre = torch.mean(torch.stack(val_mre_list))
        
    val_loss, val_mre = val_loss.item(), val_mre.item()
    
    if args.local_rank in [0, -1]:
        if is_epoch:
            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| epoch结束 {iter_or_epoch:3d} | 时间: {elapsed:5.2f}s | "
                f"验证损失/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            )
            logger.info("-" * 89 + "\n")
            writer.add_scalar("valid/mse", val_loss, iter_or_epoch * len(valid_loader))
            writer.add_scalar("valid/mre", val_mre, iter_or_epoch * len(valid_loader))
        else:
            logger.info(f"验证损失/mse {val_loss:5.4f} | mre {val_mre:5.4f}")
            writer.add_scalar("valid/mse", val_loss, iter_or_epoch)
            writer.add_scalar("valid/mre", val_mre, iter_or_epoch)

        global best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存最佳模型
            logger.info(f"保存最佳模型到 {args.save_dir}")
            torch.save(
                model.module.state_dict() if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model.state_dict(),
                str(save_dir / "best_model.pt"),
            )

        if save:
            torch.save(
                model.module.state_dict() if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model.state_dict(),
                str(save_dir / f"model-{'ep' if is_epoch else ''}{iter_or_epoch}.pt"),
            )
            
    if IS_DATA_PARALLEL:
        torch.distributed.barrier()

# 训练主循环
best_val_loss = float("inf")
logger.info("开始训练")

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, train_loader, epoch=epoch)
    eval_and_save(model, valid_loader, iter_or_epoch=epoch, is_epoch=True)

writer.flush()
writer.close()

# 与简单基线相比
data_dict = next(iter(valid_loader))
input_values = data_dict["masked_expr"]
target_values = data_dict["expr"]
predict_ones = torch.ones(input_values.shape, dtype=torch.float32)
mse = masked_mse_loss(predict_ones, target_values, input_values.eq(args.mask_value))
mre = masked_relative_error(predict_ones, target_values, input_values.eq(args.mask_value))
logger.info(f"基线比较 - MSE: {mse.item()}, MRE: {mre.item()}")

# 模型分析部分
if args.local_rank in [0, -1]:
    logger.info("开始进行模型分析...")
    
    # 保存模型配置
    model_config = {
        "vocab_size": ntokens,
        "hidden_size": args.embsize,
        "num_hidden_layers": args.nlayers,
        "num_attention_heads": args.nheads,
        "intermediate_size": args.d_hid,
        "hidden_dropout_prob": args.dropout,
        "attention_probs_dropout_prob": args.dropout,
        "max_position_embeddings": args.max_seq_len,
        "type_vocab_size": 1,
        "initializer_range": 0.02,
    }
    
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # 保存训练配置摘要
    train_summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "warmup": args.warmup_ratio_or_step,
        "precision": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
        "best_val_loss": best_val_loss,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
    }
    
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(train_summary, f, indent=2)
        
    logger.info(f"训练完成。最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"模型已保存至: {save_dir}")

# 如果在分布式环境中，等待所有进程完成
if IS_DATA_PARALLEL:
    torch.distributed.barrier()
    
# 计算并输出训练速度统计信息
if args.local_rank in [0, -1]:
    logger.info("训练性能统计:")
    total_samples = len(train_dataset) * args.epochs
    total_batches = len(train_loader) * args.epochs
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"总批次数: {total_batches}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"梯度累积步数: {args.grad_accu_steps}")
    logger.info(f"有效批次大小: {args.batch_size * args.grad_accu_steps}")
    logger.info(f"混合精度: {'BF16' if args.bf16 else ('FP16' if args.fp16 else 'FP32')}")
    logger.info(f"TF32: {'已启用' if args.tf32 else '未启用'}")
    logger.info(f"Flash Attention: {'已启用' if args.fast_transformer else '未启用'}")
