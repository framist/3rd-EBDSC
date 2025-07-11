#!/usr/bin/env python3
"""
优化的训练脚本 - 包含预定义配置以解决训练速度问题

用法示例:
    # 使用快速配置 (A6000 GPU)
    python tcn_3rd_optimized.py --config a6000_fast
    
    # 使用平衡配置 (RTX4090 GPU)  
    python tcn_3rd_optimized.py --config rtx4090_balanced
    
    # 快速测试配置 (验证代码)
    python tcn_3rd_optimized.py --config quick_test
    
    # 查看所有可用配置
    python training_configs.py --list
    
    # 查看优化建议
    python training_configs.py --tips
"""

import datetime
import sys
import os

import numpy as np
import torch
import torch.utils.data as Data
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

now = datetime.datetime.now().strftime('%m%d_%H-%M')

from thop import clever_format, profile
from torch.utils.data import random_split

from my_tools import *

seed_everything()

import wandb
from ebdsc3rd_datatools import *

# 导入训练配置
from training_configs import TrainingConfigs, apply_config_to_parser, add_config_arguments, print_optimization_tips

NAME = '3rd'

import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='3rd EBDSC 优化训练脚本 -- by framist',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 添加配置选项
    parser = add_config_arguments(parser)
    
    # 基本参数
    parser.add_argument('--cuda', type=int, default=0, help='所使用的 cuda 设备，暂不支持多设备并行')
    parser.add_argument('--num_layers', type=int, default=24, help='layers of modernTCN')
    parser.add_argument('--d_model', type=int, default=128, help='d_model')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--ratio', type=int, default=2, help='ffn ratio')
    parser.add_argument('--ls', type=int, default=51, help='large kernel sizes')
    parser.add_argument('--ss', type=int, default=5, help='small kernel size')
    parser.add_argument('--dp', type=float, default=0.5, help='drop out')
    parser.add_argument('--emb_type', type=int, default=1, help='embedding type, 0: fixed, 1: learnable, 2: learnable + pos, 3: my')
    parser.add_argument('--max_epoch', type=int, default=64, help='max train epoch')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_step_size', type=int, default=16, help='lr step size')
    
    # 对照、消融实验的一些参数
    parser.add_argument('--model', type=str, default='modernTCN', help='backbone 模型选择')
    parser.add_argument('--manual', action='store_true', default=False, help='是否手动构建交织，需要 learnable_emb')

    # wandb
    parser.add_argument('--wandb', action='store_true', default=False, help='是否使用 wandb 记录')
    parser.add_argument('--name', type=str, default='tmp_exp', help='wandb 实验名称')
    parser.add_argument('--tags', nargs='+', default=['default'], help='wandb 标签，可输入多个')

    # 3rd
    parser.add_argument('--true_sym_width', action='store_true', default=False, help='是否使用真实的符号宽度 (在对齐输出中)')
    parser.add_argument('--true_mod_type', action='store_true', default=False, help='是否使用真实的调制类型 (在对齐 mod_uniq_sym 输出中)')
    parser.add_argument('--max_code_len', type=int, default=400, help='最大码元长度')
    parser.add_argument('--mutitask_weights', nargs='+', type=float, default=[0.2, 0.3, 0.5], help='多任务损失权重 for MT, SW, CQ')
    parser.add_argument('--mod_uniq_sym', action='store_true', default=False, help='是否使用 mod 独立的符号')
    parser.add_argument('--dont_data_aug', action='store_true', default=False, help='是否不使用数据增强')
    parser.add_argument('--meanpool', action='store_true', default=False, help='是否使用 meanpool 而非 attn pool 作为池化')
    parser.add_argument('--demod_step', type=int, default=0, help='Demodulator step')
    parser.add_argument('--demod_br', type=float, default=1, help='Demodulator band width rate min=0.5')
    parser.add_argument('--sample_rate', type=float, default=1, help='sample masking rate')
    parser.add_argument('--freq_topk', type=int, default=11, help='基频采样 topk')
    parser.add_argument('--form', type=str, default='algebraic', help='Return format for demodulated signal, in [algebraic, polar]')

    parser.add_argument('--best_continue', type=str, default=None, help='继续训练最佳模型，并重命名 e.g. --best_continue _c')
    
    # 新增优化选项
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--val_frequency', type=int, default=1, help='验证频率 (每N个epoch验证一次)')
    parser.add_argument('--save_frequency', type=int, default=8, help='混淆矩阵保存频率')
    
    return parser

def print_training_info(args, config_info=None):
    """打印训练信息和预估时间"""
    print("\n" + "="*60)
    print("3rd EBDSC 训练配置信息")
    print("="*60)
    
    if config_info:
        print(f"使用配置: {config_info['description']}")
    
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"模型层数: {args.num_layers}")
    print(f"模型维度: {args.d_model}")
    print(f"批量大小: {args.batch_size}")
    print(f"最大训练轮数: {args.max_epoch}")
    print(f"学习率: {args.lr}")
    
    # 简单的时间预估
    complexity_factor = (args.num_layers / 24) * (args.d_model / 128) * (args.max_epoch / 64)
    
    if complexity_factor < 0.3:
        time_estimate = "2-6 小时"
    elif complexity_factor < 0.6:
        time_estimate = "6-15 小时"
    elif complexity_factor < 1.0:
        time_estimate = "15-30 小时"
    else:
        time_estimate = "30+ 小时 (建议使用更轻量的配置)"
    
    print(f"预估训练时间: {time_estimate}")
    print("="*60 + "\n")

def main():
    parser = create_parser()
    parser_args = parser.parse_args()
    
    # 处理特殊命令
    if parser_args.list_configs:
        from training_configs import print_all_configs
        print_all_configs()
        return
    
    # 应用预定义配置
    config_info = None
    if parser_args.config:
        try:
            config_info = apply_config_to_parser(parser, parser_args.config)
            # 重新解析参数以获取更新后的默认值
            parser_args = parser.parse_args()
            print(f"\n✓ 已应用配置: {parser_args.config.upper()}")
        except ValueError as e:
            print(f"错误: {e}")
            return
    
    # 设备设置
    use_cuda = True
    device = torch.device(f"cuda:{parser_args.cuda}" if (
        use_cuda and torch.cuda.is_available()) else "cpu")
    print("CUDA Available: ", torch.cuda.is_available(), 'use:', device)
    
    # 打印训练信息
    print_training_info(parser_args, config_info)
    
    # 检查是否需要显示优化建议
    if parser_args.num_layers >= 20 and parser_args.max_epoch >= 48 and not parser_args.config:
        print("⚠️  检测到使用了较大的模型配置，训练可能很慢。")
        print("💡 建议使用预定义配置: python tcn_3rd_optimized.py --config a6000_fast")
        print("📖 查看所有配置: python training_configs.py --list")
        print("🚀 查看优化建议: python training_configs.py --tips\n")
    
    # 继续原有的训练逻辑...
    BATCH_SIZE = parser_args.batch_size
    INPUT_CHANNELS = 2

    POS_D = D = parser_args.d_model
    P = 1
    S = 1
    R = parser_args.ratio
    NUM_LAYERS = parser_args.num_layers
    DROP_OUT = parser_args.dp

    MAX_TRAIN_EPOCH = parser_args.max_epoch

    IF_LAERNABLE_EMB = parser_args.emb_type == 1
    NAME += "_MeanPool" if parser_args.meanpool else "_AttnPool"

    NAME += f"_{parser_args.name}"

    root_dir = "../train_data/"  # 替换为实际路径
    CODE_MAP_OFFSET = 1  # 码元映射偏移

    # 创建 train_loader
    full_dataset = EBDSC3rdLoader(
        root_dir=root_dir,
        demodulator=Demodulator(
            bandwidth_ratio=parser_args.demod_br,
            step=parser_args.demod_step,
            freq_topk=parser_args.freq_topk,
            form=parser_args.form
        ),
        code_map_offset=CODE_MAP_OFFSET,
        mod_uniq_symbol=parser_args.mod_uniq_sym,
        data_aug=not parser_args.dont_data_aug,
        is_test=False,
        sample_rate=parser_args.sample_rate
    )
    if parser_args.mod_uniq_sym:
        NAME += "_mod_uniq_sym"

    NUM_CODE_CLASSES = full_dataset.num_code_classes
    NUM_MOD_CLASSES = full_dataset.num_mod_classes
    PAD_IDX = 0  # 填充符号 ID
    MAX_CODE_LENGTH = parser_args.max_code_len
    MUTITASK_WEIGHTS = parser_args.mutitask_weights

    learn_rate = parser_args.lr
    lr_step_size = parser_args.lr_step_size

    if MUTITASK_WEIGHTS[1] == 0:
        assert parser_args.true_sym_width, "SW 为 0 时，必须使用真实符号宽度" 

    is_debug = True if sys.gettrace() else False
    if is_debug:
        print("\n!!!!!!!!!!!!!!! Debugging !!!!!!!!!!!!!!!\n")
        assert parser_args.wandb == False, "Debugging 时不支持 wandb"

    # %% 模型、优化器选择
    if parser_args.model.startswith('modernTCN'):
        if 'FreTS' in parser_args.model:
            from ModernTCN_FreTS import ModernTCN_MutiTask
            NAME = f'freTSTCN_{parser_args.ls}KS{parser_args.ss}_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        elif 'FTDW' in parser_args.model:
            from ModernTCN_FTDW import ModernTCN_MutiTask
            NAME = f'FTDWTCN_{parser_args.ls}KS{parser_args.ss}_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        else:
            from ModernTCN import ModernTCN_MutiTask
            NAME = f'TCN_{parser_args.ls}KS{parser_args.ss}_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'

        model = ModernTCN_MutiTask(
                M=INPUT_CHANNELS, 
                num_code_classes=NUM_CODE_CLASSES, 
                num_mod_classes=11,
                D=D,
                ffn_ratio=R, 
                num_layers=NUM_LAYERS, 
                large_sizes=parser_args.ls,
                small_size=parser_args.ss,
                backbone_dropout=0.,
                head_dropout=DROP_OUT,
                stem = parser_args.emb_type,
                mean_pool=parser_args.meanpool
            ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    elif parser_args.model == 'Transformer':
        from models.Transformer import Configs, Model
        configs = Configs()
            
        D = D * 2
        NAME = f'TF_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        configs.d_model = D
        configs.e_layers = NUM_LAYERS
        configs.d_ff = D * R
        configs.n_heads = 2
        configs.dropout = DROP_OUT
        configs.num_code_classes = NUM_CODE_CLASSES
        configs.num_mod_classes = NUM_MOD_CLASSES
        configs.mean_pool = parser_args.meanpool
            
        model = Model(configs=configs, wide_value_emb=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
        lr_lambda = lambda step: (D ** -0.5) * min((step+1) ** -0.5, (step+1) * 16 ** -1.5)

    elif parser_args.model == 'iTransformer':
        assert IF_LAERNABLE_EMB == True, "iTransformer 模型必须使用可学习的 emb. TODO"
        D = 128 * 2
        NAME = f'iTransformer_{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        from models.iTransformer import Configs, Model
        configs = Configs()
        configs.e_layers = NUM_LAYERS
        configs.d_ff = configs.d_model * R
        configs.dropout = DROP_OUT

        model = Model(configs=configs, wide_value_emb=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
        lr_lambda = lambda step: (D ** -0.5) * min((step+1) ** -0.5, (step+1) * 16 ** -1.5) 

    elif parser_args.model == 'TimesNet':
        assert IF_LAERNABLE_EMB == True, "TimesNet 模型必须使用可学习的 emb. TODO"

        NAME = f'TimesNet_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        from models.TimesNet import Configs, Model
        configs = Configs()
        configs.d_model = D
        configs.e_layers = NUM_LAYERS
        configs.d_ff = D * R
        configs.dropout = DROP_OUT
        configs.num_code_classes = NUM_CODE_CLASSES
        configs.num_mod_classes = NUM_MOD_CLASSES

        model = Model(configs=configs, wide_value_emb=False).to(device)
        optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)

    else:
        raise ValueError('model 选择错误')

    # 定义 MultiTaskLoss
    class MultiTaskLoss(nn.Module):
        def __init__(self, mod_weight=0.2, width_weight=0.3, seq_weight=0.5, pad_idx=0, label_smoothing=0.):
            super().__init__()
            # 归一化权重
            total_weight = mod_weight + width_weight + seq_weight
            self.mod_weight = mod_weight / total_weight
            self.width_weight = width_weight / total_weight
            self.seq_weight = seq_weight / total_weight
            self.pad_idx = pad_idx
            self.label_smoothing = label_smoothing

        def forward(
            self,
            mod_logits,
            symbol_width_pred,
            code_seq_logits,
            mod_labels,
            symbol_width_labels,
            code_seq_labels,
            code_seq_masks,
        ):
            # 调制类型损失
            mod_loss = F.cross_entropy(mod_logits, mod_labels) / np.log(NUM_MOD_CLASSES)

            # 码元宽度损失
            width_loss = F.mse_loss(symbol_width_pred, symbol_width_labels) / 0.04

            # 码序列损失
            batch_size, tgt_seq_len, num_classes = code_seq_logits.size()

            # 仅计算非填充部分
            code_seq_labels = code_seq_labels.reshape(-1)
            code_seq_logits = code_seq_logits.reshape(-1, num_classes)

            active_logits = code_seq_logits[code_seq_labels != self.pad_idx]
            active_labels = code_seq_labels[code_seq_labels != self.pad_idx]
            
            seq_loss = F.cross_entropy(active_logits, active_labels, label_smoothing=self.label_smoothing) / np.log(NUM_CODE_CLASSES)

            return self.mod_weight * mod_loss + self.width_weight * width_loss + self.seq_weight * seq_loss

    if parser_args.wandb:
        wandb.init(
            project="TCN 3rd freq",
            name=parser_args.name + (parser_args.best_continue if parser_args.best_continue is not None else ''),
            config=parser_args,
            tags=parser_args.tags,
            notes=NAME,
            save_code=True,
        )

    if 0. in MUTITASK_WEIGHTS:
        print(f'Warning: 0 in {MUTITASK_WEIGHTS=}')
    criterion = MultiTaskLoss(*MUTITASK_WEIGHTS, pad_idx=PAD_IDX, label_smoothing=parser_args.label_smoothing)

    # 定义训练集和验证集的比例
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 随机划分数据集
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    # 创建训练集 DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(),
        num_workers=parser_args.num_workers,
        pin_memory=bool(torch.cuda.is_available()),
    )

    # 创建验证集 DataLoader
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=make_collate_fn(),
        num_workers=parser_args.num_workers,
        pin_memory=bool(torch.cuda.is_available()),
    )

    print(f"{NAME=}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if parser_args.best_continue is not None:
        epoch_start = load_checkpoint(model, f"./saved_models/{NAME}_best.pth", optimizer, device)
        print(f"Continue from {NAME} at {epoch_start=}")
        NAME += parser_args.best_continue
        print(f"Renamed to {NAME}")
    else:
        epoch_start = -1

    if parser_args.model in ["modernTCN", "TimesNet", "modernTCN_FreTS", "modernTCN_FTDW"]:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5, last_epoch=epoch_start)
    elif parser_args.model in ["Transformer", "iTransformer"]:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=epoch_start)
    else:
        raise ValueError("model 选择错误")
    epoch_start += 1

    scaler = GradScaler()
    torch.cuda.empty_cache()
    model.to(device)
    model.train()
    best_score = 0.0
    t = tqdm(range(epoch_start, epoch_start + MAX_TRAIN_EPOCH), dynamic_ncols=True)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in t:
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            IQ_data = batch["IQ_data"].to(device)
            code_sequence_aligned = batch["code_sequence_aligned"].to(device)
            code_mask = batch["code_mask"].to(device)
            mod_type = batch["mod_type"].to(device)
            symbol_width = batch["symbol_width"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                mod_logits, symbol_width_pred, code_seq_logits = model(IQ_data)

                # 计算损失
                loss = criterion(
                    mod_logits=mod_logits,
                    symbol_width_pred=symbol_width_pred,
                    code_seq_logits=code_seq_logits,
                    mod_labels=mod_type,
                    symbol_width_labels=symbol_width,
                    code_seq_labels=code_sequence_aligned,
                    code_seq_masks=code_mask,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            t.set_description(f"{NAME} Loss={loss.item():.2f}")
            if is_debug:
                break
            
        lr_scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段（根据频率决定是否执行）
        if epoch % parser_args.val_frequency == 0:
            model.eval()
            total_val_loss = 0
            all_MT_scores = []
            all_SW_scores = []
            all_CQ_scores = []
            all_mod_labels = []
            all_mod_preds = []
            all_acc = []
            all_cs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    IQ_data = batch["IQ_data"].to(device)
                    code_sequence_aligned = batch["code_sequence_aligned"].to(device)
                    code_mask = batch["code_mask"].to(device)
                    mod_type = batch["mod_type"].to(device)
                    symbol_width = batch["symbol_width"].to(device)
                    code_sequence = batch["code_sequence"].to(device)

                    mod_logits, symbol_width_pred, code_seq_logits = model(IQ_data,)

                    # 计算损失
                    loss = criterion(
                        mod_logits=mod_logits,
                        symbol_width_pred=symbol_width_pred,
                        code_seq_logits=code_seq_logits,
                        mod_labels=mod_type,
                        symbol_width_labels=symbol_width,
                        code_seq_labels=code_sequence_aligned,
                        code_seq_masks=code_mask,
                    )
                    total_val_loss += loss.item()

                    if MUTITASK_WEIGHTS[2] > 0.0:
                        code_sed_pred = reverse_sequence_from_logits_batch(
                            symbol_width_absl=(symbol_width if parser_args.true_sym_width else symbol_width_pred) * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT,
                            expanded_logits=code_seq_logits,
                            pad=PAD_IDX,
                            sample_rate=full_dataset.sample_rate,
                        )
                    else:
                        # 不评分
                        code_sed_pred = torch.zeros_like(code_sequence)

                    # 计算指标
                    MT_scores = compute_MT_score(mod_logits, mod_type)
                    SW_scores = compute_SW_score(symbol_width_pred, symbol_width)
                    CQ_scores, cs, acc = compute_CQ_score(
                        code_sed_pred,
                        code_sequence,
                        pad_idx=PAD_IDX,
                        code_map_offset=full_dataset.code_map_offset,
                        uniq_symbol_args={
                            "enable": full_dataset.mod_uniq_symbol,
                            "mod_preds": torch.argmax(mod_logits, dim=-1) if not parser_args.true_mod_type else mod_type,
                            "mod_labels": mod_type,
                        }
                    )

                    all_MT_scores.append(MT_scores)
                    all_SW_scores.append(SW_scores)
                    all_CQ_scores.append(CQ_scores)
                    all_mod_labels.append(mod_type)
                    all_mod_preds.append(mod_logits.argmax(dim=-1))
                    all_acc.append(acc)
                    all_cs.append(cs)

            avg_val_loss = total_val_loss / len(val_loader)

            # 聚合指标
            avg_MT_scores = torch.cat(all_MT_scores).mean().item()
            avg_SW_scores = torch.cat(all_SW_scores).mean().item()
            avg_CQ_scores = torch.cat(all_CQ_scores).mean().item()
            all_mod_labels = torch.cat(all_mod_labels)
            all_mod_preds = torch.cat(all_mod_preds)
            avg_acc = torch.cat(all_acc).mean().item()
            all_cs = torch.cat(all_cs).mean().item()

            # 计算加权总分
            avg_sample_score = 0.2 * avg_MT_scores + 0.3 * avg_SW_scores + 0.5 * avg_CQ_scores

            tqdm.write(
                f"[{epoch+1}/{MAX_TRAIN_EPOCH}] lr{lr_scheduler.get_last_lr()[0]:.2e} Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Score: {avg_sample_score:.2f}, "
                f"MT: {avg_MT_scores:.2f}, SW: {avg_SW_scores:.2f}, CQ: {avg_CQ_scores:.2f}, acc: {avg_acc:.2f}, cs: {all_cs:.2f}"
            )

            if avg_sample_score > best_score:
                best_score = avg_sample_score
                save_checkpoint(epoch, model, optimizer, f"./saved_models/{NAME}_best.pth")
                tqdm.write(f"Saved {NAME}_best.pth with best score {best_score:.2f}")

            log = {
                "Train Loss": avg_train_loss,
                "Val Loss": avg_val_loss,
                "Score": avg_sample_score,
            }

            if MUTITASK_WEIGHTS[0] > 0.0:
                log["MT"] = avg_MT_scores
            if MUTITASK_WEIGHTS[1] > 0.0:
                log["SW"] = avg_SW_scores
            if MUTITASK_WEIGHTS[2] > 0.0:
                log["CQ"] = avg_CQ_scores
                log["cs"] = all_cs
                log["Acc"] = avg_acc

            if epoch % parser_args.save_frequency == 0:
                # 绘制类别识别混淆矩阵
                cm = np.zeros((NUM_MOD_CLASSES, NUM_MOD_CLASSES))
                for target, prediction in zip(all_mod_labels, all_mod_preds):
                    cm[target, prediction] += 1
                fig, ax = plt.subplots()
                ax.matshow(cm, cmap="viridis")
                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center")

                plt.title(f"Modulation Type Confusion Matrix at E{epoch}")
                plt.xlabel("Prediction")
                plt.ylabel("Target")

                plt.savefig(f"./saved_figs/{NAME}_confusion_matrix.png")
                log["类别识别混淆矩阵"] = fig

            if parser_args.wandb:
                wandb.log(log, step=epoch)
        else:
            # 只显示训练损失
            tqdm.write(f"[{epoch+1}/{MAX_TRAIN_EPOCH}] lr{lr_scheduler.get_last_lr()[0]:.2e} Train Loss: {avg_train_loss:.4f}")

    print(f"{model=}")

    # 打印一个批次的数据形状
    for batch in val_loader:
        print("IQ_data.shape:", batch["IQ_data"].shape)

        flops, params = profile(model, inputs=(batch["IQ_data"].to(device),))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs: {flops}, Params: {params}")
        break

    if parser_args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()