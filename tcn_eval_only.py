#!/usr/bin/env python3
"""
独立评估脚本 - 仅用于模型测试，不进行训练
用法：python tcn_eval_only.py --model_path ./saved_models/your_model.pth --test_data_path ./test_data/
"""

import datetime
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm

from my_tools import *
from ebdsc3rd_datatools import *

# 设置随机种子
seed_everything()

def create_parser():
    parser = argparse.ArgumentParser(
        description='3rd EBDSC 模型评估脚本 - 仅测试模式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型路径，例如 ./saved_models/TCN_best.pth')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='测试数据路径，例如 ./test_data/ 或 ../train_data/（用于验证）')
    
    # 模型参数（需要与训练时保持一致）
    parser.add_argument('--cuda', type=int, default=0, help='所使用的 cuda 设备')
    parser.add_argument('--num_layers', type=int, default=24, help='ModernTCN 层数')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--batch_size', type=int, default=100, help='批量大小')
    parser.add_argument('--ratio', type=int, default=2, help='FFN 比例')
    parser.add_argument('--ls', type=int, default=51, help='大核尺寸')
    parser.add_argument('--ss', type=int, default=5, help='小核尺寸')
    parser.add_argument('--dp', type=float, default=0.5, help='dropout')
    parser.add_argument('--emb_type', type=int, default=1, help='嵌入类型')
    parser.add_argument('--model', type=str, default='modernTCN', help='模型类型')
    parser.add_argument('--meanpool', action='store_true', default=False, help='使用均值池化')
    
    # 数据参数
    parser.add_argument('--max_code_len', type=int, default=400, help='最大码元长度')
    parser.add_argument('--mod_uniq_sym', action='store_true', default=False, help='使用调制独立符号')
    parser.add_argument('--demod_step', type=int, default=0, help='解调步骤')
    parser.add_argument('--demod_br', type=float, default=1, help='解调带宽比')
    parser.add_argument('--freq_topk', type=int, default=11, help='基频采样 topk')
    parser.add_argument('--form', type=str, default='algebraic', help='解调信号返回格式')
    
    # 评估参数
    parser.add_argument('--true_sym_width', action='store_true', default=False, 
                       help='评估时使用真实符号宽度')
    parser.add_argument('--true_mod_type', action='store_true', default=False,
                       help='评估时使用真实调制类型')
    parser.add_argument('--output_csv', type=str, default='eval_results.csv',
                       help='输出结果CSV文件名')
    
    return parser

def load_model(args, device, num_code_classes, num_mod_classes):
    """根据参数加载模型"""
    if args.model.startswith('modernTCN'):
        if 'FreTS' in args.model:
            from ModernTCN_FreTS import ModernTCN_MutiTask
        elif 'FTDW' in args.model:
            from ModernTCN_FTDW import ModernTCN_MutiTask
        else:
            from ModernTCN import ModernTCN_MutiTask
            
        model = ModernTCN_MutiTask(
            M=2,  # IQ 两个通道
            num_code_classes=num_code_classes,
            num_mod_classes=11,
            D=args.d_model,
            ffn_ratio=args.ratio,
            num_layers=args.num_layers,
            large_sizes=args.ls,
            small_size=args.ss,
            backbone_dropout=0.,
            head_dropout=args.dp,
            stem=args.emb_type,
            mean_pool=args.meanpool
        ).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    return model

def evaluate_model(model, data_loader, device, args, dataset):
    """评估模型性能"""
    model.eval()
    
    all_MT_scores = []
    all_SW_scores = []
    all_CQ_scores = []
    all_mod_labels = []
    all_mod_preds = []
    all_acc = []
    all_cs = []
    
    # 用于保存结果的列表
    results = []
    
    PAD_IDX = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="评估中")):
            IQ_data = batch["IQ_data"].to(device)
            code_sequence_aligned = batch["code_sequence_aligned"].to(device)
            code_mask = batch["code_mask"].to(device) 
            mod_type = batch["mod_type"].to(device)
            symbol_width = batch["symbol_width"].to(device)
            code_sequence = batch["code_sequence"].to(device)
            
            # 如果有文件名信息，保存用于输出
            file_names = batch.get("file_name", [f"batch_{batch_idx}_sample_{i}" for i in range(len(IQ_data))])
            
            # 模型推理
            mod_logits, symbol_width_pred, code_seq_logits = model(IQ_data)
            
            # 计算码序列预测
            if hasattr(dataset, 'sample_rate'):
                code_sed_pred = reverse_sequence_from_logits_batch(
                    symbol_width_absl=(symbol_width if args.true_sym_width else symbol_width_pred) * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT,
                    expanded_logits=code_seq_logits,
                    pad=PAD_IDX,
                    sample_rate=dataset.sample_rate,
                )
            else:
                # 简单的argmax解码
                code_sed_pred = torch.argmax(code_seq_logits, dim=-1)
            
            # 计算指标
            MT_scores = compute_MT_score(mod_logits, mod_type)
            SW_scores = compute_SW_score(symbol_width_pred, symbol_width)
            CQ_scores, cs, acc = compute_CQ_score(
                code_sed_pred,
                code_sequence,
                pad_idx=PAD_IDX,
                code_map_offset=getattr(dataset, 'code_map_offset', 1),
                uniq_symbol_args={
                    "enable": args.mod_uniq_sym,
                    "mod_preds": torch.argmax(mod_logits, dim=-1) if not args.true_mod_type else mod_type,
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
            
            # 保存详细结果用于输出CSV
            batch_size = len(IQ_data)
            for i in range(batch_size):
                # 将码序列转换为字符串格式
                pred_seq = code_sed_pred[i].cpu().numpy()
                # 移除padding
                pred_seq = pred_seq[pred_seq != PAD_IDX]
                pred_seq_str = ' '.join(map(str, pred_seq - getattr(dataset, 'code_map_offset', 1)))
                
                result = {
                    'file_name': file_names[i] if isinstance(file_names[i], str) else f"sample_{batch_idx}_{i}",
                    'modulation_type': int(mod_logits.argmax(dim=-1)[i].cpu().item() + 1),  # +1 because of 1-based indexing
                    'symbol_width': float(symbol_width_pred[i].cpu().item()),
                    'code_sequence': pred_seq_str
                }
                results.append(result)
    
    # 计算总体指标
    avg_MT_scores = torch.cat(all_MT_scores).mean().item()
    avg_SW_scores = torch.cat(all_SW_scores).mean().item()
    avg_CQ_scores = torch.cat(all_CQ_scores).mean().item()
    all_mod_labels = torch.cat(all_mod_labels)
    all_mod_preds = torch.cat(all_mod_preds)
    avg_acc = torch.cat(all_acc).mean().item()
    all_cs = torch.cat(all_cs).mean().item()
    
    # 计算加权总分
    avg_sample_score = 0.2 * avg_MT_scores + 0.3 * avg_SW_scores + 0.5 * avg_CQ_scores
    
    print(f"\n=== 评估结果 ===")
    print(f"总分 (Score): {avg_sample_score:.2f}")
    print(f"调制识别 (MT): {avg_MT_scores:.2f}")
    print(f"符号宽度 (SW): {avg_SW_scores:.2f}") 
    print(f"码序列 (CQ): {avg_CQ_scores:.2f}")
    print(f"准确率 (Acc): {avg_acc:.2f}")
    print(f"余弦相似度 (CS): {all_cs:.2f}")
    print(f"样本数量: {len(all_mod_labels)}")
    
    return results, {
        'total_score': avg_sample_score,
        'mt_score': avg_MT_scores,
        'sw_score': avg_SW_scores, 
        'cq_score': avg_CQ_scores,
        'accuracy': avg_acc,
        'cosine_similarity': all_cs
    }

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
        
    # 检查测试数据路径
    if not os.path.exists(args.test_data_path):
        print(f"错误: 测试数据路径不存在: {args.test_data_path}")
        sys.exit(1)
    
    print(f"加载模型: {args.model_path}")
    print(f"测试数据: {args.test_data_path}")
    
    # 创建数据集
    try:
        dataset = EBDSC3rdLoader(
            root_dir=args.test_data_path,
            demodulator=Demodulator(
                bandwidth_ratio=args.demod_br,
                step=args.demod_step,
                freq_topk=args.freq_topk,
                form=args.form
            ),
            code_map_offset=1,
            mod_uniq_symbol=args.mod_uniq_sym,
            data_aug=False,  # 评估时不使用数据增强
            is_test=True,  # 设置为测试模式
            sample_rate=1.0
        )
    except Exception as e:
        print(f"数据集加载失败: {e}")
        sys.exit(1)
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 评估时不打乱
        collate_fn=make_collate_fn(),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(data_loader)}")
    
    # 加载模型
    model = load_model(args, device, dataset.num_code_classes, dataset.num_mod_classes)
    
    # 加载权重
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型权重 (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("成功加载模型权重")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        sys.exit(1)
    
    # 评估模型
    results, metrics = evaluate_model(model, data_loader, device, args, dataset)
    
    # 保存结果到CSV
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\n结果已保存到: {args.output_csv}")
    
    return metrics

if __name__ == "__main__":
    metrics = main()