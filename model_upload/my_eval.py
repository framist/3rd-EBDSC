import os
import subprocess
import time
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ebdsc3rd_datatools import (EBDSC3rdLoader, compute_CQ_score,
                                compute_MT_score, compute_SW_score)


def get_gpu_info():
    ans = ""
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        ans += output
    except subprocess.CalledProcessError:
        ans += "\nnvidia-smi not found"

    ans += f"\nCUDA 是否可用：{torch.cuda.is_available()}"
    ans += f"\n{torch.cuda.get_device_properties(0)}"

    # 获取当前设备
    ans += f"\n当前设备：{torch.cuda.current_device()}"

    # 获取设备名称
    ans += f"\n设备名称：{torch.cuda.get_device_name(0)}"

    # 获取可用的 GPU 数量
    ans += f"\n可用 GPU 数量：{torch.cuda.device_count()}"

    # 获取当前 GPU 的显存使用情况（以字节为单位）
    ans += f"\n当前显存分配：{torch.cuda.memory_allocated()/1024**3:.2f}GB"
    ans += f"\n当前显存缓存：{torch.cuda.memory_reserved()/1024**3:.2f}GB"
    ans += f"\n总显存：{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB"

    return ans


def timer(info):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            t = time.time()
            result = func(*args, **kwargs)
            log_string = f"{info} 完成，耗时 {time.time()-t:.2f}s [{func.__name__}]"
            print(log_string)
            # assert False, f"{log_string}\n{get_gpu_info()}\n{result=}"
            return result

        return wrapped_function

    return logging_decorator


def load_true_labels(test_dir):
    """按照 EBDSC3rdLoader 的逻辑加载真实标签"""
    test_dir = Path(test_dir)
    labels = []

    # 遍历所有 CSV 文件
    for csv_path in test_dir.glob("**/*.csv"):
        df = pd.read_csv(csv_path, names=EBDSC3rdLoader.DATA_COLUMNS)

        label = {
            "file_name": csv_path.name,
            "modulation_type": df["mod_type"].dropna().iloc[0],
            "symbol_width": df["symbol_width"].dropna().iloc[0],
            "code_sequence": " ".join(map(str, df["code_sequence"].dropna().astype(int))),
        }
        labels.append(label)

    return pd.DataFrame(labels)


def evaluate_result(pred_csv, true_labels_dir):
    """评估预测结果"""
    # 读取预测和真实标签
    pred_df = pd.read_csv(pred_csv)
    true_df = load_true_labels(true_labels_dir)

    # 确保文件名对齐
    pred_df = pred_df.set_index("file_name").sort_index()
    true_df = true_df.set_index("file_name").sort_index()
    assert len(pred_df) == len(true_df), "预测数量与真实标签数量不匹配"

    # 计算 MT_score
    mod_pred = torch.tensor(pred_df["modulation_type"].values) - 1
    mod_true = torch.tensor(true_df["modulation_type"].values) - 1
    mt_scores = compute_MT_score(F.one_hot(mod_pred, num_classes=11).float(), mod_true)

    # 计算 SW_score
    sw_pred = torch.tensor(pred_df["symbol_width"].values)
    sw_true = torch.tensor(true_df["symbol_width"].values)
    sw_scores = compute_SW_score(sw_pred, sw_true)

    # 计算 CQ_score
    def pad_sequence(seq, target_length=400):
        seq_list = list(map(int, seq.split()))
        if len(seq_list) < target_length:
            # 使用 0 填充
            seq_list.extend([0] * (target_length - len(seq_list)))
        return seq_list[:target_length]  # 如果超长则截断

    code_true = torch.tensor([pad_sequence(seq) for seq in true_df["code_sequence"].values])
    code_pred = torch.tensor([pad_sequence(seq) for seq in pred_df["code_sequence"].values])
    cq_scores_, cs_scores, acc_scores = compute_CQ_score(code_pred + 1, code_true + 1)
    
    cq_scores = []
    true_seqs = code_true.cpu().numpy()
    pred_seqs = code_pred.cpu().numpy()

    for i in range(code_true.shape[0]):
        true_vec = true_seqs[i].astype(float)
        pred_vec = pred_seqs[i].astype(float)
        norm_true = np.linalg.norm(true_vec)
        norm_pred = np.linalg.norm(pred_vec)
        if norm_true == 0 or norm_pred == 0:
            CS = 0.0
        else:
            CS = np.dot(true_vec, pred_vec) / (norm_true * norm_pred)

        # 转换为 CQ_score
        if CS < 0.7:
            CQ_score = 0.0
        elif CS > 0.95:
            CQ_score = 100.0
        else:
            CQ_score = ((CS - 0.7) / (0.95 - 0.7)) * 100
        CQ_score = np.clip(CQ_score, 0.0, 100.0)
        cq_scores.append(CQ_score)    
    cq_scores = torch.tensor(cq_scores)
    
    if abs(cq_scores_.mean().item() - cq_scores.mean().item()) > 1e-2:
        print(f"CQ_score 计算结果不一致:{cq_scores_.mean().item()} vs {cq_scores.mean().item()}")

    # 真值调制类别为未知选项或预测类别为未知，则选择预测正确则得分 100 分，如果预测错误得分 0 分
    mask = (mod_true == 10) | (mod_pred == 10)
    cs_scores[mask] = 0.0
    acc_scores[mask] = 0.0
    mask = (mod_true == 10) & (mod_pred == 10)
    cs_scores[mask] = 100.0
    acc_scores[mask] = 100.0
    # 计算平均分数
    mt_avg = mt_scores.mean().item()
    sw_avg = sw_scores.mean().item()
    cq_avg = cq_scores.mean().item()
    total_avg = 0.2 * mt_avg + 0.3 * sw_avg + 0.5 * cq_avg
    
    
    cs = cs_scores.mean().item()
    acc = acc_scores.mean().item()

    # 输出详细报告
    print("\n=== 本地测试评估报告 ===")
    print(f"MT Score (调制识别): {mt_avg:.6f}")
    print(f"SW Score (码元宽度): {sw_avg:.6f}")
    print(f"CQ Score (码序列) : {cq_avg:.6f}")
    print(f"Total Score     : {total_avg:.2f}")
    print(f"cs Score        : {cs:.2f}")
    print(f"acc             : {acc:.2f}")

    # 按调制类型分析
    print("\n--- 按调制类型分析 ---")
    for mod in range(11):
        mask = mod_true == mod

        mod_mt = mt_scores[mask].mean().item()
        mod_sw = sw_scores[mask].mean().item()
        mod_cq = cq_scores[mask].mean().item()
        mod_total = 0.2 * mod_mt + 0.3 * mod_sw + 0.5 * mod_cq
        
        cs = cs_scores[mask].mean().item()
        acc = acc_scores[mask].mean().item()

        print(f"{EBDSC3rdLoader.MOD_TYPE[mod]:7} MT {mod_mt:.2f} SW {mod_sw:.2f} CQ {mod_cq:.2f} Total {mod_total:.2f} cs {cs:.2f} acc {acc:.2f}")

    return {"mt_score": mt_avg, "sw_score": sw_avg, "cq_score": cq_avg, "total_score": total_avg}


if __name__ == "__main__":
    # 评估预测结果
    results = evaluate_result("result.csv", "./data/test")
