
# autocorrect: false
"""
= 1 step
= FreTS TCN
= 开集识别 传统 SoftMax
    + THRESHOLD = 0.7  # 开集识别阈值
= SAMPLE_RATE = 1  # 采样率
= 原生复数 hidden FFN
= 投票
    + VOTE_TIMES = 11
    + 修改为概率融合方式
- best score 在验证集相比 FreTS_XXXL_best 低 ( SW: 97.76 < SW: 98.24)

[48/64] lr1.25e-04 Train Loss: 0.3165, Val Loss: 0.3262, Score: 66.71, MT: 79.17, SW: 97.98, CQ: 42.97, acc: 0.27, cs: 0.73
40 Saved freTSTCN_51KS5_128D20L2R5dp_3rd_AttnPool_sr7_XXXL_FreTS原复2c_随移1c_best.pth with best score 66.71
FLOPs: 1.657T, Params: 3.290M

vote_times = 1 THRESHOLD = 0.7
=== 本地测试评估报告 ===
MT Score (调制识别): 73.699997
SW Score (码元宽度): 98.074499
CQ Score (码序列) : 35.443849
Total Score     : 61.88
cs Score        : 0.51
acc             : 0.55

vote_times = 3 THRESHOLD = 0.7
=== 本地测试评估报告 ===
MT Score (调制识别): 74.800003
SW Score (码元宽度): 98.218944
CQ Score (码序列) : 36.193097
Total Score     : 62.52
cs Score        : 0.52
acc             : 0.56

vote_times = 5 THRESHOLD = 0.7

=== 本地测试评估报告 ===
MT Score (调制识别): 74.900002
SW Score (码元宽度): 98.247621
CQ Score (码序列) : 36.237539
Total Score     : 62.57
cs Score        : 0.52
acc             : 0.56

vote_times = 5 THRESHOLD = 0.8

=== 本地测试评估报告 ===
MT Score (调制识别): 72.900002
SW Score (码元宽度): 98.247621
CQ Score (码序列) : 36.237539
Total Score     : 62.17
cs Score        : 0.51
acc             : 0.54

概率融合 vote_times = 5 THRESHOLD = 0.8

=== 本地测试评估报告 ===
MT Score (调制识别): 72.500000
SW Score (码元宽度): 98.163374
CQ Score (码序列) : 36.315554
Total Score     : 62.11
cs Score        : 0.50
acc             : 0.54

概率融合 vote_times = 11 THRESHOLD = 0.7

main 完成，耗时 339.50s [main]

=== 本地测试评估报告 ===
MT Score (调制识别): 74.500000
SW Score (码元宽度): 98.118156
CQ Score (码序列) : 36.517194
Total Score     : 62.59
cs Score        : 0.52
acc             : 0.56

--- 按调制类型分析 ---
BPSK    MT 100.00 SW 99.76 CQ 0.00 Total 49.93 cs 0.02 acc 0.84
QPSK    MT 88.00 SW 98.27 CQ 27.52 Total 60.84 cs 0.69 acc 0.69
8PSK    MT 80.00 SW 98.46 CQ 35.79 Total 63.43 cs 0.65 acc 0.56
MSK     MT 100.00 SW 97.47 CQ 26.35 Total 62.42 cs 0.67 acc 0.91
8QAM    MT 92.00 SW 99.05 CQ 19.89 Total 58.06 cs 0.68 acc 0.66
16QAM   MT 53.00 SW 96.96 CQ 34.55 Total 56.96 cs 0.43 acc 0.32
32QAM   MT 48.00 SW 97.27 CQ 30.85 Total 54.21 cs 0.39 acc 0.32
8APSK   MT 62.00 SW 97.73 CQ 65.00 Total 74.22 cs 0.58 acc 0.45
16APSK  MT 64.00 SW 96.89 CQ 67.18 Total 75.46 cs 0.58 acc 0.42
32APSK  MT 58.00 SW 99.33 CQ 58.04 Total 70.42 cs 0.52 acc 0.39
UNKNOWN MT nan SW nan CQ nan Total nan cs nan acc nan

---old---

tcn_3rd.py --model modernTCN_FreTS --num_layers 24 --ratio 2 --batch_size 32 --tags freTS step1 fnc --name sr7_step1_FreTS_XXXL --true_sym_width --demod_step 1 --sample_rate 0.7 --wandb
[63/64] lr1.25e-04 Train Loss: 0.3092, Val Loss: 0.3317, Score: 66.74, MT: 78.39, SW: 98.24, CQ: 43.18, acc: 0.27, cs: 0.80
Saved freTSTCN_51KS5_128D24L2R5dp_3rd_AttnPool_sr7_step1_FreTS_XXXL_best.pth with best score 66.74

local test:

=== 本地测试评估报告 ===
MT Score (调制识别): 75.80
SW Score (码元宽度): 98.36
CQ Score (码序列) : 29.10
Total Score     : 59.22
cs Score        : 0.56
acc             : 0.54


--- 按调制类型分析 ---
BPSK    MT 99.00 SW 99.91 CQ 0.97 Total 50.26 cs 0.55 acc 0.74
QPSK    MT 87.00 SW 98.75 CQ 7.04 Total 50.55 cs 0.55 acc 0.59
8PSK    MT 91.00 SW 98.63 CQ 29.61 Total 62.60 cs 0.68 acc 0.55
MSK     MT 100.00 SW 98.60 CQ 37.19 Total 68.17 cs 0.74 acc 0.92
8QAM    MT 85.00 SW 98.14 CQ 30.31 Total 61.60 cs 0.66 acc 0.61
16QAM   MT 53.00 SW 97.16 CQ 12.17 Total 45.83 cs 0.39 acc 0.34
32QAM   MT 50.00 SW 97.80 CQ 19.00 Total 48.84 cs 0.38 acc 0.34
8APSK   MT 68.00 SW 97.13 CQ 50.02 Total 67.75 cs 0.60 acc 0.48
16APSK  MT 67.00 SW 98.11 CQ 54.00 Total 69.83 cs 0.59 acc 0.44
32APSK  MT 58.00 SW 99.36 CQ 50.68 Total 66.75 cs 0.52 acc 0.40
UNKNOWN MT nan SW nan CQ nan Total nan cs nan acc nan



TODO：SW 离散化提交，不过本地测试无大影响（甚至更差一点）
TODO：sample rate 测试集修正，不过本地测试无大影响

TODO: sample rate = 1.0 测试
TODO：THRESHOLD > 0.7 测试
---

NOTE 
- 注意目标平台似乎不会爆显存而是使用共享内存，大大延缓计算速度！
- 本地测试中 CQ 比预期小 (这个可能受 true sym width 影响，但具体待量化实验)

---

这是一个提交示例 Python 脚本，供选手参考。
-测评环境为 Python3.8
-测评环境中提供了基础的包和框架，具体版本请查看【https://github.com/Datacastle-Algorithm-Department/images/blob/main/doc/py38.md】
-如有测评环境未安装的包，请在 requirements.txt 里列明，最好列明版本，例如：numpy==1.23.5
-如不需要安装任何包，请保持 requirements.txt 文件为空即可，但是提交时一定要有此文件
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ebdsc3rd_datatools import *
from ModernTCN_FreTS import *
from my_tools import *
from my_eval import *

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

BATCH_SIZE = 128
CODE_MAP_OFFSET = 1  # 码元映射偏移
PAD_IDX = 0  # 填充符号 ID
THRESHOLD = 0.7  # 开集识别阈值
SAMPLE_RATE = 1  # 采样率
VOTE_TIMES = 11

seed_everything()


def get_symbol_width(device, loader, model):
    symbol_width_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            IQ_data = batch["IQ_data"].to(device)
            filenames = batch["filename"]

            _, symbol_width_pred, _ = model(
                IQ_data,
            )

            # for i in range(len(filenames)):
            #     symbol_width_list.append(
            #         symbol_width_pred[i].item(),
            #     )
            symbol_width_pred = symbol_width_pred.clamp(0.25, 1)
            symbol_width_list.extend(symbol_width_pred.tolist())

    return np.array(symbol_width_list)


# mod 开集识别函数
def mod_open_set(logits, threshold=THRESHOLD):
    """
    开集识别函数，传统 SoftMax + 阈值方法
    """
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = torch.max(probs, dim=1)

    # 应用阈值判断未知类（ID=11）
    unknown_mask = max_probs < threshold
    preds[unknown_mask] = 10
    return preds



def get_result(device, loader, model, symbol_width_list=None):
    result_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            IQ_data = batch["IQ_data"].to(device)
            filenames = batch["filename"]
            idx = batch["idx"]

            mod_logits, symbol_width_pred, code_seq_logits = model(
                IQ_data,
            )

            # - mod 开集识别
            mod_preds = mod_open_set(mod_logits)

            if symbol_width_list is not None:
                symbol_width_pred = symbol_width_list[idx]
            # else:
            #     # - 根据先验选择最近的宽度值 但本地实验有弱负面影响
            #     allowed_widths = torch.tensor(
            #         [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], device=device
            #     ) / EBDSC3rdLoader.SYMBOL_WIDTH_UNIT
            #     # 计算与每个允许值的距离
            #     distances = torch.abs(symbol_width_pred.unsqueeze(1) - allowed_widths)
            #     # 选择最近的允许值
            #     symbol_width_pred = allowed_widths[torch.argmin(distances, dim=1)]

            code_seq_preds = reverse_sequence_from_logits_batch(
                symbol_width_absl=symbol_width_pred * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT,
                expanded_logits=code_seq_logits,
                pad=PAD_IDX,
                sample_rate=SAMPLE_RATE,
            )

            generated_code_seq = torch.where(
                code_seq_preds != PAD_IDX, code_seq_preds - CODE_MAP_OFFSET, code_seq_preds
            )
            
            generated_code_seq = generated_code_seq.tolist()

            for i in range(len(filenames)):
                result_list.append(
                    [
                        filenames[i],
                        mod_preds[i].item() + 1,
                        symbol_width_pred[i].item(),
                        " ".join(map(str, generated_code_seq[i])),
                    ]
                )
            # break

    return result_list

def get_result_vote(device, loaders, model, vote_times):
    """带投票的推理，通过多次对同一批数据进行预测并合并结果"""
    
    result_list = []
    
    with torch.no_grad():
        for i_l in tqdm(range(len(loaders[0]))):
            
            # 存储多轮推理的结果
            all_mod_probs = []  # 改为存储概率而非logits
            all_symbol_width_preds = []
            all_code_seq_logits = []
            
            # 进行多轮推理
            for i_v in range(vote_times):
                loader = loaders[i_v]
                batch = next(loader)
                IQ_data = batch["IQ_data"].to(device)
                filenames = batch["filename"]
            
                mod_logits, symbol_width_pred, code_seq_logits = model(IQ_data)
                
                # 对调制类型先应用softmax转换为概率
                mod_probs = torch.softmax(mod_logits, dim=-1)
                code_seq_probs = torch.softmax(code_seq_logits, dim=-1)
                
                all_mod_probs.append(mod_probs)  # 存储softmax后的概率
                all_symbol_width_preds.append(symbol_width_pred)
                all_code_seq_logits.append(code_seq_probs)
            
            # 对多轮结果取平均
            mod_probs_avg = torch.mean(torch.stack(all_mod_probs), dim=0)  # 平均概率
            symbol_width_avg = torch.mean(torch.stack(all_symbol_width_preds), dim=0)
            code_seq_probs_avg = torch.mean(torch.stack(all_code_seq_logits), dim=0)
            
            # 直接使用阈值函数处理平均后的概率
            max_probs, mod_preds = torch.max(mod_probs_avg, dim=1)
            unknown_mask = max_probs < THRESHOLD
            mod_preds[unknown_mask] = 10

            code_seq_preds = reverse_sequence_from_logits_batch(
                symbol_width_absl=symbol_width_avg * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT,
                expanded_logits=code_seq_probs_avg,
                pad=PAD_IDX,
                sample_rate=SAMPLE_RATE,
            )

            generated_code_seq = torch.where(
                code_seq_preds != PAD_IDX, code_seq_preds - CODE_MAP_OFFSET, code_seq_preds
            )
            
            generated_code_seq = generated_code_seq.tolist()

            for i in range(len(filenames)):
                result_list.append(
                    [
                        filenames[i],
                        mod_preds[i].item() + 1,
                        symbol_width_pred[i].item(),
                        " ".join(map(str, generated_code_seq[i])),
                    ]
                )

    return result_list

def eva(testpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type != "cpu", "请使用 GPU"

    # 加载测试集

    dataset = EBDSC3rdLoader(
        testpath,
        is_test=True,
        demodulator=Demodulator(step=1, freq_topk=-1),  # <- NOTE
    )

    # loader = DataLoader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=make_collate_fn(is_test=True),
    #     num_workers=8,
    #     pin_memory=True if torch.cuda.is_available() else False,
    # )
    # 创建新的数据加载器，每个用于一轮投票
    loaders = []
    for _ in range(VOTE_TIMES):
        # 创建相同参数但可能有不同随机增强的数据加载器
        new_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=make_collate_fn(is_test=True),
            num_workers=8,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        loaders.append(iter(new_loader))

    cp = torch.load("./freTSTCN_51KS5_128D20L2R5dp_3rd_AttnPool_sr7_XXXL_FreTS原复2c_随移1c_best.pth", map_location=device)
    # 加载模型
    print(f"{cp['epoch']=}")
    model = cp["model"]
    # model.structural_reparam()
    # torch.save(model, "model.pth")
    # model = torch.load("./model.pth", map_location=device)
    
    # print(model)
    result_list = get_result_vote(device, loaders, model, VOTE_TIMES)

    return result_list


@timer("main")
def main(to_pred_dir, result_save_path):
    """
    主函数，用于执行脚本的主要逻辑。

    参数：
        to_pred_dir: 测试集文件夹上层目录路径，不可更改！
        result_save_path: 预测结果文件保存路径，官方已指定为 csv 格式，不可更改！
    """
    # 获取测试集文件夹路径，不可更改！
    testpath = os.path.join(os.path.abspath(to_pred_dir), "test")

    # 初始化结果文件，定义表头
    # result = ["file_name,modulation_type,symbol_width,code_sequence"]
    result_columns_name = ["file_name", "modulation_type", "symbol_width", "code_sequence"]

    result_list = eva(testpath)
    result_df = pd.DataFrame(result_list, columns=result_columns_name)
    # 将预测结果保存到 result_save_path，保存方式可修改，但是注意保存路径不可更改！！！
    # 如果 result 为已经预测好的 DataFrame 数据，则可以直接使用 pd.to_csv() 的方式进行保存
    result_df.to_csv(result_save_path, index=None)

    return result_df


if os.environ.get("USER") == "framist":
    main("./data", "./result.csv")

    evaluate_result("result.csv", "./data/test")
    exit(0)
if os.environ.get("USER") in ["hxn", "huxianan"]:
    BATCH_SIZE *= 2
    main("./data", "./result.csv")

    evaluate_result("result.csv", "./data/test")
    exit(0)
else:
    BATCH_SIZE *= 2
    
    
if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 官方给出的测试文件夹上层的路径，不可更改！
    result_save_path = sys.argv[2]  # 官方给出的预测结果保存文件路径，已指定格式为 csv，不可更改！
    main(to_pred_dir, result_save_path)  # 运行 main 脚本，入参只有 to_pred_dir, result_save_path，不可更改！


"""
2025/2/28 ♥️ by Framist
"""
