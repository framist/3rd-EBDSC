import datetime
import sys

import numpy as np
import torch
import torch.utils.data as Data
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

now = datetime.datetime.now().strftime('%m%d_%H-%M')

from thop import clever_format, profile
from torch.utils.data import random_split

from my_tools import *

seed_everything()

import wandb
from ebdsc3rd_datatools import *
from loss import MultiTaskLoss

NAME = '3rd_freq'

# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 中文字体设置
# plt.rcParams['axes.unicode_minus'] = False  # 负号显示设置

import argparse

parser = argparse.ArgumentParser(description='Code for 3nd EBDSC -- Wide-Value-Embs TCN -- by framist',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cuda', type=int, default=0, help='所使用的 cuda 设备，暂不支持多设备并行')
parser.add_argument('--num_layers', type=int, default=24, help='layers of modernTCN')
parser.add_argument('--d_model', type=int, default=128, help='d_model')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--ratio', type=int, default=2, help='ffn ratio')
parser.add_argument('--ls', type=int, default=51, help='large kernel sizes')
parser.add_argument('--ss', type=int, default=5, help='small kernel size')
parser.add_argument('--dp', type=float, default=0.5, help='drop out')
parser.add_argument('--max_epoch', type=int, default=64, help='max train epoch')

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
parser.add_argument('--true_sym_width', action='store_true', default=False, help='是否使用真实的符号宽度')
parser.add_argument('--max_code_len', type=int, default=400, help='最大码元长度')
parser.add_argument('--mutitask_weights', nargs='+', type=float, default=[0.2, 0.3, 0.5], help='多任务损失权重 for MT, SW, CQ')
parser.add_argument('--mod_uniq_sym', action='store_true', default=False, help='是否使用 mod 独立的符号')
parser.add_argument('--data_aug', action='store_true', default=False, help='是否使用数据增强')
parser.add_argument('--meanpool', action='store_true', default=False, help='是否使用 meanpool 而非 attn pool 作为池化')
parser.add_argument('--demod_step', type=int, default=0, help='Demodulator step')
parser.add_argument('--demod_br', type=float, default=1, help='Demodulator band width rate min=0.5')

parser.add_argument('--best_continue', action='store_true', default=False, help='是否继续训练')

parser_args = parser.parse_args()

use_cuda = True
device = torch.device(f"cuda:{parser_args.cuda}" if (
    use_cuda and torch.cuda.is_available()) else "cpu")
# device = torch.device("cpu")
print("CUDA Available: ", torch.cuda.is_available(), 'use:', device)


BATCH_SIZE = parser_args.batch_size
INPUT_CHANNELS = 2

# kernel_size = 51  # 51 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35
POS_D = D = parser_args.d_model
P = 1
S = 1
R = parser_args.ratio
NUM_LAYERS = parser_args.num_layers
DROP_OUT = parser_args.dp # dropout 仅指分类头的。不能在主干。ref. https://arxiv.org/abs/1801.05134v1

MAX_TRAIN_EPOCH = parser_args.max_epoch

IF_LAERNABLE_EMB = True
NAME += "_MeanPool" if parser_args.meanpool else "_AttnPool"
NAME += f"_{parser_args.name}"

root_dir = "../train_data/"  # 替换为实际路径
CODE_MAP_OFFSET = 1  # 码元映射偏移

# 创建 train_loader
full_dataset = EBDSC3rdLoader(
    root_dir=root_dir,
    demodulator=Demodulator(
        bandwidth_ratio=parser_args.demod_br,
        step=parser_args.demod_step
    ),
    code_map_offset=CODE_MAP_OFFSET,
    mod_uniq_symbol=parser_args.mod_uniq_sym,
    data_aug=parser_args.data_aug,
    is_test=False
)
if parser_args.mod_uniq_sym:
    NAME += "_mod_uniq_sym"

NUM_CODE_CLASSES = full_dataset.num_code_classes
NUM_MOD_CLASSES = full_dataset.num_mod_classes
PAD_IDX = 0  # 填充符号 ID
MAX_CODE_LENGTH = parser_args.max_code_len
MUTITASK_WEIGHTS = parser_args.mutitask_weights

if MUTITASK_WEIGHTS[1] == 0:
    assert parser_args.true_sym_width, "SW 为 0 时，必须使用真实符号宽度" 

is_debug = True if sys.gettrace() else False
if is_debug:
    print("\n!!!!!!!!!!!!!!! Debugging !!!!!!!!!!!!!!!\n")
    assert parser_args.wandb == False, "Debugging 时不支持 wandb"
    
# %% 模型、优化器选择
if parser_args.model == 'modernTCN':
    NAME = f'TCN_{parser_args.ls}KS{parser_args.ss}_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
    # from TCNmodelPosAll import ModernTCN_DC
    from ModernTCN import ModernTCN_MutiTask

    # 不可结构重参数化：
    # model = ModernTCN_DC(INPUT_CHANNELS, WINDOW_SIZE, TAG_LEN, D=D,
    #                      P=P, S=S, kernel_size=kernel_size, r=R, num_layers=NUM_LAYERS, pos_D=POS_D).to(device)
    # 可结构重参数化：
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
            stem = IF_LAERNABLE_EMB,
            mean_pool=parser_args.meanpool
        ).to(device)


    # * CNN 使用的优化器

    learn_rate = parser_args.lr
    lr_step_size = parser_args.lr_step_size
    # learn_rate = 4e-3
    # learn_rate = 2e-4
    # MIN_LR = 1e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # < 50  e 0.004
    # < 100 e 0.002
    # < 150 e 0.001
    # < 200 e 0.0005
    # < 250 e 0.00025
    # < 300 e 0.000125
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5)


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

    # * TF 使用的优化器
    learn_rate = parser_args.lr
    optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)
    lr_lambda = lambda step: (D ** -0.5) * min((step+1) ** -0.5, (step+1) * 16 ** -1.5)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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

    # * TF 使用的优化器
    learn_rate = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    lr_lambda = lambda step: (D ** -0.5) * min((step+1) ** -0.5, (step+1) * 16 ** -1.5) 
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    # 优化器
    learn_rate = 0.001
    optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

else:
    raise ValueError('model 选择错误')

if parser_args.wandb:
    wandb.init(
        project="TCN 3rd freq",
        name=parser_args.name,
        config=parser_args,
        tags=parser_args.tags,
        notes=NAME,
        save_code=True,
    )


if 0. in MUTITASK_WEIGHTS:
    print(f'Warning: 0 in {MUTITASK_WEIGHTS=}')
criterion = MultiTaskLoss(*MUTITASK_WEIGHTS, pad_idx=PAD_IDX)

# 定义训练集和验证集的比例，例如 80% 训练，20% 验证
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
    num_workers=8,
    pin_memory=bool(torch.cuda.is_available()),
)

# 创建验证集 DataLoader
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=make_collate_fn(),
    num_workers=8,
    pin_memory=bool(torch.cuda.is_available()),
)


print(f"{NAME=}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"{model=}")




if parser_args.best_continue:
    # model.load_state_dict(torch.load(f"./saved_models/{NAME}_best.pth", map_location=device))    
    epoch_start = load_checkpoint(model, f"./saved_models/{NAME}_best.pth", optimizer, device)
else:
    epoch_start = 0

scaler = GradScaler()
torch.cuda.empty_cache()
model.to(device)
model.train()
best_score = 0.0
t = tqdm(range(epoch_start, epoch_start + MAX_TRAIN_EPOCH), dynamic_ncols=True)
scaler = torch.cuda.amp.GradScaler()
for epoch in t:
    # - 训练阶段
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        IQ_data = batch["IQ_data"].to(device)  # [batch_size, max_IQ_len, 2]
        code_sequence_aligned = batch["code_sequence_aligned"].to(device)  # [batch_size, max_IQ_len]
        code_mask = batch["code_mask"].to(device)  # [batch_size, max_code_len]
        mod_type = batch["mod_type"].to(device)  # [batch_size]
        symbol_width = batch["symbol_width"].to(device)  # [batch_size]

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

    # - 验证阶段
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
                )
            else:
                # 不评分
                code_sed_pred = torch.zeros_like(code_sequence)

            # 计算指标
            MT_scores = compute_MT_score(mod_logits, mod_type)
            SW_scores = compute_SW_score(symbol_width_pred, symbol_width)
            CQ_scores, cs = compute_CQ_score(
                code_sed_pred,
                code_sequence,
                pad_idx=PAD_IDX,
                code_map_offset=full_dataset.code_map_offset,
                mod_uniq_symbol=(full_dataset.mod_uniq_symbol, mod_logits, mod_type),
            )
            acc = compute_sequence_accuracy(code_sed_pred, code_sequence, pad_idx=PAD_IDX)

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
    all_mod_labels = torch.cat(all_mod_labels)  # [num_val_samples]
    all_mod_preds = torch.cat(all_mod_preds)  # [num_val_samples]
    avg_acc = torch.cat(all_acc).mean().item()
    all_cs = torch.cat(all_cs).mean().item()

    # 计算加权总分
    avg_sample_score = 0.2 * avg_MT_scores + 0.3 * avg_SW_scores + 0.5 * avg_CQ_scores

    tqdm.write(
        f"[{epoch+1}/{MAX_TRAIN_EPOCH}] Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, Val Score: {avg_sample_score:.2f}, "
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
        log["acc"] = avg_acc
        log["cs"] = all_cs

    if epoch % 8 == 0:
        # 绘制 类别识别 混淆矩阵
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


# 打印一个批次的数据形状
for batch in val_loader:
    print("IQ_data.shape:", batch["IQ_data"].shape)  # [batch_size, max_IQ_len, 2]

    flops, params = profile(model, inputs=(batch["IQ_data"].to(device),))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    break

wandb.finish()
