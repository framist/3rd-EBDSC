# %%
import itertools
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np


# %%
# 1. 定义 Dataset 类
class EBDSC3rdLoader(Dataset):
    # 16APSK  16QAM  32APSK  32QAM  8APSK  8PSK  8QAM  BPSK  MSK  QPSK  UNKNOWN
    # 注意文件中的 mod_type 从 1 开始
    # 1: BPSK, 2: QPSK, 3: 8PSK, 4: MSK, 5: 8QAM, 6: 16QAM, 7: 32QAM, 8: 8APSK, 9: 16APSK, 10: 32APSK, 11: UNKNOWN
    MOD_TYPE = dict(
        zip(
            range(11),
            ["BPSK", "QPSK", "8PSK", "MSK", "8QAM", "16QAM", "32QAM", "8APSK", "16APSK", "32APSK", "UNKNOWN"],
        )
    )
    DATA_COLUMNS = ["I", "Q", "code_sequence", "mod_type", "symbol_width"]
    PAD = 0

    # 码元宽度的单位与 IQ 单位差 20 倍
    SYMBOL_WIDTH_UNIT = 20

    def __init__(self, root_dir, code_map_offset=1, max_code_length=400):
        """
        Args:
            root_dir (str): 数据集根目录。
            code_map_offset (int): 符号到整数的映射偏移量。
                e.g.
                    - 0 -> <PAD> (填充符号)
                    - 1 -> <UNK> (未知符号)
                    - 2 -> <SOS> (起始符号)
                    - 3 -> <EOS> (终止符号)
            max_code_length (int): 码序列的最大长度（用于填充）。
        """
        self.samples = []
        self.code_mapping = code_map_offset
        self.num_code_classes = 32 + code_map_offset  # 符号类别数
        self.max_code_length = max_code_length

        for label in range(11):
            folder_path = os.path.join(root_dir, self.MOD_TYPE[label])
            if os.path.isdir(folder_path):
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
                self.samples.extend(iter(files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        df = pd.read_csv(file_path, names=self.DATA_COLUMNS, header=None)

        # 提取 I/Q 数据
        IQ_data = df[["I", "Q"]].dropna().to_numpy().astype(np.float32)
        IQ_data = (IQ_data - IQ_data.mean(axis=0)) / IQ_data.std(axis=0)  # 标准化

        # 提取 mod_type 和 symbol_width
        # mod_type = int(df["mod_type"].dropna().iloc[0]) if not df["mod_type"].dropna().empty else 11  # 11 表示 UNKNOWN
        mod_type = df["mod_type"].dropna().iloc[0] - 1  # 从 0 开始
        symbol_width = df["symbol_width"].dropna().iloc[0]

        # 提取 code_sequence，映射为整数 ID
        code_sequence = df["code_sequence"].dropna().astype(int).to_numpy()
        # 根据 code_mapping 将符号映射为唯一的整数 ID
        mapped_code_sequence = code_sequence + self.code_mapping
        code_sequence_aligned = repeat_and_pad_sequence(
            symbol_width * self.SYMBOL_WIDTH_UNIT, len(IQ_data), mapped_code_sequence, self.PAD
        )

        return {
            "IQ_data": torch.from_numpy(IQ_data),  # [seq_len, 2]
            "code_sequence_aligned": torch.tensor(code_sequence_aligned, dtype=torch.long),  # [code_len]
            "mod_type": torch.tensor(mod_type, dtype=torch.long),  # scalar
            "symbol_width": torch.tensor(symbol_width, dtype=torch.float32),  # scalar
            "IQ_length": torch.tensor(len(IQ_data), dtype=torch.long),  # scalar
            "code_sequence": torch.tensor(mapped_code_sequence, dtype=torch.long),  # [code_len]
        }


def repeat_and_pad_sequence(symbol_width_absl: float, length: int, code_sequence: np.ndarray, pad: int = 0):
    """code_sequence 拓展为 IQ 数据长度 len(mapped_code_sequence) = len(IQ_data)
    e.g.
        |<------->| = symbol_width * SYMBOL_WIDTH_UNIT
        [1,       2,       3... 1] ->
        [1, 1, 1, 2, 2, 2, 3... 1]
        |<---------------------->| = len(IQ_data)
    计算每个码元需要重复的次数

    Args:
        symbol_width (float): 码元宽度。
        length (int): IQ 数据长度。
        code_sequence (np.ndarray): 码序列。
    Returns:
        np.ndarray: 拓展后的码序列。
    """
    repeat_count = int(symbol_width_absl)  # 数据集中此项一定是整数
    # 使用 numpy 的 repeat 函数展开码序列
    s = np.repeat(code_sequence, repeat_count)
    # 确保展开后的长度与 IQ 数据长度匹配
    if len(s) > length:
        s = s[:length]
    elif len(s) < length:
        pad_length = length - len(s)
        s = np.pad(s, (0, pad_length), "constant", constant_values=pad)

    return s


def repeat_and_pad_sequence_batch(
    symbol_width_absl: torch.Tensor, length: int, code_sequence: torch.Tensor, pad: int = 0
) -> torch.Tensor:
    """code_sequence 拓展为 IQ 数据长度，支持 batch 处理

    Args:
        symbol_width_absl (Tensor): [batch_size] 每个样本的码元宽度
        length (int): 目标序列长度
        code_sequence (Tensor): [batch_size, seq_len] 码序列
        pad (int): 填充值

    Returns:
        Tensor: [batch_size, length] 拓展后的码序列
    """
    batch_size = code_sequence.size(0)

    # repeat_interleave 支持不同的重复次数
    repeat_counts = symbol_width_absl.int()  # [batch_size]
    # repeat_interleave 沿着 seq_len 维度重复 除了指定的 dim 维度外，其他维度的大小相同
    s = torch.repeat_interleave(code_sequence, repeat_counts.unsqueeze(1), dim=-1)

    # 处理长度不匹配
    if s.size(1) > length:
        s = s[:, :length]
    elif s.size(1) < length:
        # 计算需要填充的长度
        pad_length = length - s.size(1)
        # 使用 F.pad 进行填充
        s = F.pad(s, (0, pad_length), value=pad)

    return s


def reverse_sequence(symbol_width_absl: float, expanded_sequence: np.ndarray, pad: int = 0):
    """从展开的序列恢复原始码序列
    e.g.
        [1, 1, 1, 2, 2, 2, 3... 1] ->
        [1,       2,       3... 1]
        |<------->| = symbol_width_absl

    Args:
        symbol_width_absl (float): 码元宽度
        expanded_sequence (np.ndarray): 展开后的序列

    Returns:
        np.ndarray: 原始码序列
    """
    repeat_count = int(symbol_width_absl)
    # 每隔 repeat_count 取一个样本
    original_sequence = expanded_sequence[::repeat_count]
    # 移除末尾的 padding（如果存在）
    # TODO 通过查找最后一个非 PAD 值的位置
    last_non_pad = np.where(original_sequence != pad)[0][-1] if len(original_sequence) > 0 else -1
    return original_sequence[: last_non_pad + 1] if last_non_pad >= 0 else np.array([])


def reverse_sequence_from_logits(symbol_width_absl: float, expanded_logits: np.ndarray, pad: int = 0):
    """从展开的 logits 序列恢复原始码序列，通过选择窗口内累计概率最大的类别

    Args:
        symbol_width_absl (float): 码元宽度
        expanded_logits (np.ndarray): 展开后的 logits 序列 [seq_len, num_classes]

    Returns:
        np.ndarray: 原始码序列
    """
    repeat_count = int(symbol_width_absl)
    seq_len = len(expanded_logits)

    # 计算原始序列长度
    orig_len = seq_len // repeat_count + (1 if seq_len % repeat_count else 0)
    original_sequence = np.zeros(orig_len, dtype=np.int64)

    # 对每个码元位置进行处理
    for i in range(orig_len):
        start_idx = i * repeat_count
        end_idx = min(start_idx + repeat_count, seq_len)

        # 获取当前窗口的 logits 并累加
        window_logits = expanded_logits[start_idx:end_idx]
        summed_logits = np.sum(window_logits, axis=0)

        # 选择累计概率最大的类别
        original_sequence[i] = np.argmax(summed_logits)

    # 移除末尾的 padding
    last_non_pad = np.where(original_sequence != pad)[0][-1] if len(original_sequence) > 0 else -1
    return original_sequence[: last_non_pad + 1] if last_non_pad >= 0 else np.array([])


def reverse_sequence_from_logits_batch(
    symbol_width_absl: torch.Tensor, expanded_logits: torch.Tensor, pad: int = 0
) -> torch.Tensor:
    """从展开的 logits 序列恢复原始码序列，支持批处理

    Args:
        symbol_width_absl (Tensor): [batch_size] 每个样本的码元宽度
        expanded_logits (Tensor): [batch_size, seq_len, num_classes] logits 序列
        pad (int): 填充值

    Returns:
        Tensor: [batch_size, orig_len] 原始码序列
    """
    # * Softmax 转换为概率
    expanded_probs = F.softmax(expanded_logits, dim=-1)
    batch_size, seq_len, num_classes = expanded_probs.shape

    repeat_counts = symbol_width_absl.int()  # [batch_size]
    repeat_counts = torch.clip(repeat_counts, min=5, max=20)  # 根据数据集

    # 计算每个样本的原始序列长度
    orig_lens = torch.div(seq_len, repeat_counts, rounding_mode="floor")
    max_orig_len = orig_lens.max().item()

    # 初始化输出
    original_sequences = torch.full((batch_size, max_orig_len), pad, device=expanded_probs.device)

    # 对每个样本处理
    for b in range(batch_size):
        # 使用 unfold 切分窗口
        windows = expanded_probs[b].unfold(
            0, repeat_counts[b], repeat_counts[b]
        )  # [num_windows, repeat_count, num_classes]

        # 在窗口内求和并取最大值索引
        summed_logits = windows.sum(dim=-1)  # [orig_len, num_classes]
        pred_codes = summed_logits.argmax(dim=-1)  # [orig_len]

        # 填充结果
        orig_len = orig_lens[b]
        original_sequences[b, :orig_len] = pred_codes[:orig_len]

    return original_sequences

    
def collate_fn(batch):
    """
    自定义的 collate 函数，用于处理可变长度的 I/Q 数据和 code_sequence。
    Args:
        batch (list): 包含多个样本，每个样本是一个字典。
    Returns:
        dict: 包含批处理后的 I/Q 数据、code_sequence、掩码、mod_type 和 symbol_width。
    """
    # 提取各个字段
    IQ_data_list = [item["IQ_data"] for item in batch]
    code_seq_list = [item["code_sequence"] for item in batch]
    code_seq_aligned_list = [item["code_sequence_aligned"] for item in batch]
    mod_type_list = [item["mod_type"] for item in batch]
    symbol_width_list = [item["symbol_width"] for item in batch]
    IQ_length_list = [item["IQ_length"] for item in batch]

    # 填充 I/Q 数据
    IQ_padded = torch.nn.utils.rnn.pad_sequence(
        IQ_data_list, batch_first=True, padding_value=0.0
    )  # [batch_size, max_IQ_len, 2]
    IQ_lengths = torch.stack(IQ_length_list)  # [batch_size]

    # 填充 code_sequence_aligned
    code_seq_aligned_list = torch.nn.utils.rnn.pad_sequence(
        code_seq_aligned_list, batch_first=True, padding_value=0
    )  # [batch_size, max_code_len]

    # 填充 code_sequence
    code_padded = torch.nn.utils.rnn.pad_sequence(
        code_seq_list, batch_first=True, padding_value=0
    )  # [batch_size, max_code_len]

    # 创建 code_sequence 的掩码（1 表示有效，0 表示填充）
    code_mask = (code_seq_aligned_list != 0).long()  # [batch_size, max_code_len]

    return {
        "IQ_data": IQ_padded,  # [batch_size, max_IQ_len, 2]
        "IQ_length": IQ_lengths,  # [batch_size]
        "code_sequence_aligned": code_seq_aligned_list,  # [batch_size, max_code_len]
        "code_mask": code_mask,  # [batch_size, max_code_len]
        "mod_type": torch.stack(mod_type_list),  # [batch_size]
        "symbol_width": torch.stack(symbol_width_list),  # [batch_size]
        "code_sequence": code_padded,  # [batch_size, code_len]
    }


def compute_MT_score(mod_logits, mod_labels):
    """
    计算调制类别识别准确率 (Acc)。
    Args:
        mod_logits (Tensor): [batch_size, num_mod_classes]
        mod_labels (Tensor): [batch_size]
    Returns:
        Tensor: [batch_size] 每个样本的 MT_score
    """
    preds = torch.argmax(mod_logits, dim=1)
    correct = (preds == mod_labels).float()
    return correct * 100  # 100 或 0


def compute_SW_score(symbol_width_pred, symbol_width_labels):
    """
    计算码元宽度回归得分 (SW_score)。
    Args:
        symbol_width_pred (Tensor): [batch_size]
        symbol_width_labels (Tensor): [batch_size]
    Returns:
        Tensor: [batch_size] 每个样本的 SW_score
    """
    ER = torch.abs(symbol_width_labels - symbol_width_pred) / symbol_width_pred
    SW_score = torch.where(
        ER <= 0.05,
        torch.tensor(100.0, device=ER.device),
        torch.where(ER > 0.2, torch.tensor(0.0, device=ER.device), 100 - ((ER - 0.05) / (0.2 - 0.05)) * 100),
    )
    SW_score = torch.clamp(SW_score, min=0.0, max=100.0)
    return SW_score


def compute_CQ_score(code_seq_pred: torch.Tensor, code_seq_labels: torch.Tensor, pad_idx: int=0, code_map_offset:int=1):
    """
    计算码序列解调余弦相似度得分 (CQ_score)。
    NOTE 计算时需减去 code_map_offset
    Args:
        code_seq_pred (Tensor): [batch_size, tgt_seq_len]
        code_seq_labels (Tensor): [batch_size, tgt_seq_len]
        pad_idx (int): 填充符号的索引
    Returns:
        Tensor: [batch_size] 每个样本的 CQ_score
    """
    batch_size, tgt_seq_len = code_seq_labels.size()
    scores = []
    for i in range(batch_size):
        true_seq = code_seq_labels[i].cpu().numpy()
        pred_seq = code_seq_pred[i].cpu().numpy()

        # 截断或填充预测序列
        true_length = np.sum(true_seq != pad_idx)
        # pred_length = np.sum(pred_seq != pad_idx)
        max_length = true_length
        pred_seq = pred_seq[:max_length]
        true_seq = true_seq[:max_length]
        if len(pred_seq) < max_length:
            pred_seq = np.pad(pred_seq, (0, max_length - len(pred_seq)), "constant", constant_values=pad_idx)

        # 计算余弦相似度
        true_vec = true_seq.astype(float) - code_map_offset
        pred_vec = pred_seq.astype(float) - code_map_offset
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
        scores.append(CQ_score)
    return torch.tensor(scores, device=code_seq_pred.device)



def confusion_matrix(
    predictions, targets, plot_name: str = None, tag_len: int = 12, average: str = "weighted", if_save=False
):
    """
    混淆矩阵
    """
    confusion_matrix = np.zeros((tag_len, tag_len))

    for target, prediction in zip(targets, predictions):
        confusion_matrix[target, prediction] += 1

    plt.figure()
    plt.imshow(confusion_matrix / np.maximum(1, np.sum(confusion_matrix, axis=1)[:, None]))
    # 同时在方格内显示数值
    for i, j in itertools.product(range(tag_len), range(tag_len)):
        plt.text(j, i, f"{confusion_matrix[i, j]:.0f}", ha="center", va="center", color="blue")
    plt.title("confusion matrix")
    plt.xlabel("prediction")
    plt.ylabel("target")
    plt.colorbar()
    plt.title(f"{plot_name}")
    if if_save:
        plt.savefig(f"saved_figs/{plot_name}.png")
    plt.show()


if __name__ == "__main__":
    p = torch.tensor([0, 10]).unsqueeze(0) + 1
    l = torch.tensor([10, 0]).unsqueeze(0) + 1
    assert compute_CQ_score(p, l, pad_idx=0, code_map_offset=1)[0] == 0.
