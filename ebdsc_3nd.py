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

    # 起始偏置
    # BPSK: {0, 1}
    # QPSK: {0, 1, 2, 3}
    # 8PSK: {0, 1, 2, 3, 4, 5, 6, 7}
    # MSK: {0, 1}
    # 8QAM: {0, 1, 2, 3, 4, 5, 6, 7}
    # 16QAM: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    # 32QAM: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    # 8APSK: {0, 1, 2, 3, 4, 5, 6, 7}
    # 16APSK: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    # 32APSK: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    START_OFFSET = [0, 2, 6, 14, 16, 24, 40, 72, 80, 96, 128]  # 0 ~ 10(UNKNOWN)

    def __init__(self, root_dir, code_map_offset: int = 1, mod_uniq_symbol: bool = False):
        """
        Args:
            root_dir (str): 数据集根目录。
            code_map_offset (int): 符号到整数的映射偏移量。
                e.g.
                    - 0 -> <PAD> (填充符号)
                    - 1 -> <UNK> (未知符号)
                    - 2 -> <SOS> (起始符号)
                    - 3 -> <EOS> (终止符号)
        """
        self.samples = []
        self.code_map_offset = code_map_offset
        self.mod_uniq_symbol = mod_uniq_symbol

        # 符号类别数
        if mod_uniq_symbol:
            self.num_code_classes = EBDSC3rdLoader.START_OFFSET[-1] + 1
        else:
            self.num_code_classes = 32
        self.num_code_classes = self.num_code_classes + code_map_offset

        # mod 类别数
        self.num_mod_classes = 11

        for label in range(11):
            folder_path = os.path.join(root_dir, EBDSC3rdLoader.MOD_TYPE[label])
            if os.path.isdir(folder_path):
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
                self.samples.extend(iter(files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        df = pd.read_csv(file_path, names=EBDSC3rdLoader.DATA_COLUMNS, header=None)

        # - 提取 I/Q 数据
        IQ_data = df[["I", "Q"]].dropna().to_numpy().astype(np.float32)
        IQ_data = (IQ_data - IQ_data.mean(axis=0)) / IQ_data.std(axis=0)  # 标准化

        # - 提取 mod_type 和 symbol_width
        # mod_type = int(df["mod_type"].dropna().iloc[0]) if not df["mod_type"].dropna().empty else 11  # 11 表示 UNKNOWN
        mod_type = df["mod_type"].dropna().astype(int).iloc[0] - 1  # 从 0 开始
        symbol_width = df["symbol_width"].dropna().iloc[0]

        # - 提取 code_sequence
        code_sequence = df["code_sequence"].dropna().astype(int).to_numpy()

        # 让位 PAD
        mapped_code_sequence = code_sequence + self.code_map_offset
        # 映射为唯一的整数 ID
        if self.mod_uniq_symbol:
            mapped_code_sequence = unique_symbol(mapped_code_sequence, mod_type)

        # - 对齐 code_sequence
        code_sequence_aligned = repeat_and_pad_sequence(
            symbol_width * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT, len(IQ_data), mapped_code_sequence, EBDSC3rdLoader.PAD
        )

        return {
            "IQ_data": torch.from_numpy(IQ_data),  # [seq_len, 2]
            "code_sequence_aligned": torch.tensor(code_sequence_aligned, dtype=torch.long),  # [code_len]
            "mod_type": torch.tensor(mod_type, dtype=torch.long),  # scalar
            "symbol_width": torch.tensor(symbol_width, dtype=torch.float32),  # scalar
            "IQ_length": torch.tensor(len(IQ_data), dtype=torch.long),  # scalar
            "code_sequence": torch.tensor(mapped_code_sequence, dtype=torch.long),  # [code_len]
        }


def unique_symbol(code_sequence: np.ndarray, mod_type: int):
    """
    码元符号根据 mod_type 唯一化
    modtpye: 0 ~ 10
    """
    start = EBDSC3rdLoader.START_OFFSET[mod_type]
    return code_sequence + start


def un_unique_symbol(code_sequence, mod_type):
    """
    码元符号根据 mod_type 反唯一化
    modtpye: 0 ~ 10
    """
    start = EBDSC3rdLoader.START_OFFSET[mod_type]
    return code_sequence - start


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
) -> torch.LongTensor:
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
    original_sequences = torch.full((batch_size, max_orig_len), pad, device=expanded_probs.device, dtype=torch.long)

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
    TODO 限制 IQ 与 code_sequence 的长度
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
    preds = torch.argmax(mod_logits, dim=-1)
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


def _compute_CQ_score(
    code_seq_preds: torch.LongTensor,
    code_seq_labels: torch.LongTensor,
    pad_idx: int = 0,
    code_map_offset: int = 1,
    mod_uniq_symbol: tuple = (False, None, None),
):
    """
    计算码序列解调余弦相似度得分 (CQ_score)。
    NOTE 计算时需减去 code_map_offset
    TODO 提高计算效率
    Args:
        code_seq_preds (Tensor): [batch_size, tgt_seq_len]
        code_seq_labels (Tensor): [batch_size, tgt_seq_len]
        pad_idx (int): 填充符号的索引
    Returns:
        Tensor: [batch_size] 每个样本的 CQ_score
    """
    batch_size, tgt_seq_len = code_seq_labels.size()
    if mod_uniq_symbol[0]:
        mod_pred = torch.argmax(mod_uniq_symbol[1], dim=-1)

    scores = []
    true_seqs = code_seq_labels.cpu().numpy()
    pred_seqs = code_seq_preds.cpu().numpy()

    for i in range(batch_size):

        # 截断或填充预测序列 true_seqs[i] 第一个 0
        true_length = np.sum(true_seqs[i] != pad_idx)

        pred_seq = pred_seqs[i, :true_length]
        true_seq = true_seqs[i, :true_length]
        if len(pred_seq) < true_length:
            pred_seq = np.pad(
                pred_seq, (0, true_length - len(pred_seq)), "constant", constant_values=code_map_offset
            )  # NOTE 使用 code_map_offset 填充

        # 计算余弦相似度
        true_vec = true_seq.astype(float) - code_map_offset
        pred_vec = pred_seq.astype(float) - code_map_offset
        if mod_uniq_symbol[0]:
            # TODO
            pred_vec = un_unique_symbol(pred_vec, mod_pred[i])
            true_vec = un_unique_symbol(true_vec, mod_uniq_symbol[2][i])

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
    return torch.tensor(scores, device=code_seq_preds.device)


def compute_CQ_score(
    code_seq_preds: torch.Tensor,
    code_seq_labels: torch.Tensor,
    pad_idx: int = 0,
    code_map_offset: int = 1,
    mod_uniq_symbol: tuple = (False, None, None),
) -> torch.Tensor:
    """
    计算码序列解调余弦相似度得分 (CQ_score)。
    NOTE 计算时需减去 code_map_offset
    Args:
        code_seq_preds (Tensor): [batch_size, tgt_seq_len] 预测的码序列
        code_seq_labels (Tensor): [batch_size, tgt_seq_len] 真实的码序列
        pad_idx (int): 填充符号的索引
        code_map_offset (int): 码映射偏移量
    Returns:
        Tensor: [batch_size] 每个样本的 CQ_score
    """
    device = code_seq_preds.device
    batch_size_pred, pred_seq_len = code_seq_preds.size()
    batch_size, tgt_seq_len = code_seq_labels.size()

    # TODO
    if mod_uniq_symbol[0]:
        offset = torch.tensor(EBDSC3rdLoader.START_OFFSET, device=device, dtype=torch.long)
        
        mod_preds = torch.argmax(mod_uniq_symbol[1], dim=-1)
        code_seq_preds = torch.where(code_seq_preds != 0, code_seq_preds - offset[mod_preds].unsqueeze(1), code_seq_preds)
        code_seq_preds = code_seq_preds.clamp(min=0+code_map_offset)    # TODO
        
        mod_labels = mod_uniq_symbol[2]        
        code_seq_labels = torch.where(code_seq_labels != 0, code_seq_labels - offset[mod_labels].unsqueeze(1), code_seq_labels)
        # TODO
        assert torch.sum(code_seq_labels < 0) == 0
        assert torch.sum(code_seq_labels > 32) == 0

    
    
    # 计算每个样本的真实长度（非 pad 的长度）
    true_lengths = (code_seq_labels != pad_idx).sum(dim=1)  # [batch_size]

    # 确定每个样本的最大真实长度，以便截断预测序列
    max_true_length = true_lengths.max().item()

    # 如果预测序列长度超过最大真实长度，进行截断；否则，进行填充
    if pred_seq_len > max_true_length:
        # 截断预测序列
        pred_seq_truncated = code_seq_preds[:, :max_true_length]
    else:
        # 填充预测序列，使其长度与最大真实长度一致
        padding_size = max_true_length - pred_seq_len

        padding = torch.full(
            (batch_size_pred, padding_size), code_map_offset, dtype=code_seq_preds.dtype, device=device
        )
        pred_seq_truncated = torch.cat([code_seq_preds, padding], dim=1)

    # 创建掩码，表示每个位置是否在真实长度内
    mask = torch.arange(max_true_length, device=device).unsqueeze(0).expand(
        batch_size, max_true_length
    ) < true_lengths.unsqueeze(
        1
    )  # [batch_size, max_true_length]

    # 将序列转换为向量，并减去 code_map_offset
    true_vec = (
        code_seq_labels[:, :max_true_length] - code_map_offset
    ).float() * mask.float()  # [batch_size, max_true_length]
    pred_vec = (pred_seq_truncated - code_map_offset).float() * mask.float()  # [batch_size, max_true_length]

    # 计算余弦相似度
    dot_product = (true_vec * pred_vec).sum(dim=1)  # [batch_size]
    norm_true = true_vec.norm(p=2, dim=1)  # [batch_size]
    norm_pred = pred_vec.norm(p=2, dim=1)  # [batch_size]

    # 处理范数为零的情况，避免除以零
    cosine_similarity = torch.where(
        (norm_true == 0) | (norm_pred == 0), torch.tensor(0.0, device=device), dot_product / (norm_true * norm_pred)
    )

    # print(f"{cosine_similarity=}")
    # 转换为 CQ_score
    CQ_score = torch.zeros_like(cosine_similarity)
    # CS > 0.95
    mask_high = cosine_similarity > 0.95
    CQ_score[mask_high] = 100.0
    # 0.7 <= CS <= 0.95
    mask_mid = (cosine_similarity >= 0.7) & (cosine_similarity <= 0.95)
    CQ_score[mask_mid] = ((cosine_similarity[mask_mid] - 0.7) / (0.95 - 0.7)) * 100
    # CS < 0.7 保持为 0

    # 确保分数在 [0, 100] 范围内
    CQ_score = CQ_score.clamp(0.0, 100.0)
    

    return CQ_score

import torch

def compute_sequence_accuracy(
    code_seq_preds: torch.Tensor,
    code_seq_labels: torch.Tensor,
    pad_idx: int = 0,
    code_map_offset: int = 1
) -> torch.Tensor:
    """
    计算序列准确率（Sequence Accuracy）。

    序列准确率衡量预测序列与真实序列在所有非填充位置上匹配的比例。

    Args:
        code_seq_preds (Tensor): [batch_size, pred_seq_len] 预测的码序列。
        code_seq_labels (Tensor): [batch_size, label_seq_len] 真实的码序列。
        pad_idx (int, optional): 填充符号的索引。默认为 0。
        code_map_offset (int, optional): 码映射偏移量。默认为 1。

    Returns:
        Tensor: [batch_size] 每个样本的序列准确率，值介于 0.0 和 1.0 之间。
    """
    device = code_seq_preds.device
    batch_size_pred, pred_seq_len = code_seq_preds.size()
    batch_size_label, label_seq_len = code_seq_labels.size()

    if batch_size_pred != batch_size_label:
        raise ValueError("code_seq_preds 和 code_seq_labels 的 batch_size 必须相同")

    # 计算每个样本的真实长度（非 pad 的长度）
    true_lengths = (code_seq_labels != pad_idx).sum(dim=1)  # [batch_size]

    # 确定每个样本的最大真实长度
    max_true_length = true_lengths.max().item()

    # 截断或填充预测序列至最大真实长度
    if pred_seq_len > max_true_length:
        # 截断预测序列
        pred_seq_truncated = code_seq_preds[:, :max_true_length]
    else:
        # 填充预测序列，使其长度与最大真实长度一致
        padding_size = max_true_length - pred_seq_len
        if padding_size > 0:
            padding = torch.full(
                (batch_size_pred, padding_size),
                pad_idx,
                dtype=code_seq_preds.dtype,
                device=device
            )
            pred_seq_truncated = torch.cat([code_seq_preds, padding], dim=1)
        else:
            pred_seq_truncated = code_seq_preds

    # 截断真实序列至最大真实长度
    true_seq_truncated = code_seq_labels[:, :max_true_length]

    # 创建一个掩码，表示每个位置是否在真实长度内
    # [batch_size, max_true_length]
    mask = torch.arange(max_true_length, device=device).unsqueeze(0).expand(batch_size_label, max_true_length) < true_lengths.unsqueeze(1)

    # 填充预测序列：使用 code_map_offset 填充预测序列的无效位置
    # 这样在减去 code_map_offset 后，填充位置的值为 0
    pred_seq_padded = torch.where(
        mask,
        pred_seq_truncated,
        torch.full_like(pred_seq_truncated, code_map_offset)
    )

    # 将序列转换为向量，并减去 code_map_offset
    # 有效位置的值为 (value - code_map_offset)
    # 填充位置的值为 (code_map_offset - code_map_offset) = 0
    true_vec = (true_seq_truncated - code_map_offset).float() * mask.float()  # [batch_size, max_true_length]
    pred_vec = (pred_seq_padded - code_map_offset).float() * mask.float()    # [batch_size, max_true_length]

    # 比较预测序列与真实序列是否匹配
    # 仅考虑掩码为 True 的位置
    correct = (true_vec == pred_vec) & mask  # [batch_size, max_true_length]

    # 计算每个样本的正确预测数
    correct_counts = correct.sum(dim=1).float()  # [batch_size]

    # 计算每个样本的准确率
    # 避免除以零：将 true_lengths 中为 0 的样本设定为 1.0（定义为空序列为完全正确）
    per_sample_accuracy = correct_counts / true_lengths.clamp(min=1).float()  # [batch_size]

    # 对于 true_length 为 0 的样本，设定准确率为 1.0
    per_sample_accuracy = torch.where(
        true_lengths > 0,
        per_sample_accuracy,
        torch.ones_like(per_sample_accuracy)
    )

    return per_sample_accuracy

# 示例用法
if __name__ == "__main__":
    # 示例数据
    code_seq_preds = torch.tensor([
        [2, 3, 4, 5, 6, 7],   # 预测序列长度 6
        [1, 2, 3, 4, 0, 0],   # 预测序列长度 4
        [3, 3, 3, 3, 3, 3],   # 预测序列长度 6
        [0, 0, 0, 0, 0, 0]    # 预测序列全是 pad
    ])
    
    code_seq_labels = torch.tensor([
        [2, 3, 4, 5, 0, 0],   # 真实序列长度 4
        [1, 2, 3, 4, 5, 0],   # 真实序列长度 5
        [3, 3, 3, 3, 3, 3],   # 真实序列长度 6
        [0, 0, 0, 0, 0, 0]    # 真实序列全是 pad
    ])
    
    pad_idx = 0
    code_map_offset = 1
    
    # 计算序列准确率
    per_sample_accuracy = compute_sequence_accuracy(code_seq_preds, code_seq_labels, pad_idx, code_map_offset)
    
    print("Per-sample Accuracy:", per_sample_accuracy)

    
    # for i in range(1, 32):
    #     l = torch.ones((2, 32)) + 5
    #     p = torch.randint(0, i, l.size())
    #     # assert compute_CQ_score(p, l, pad_idx=0, code_map_offset=1)[0] == 0.0
    #     s = compute_CQ_score(p, l, pad_idx=0, code_map_offset=1)
    #     print(f"{i}: {s.mean().item()}")

    # 示例数据
    code_seq_preds = torch.tensor(
        [
            [2, 3, 1, 0, 0],
            [1, 2, 3, 4, 0],
            [3, 3, 3, 3, 3],
            [3, 2, 4, 5, 2],
            [1, 1, 2, 1, 2],
            [1, 1, 2, 1, 2],
        ]
    )

    code_seq_labels = torch.tensor(
        [
            [2, 3, 1, 1, 0],
            [1, 2, 3, 0, 0],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [1, 1, 1, 1, 2],
            [11, 11, 12, 11, 12],
        ]
    )

    pad_idx = 0
    code_map_offset = 1

    # 计算 CQ_score
    cq_scores = compute_CQ_score(code_seq_preds, code_seq_labels, pad_idx, code_map_offset)
    print(cq_scores)
