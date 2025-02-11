# %%
import itertools
import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from scipy import signal
from sklearn.metrics import accuracy_score
import numpy as np

from typing import Optional

from my_tools import compute_topk_freqs


class Demodulator:
    def __init__(self, freq_topk: int = 4, bandwidth_ratio: float = 1, step: int = 0):
        """初始化解调器"""
        self.freq_topk = freq_topk
        self.bandwidth_ratio = bandwidth_ratio
        self.nyquist = 1 / 2
        self.step = step
        # TODO 实现滤波器的选择

    def _downconvert(self, iq_data: np.ndarray, carrier_freq):
        """将信号下变频到基带

        Args:
            iq_data: 复数 IQ 信号
            carrier_freq: 载波频率

        Returns:
            下变频后的信号
        """
        t = np.arange(len(iq_data))
        local_carrier = np.exp(-2j * np.pi * carrier_freq * t)
        return iq_data * local_carrier

    def _lowpass_filter(self, s, cutoff_freq):
        """低通滤波

        Args:
            signal: 输入信号
            cutoff_freq: 截止频率

        Returns:
            滤波后的信号
        """
        b, a = signal.butter(8, cutoff_freq / self.nyquist, btype="low")
        return signal.filtfilt(b, a, s)

    def demod(self, iq_data: np.ndarray, symbol_width_absl):
        """解调主函数

        Args:
            iq_data: IQ 数据，shape=(N,2)
            symbol_width_absl: 码元宽度
        """
        if self.step <= 0:
            return iq_data

        # 转换为复数 IQ 信号
        if not np.iscomplexobj(iq_data):
            iq_complex = iq_data[:, 0] + 1j * iq_data[:, 1]

        # - 带通滤波
        pass

        # - 1. 下变频
        # 找到  top-k 频率分量根据幅度作为概率随机采样载波频率
        freqs, mags = compute_topk_freqs(iq_complex, topk=self.freq_topk)
        # mags_softmax = F.softmax(torch.tensor(mags), dim=0).numpy()
        carrier_freq = np.random.choice(freqs, p=mags / np.sum(mags))

        s = self._downconvert(iq_complex, carrier_freq)
        if self.step <= 1:
            return np.stack([s.real, s.imag], axis=1)

        # - 2. 基带低通滤波
        assert symbol_width_absl is not None
        s = self._lowpass_filter(s, 1 / (symbol_width_absl) * self.bandwidth_ratio)

        return np.stack([s.real, s.imag], axis=1)


class EBDSC3rdLoader(Dataset):
    # 16APSK  16QAM  32APSK  32QAM  8APSK  8PSK  8QAM  BPSK  MSK  QPSK  UNKNOWN
    # 注意文件中的 mod_type 从 1 开始
    # 1: BPSK, 2: QPSK, 3: 8PSK, 4: MSK, 5: 8QAM, 6: 16QAM, 7: 32QAM, 8: 8APSK, 9: 16APSK, 10: 32APSK, 11: UNKNOWN
    MOD_TYPE_NUM = 11
    MOD_TYPE = dict(
        zip(
            range(MOD_TYPE_NUM),
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

    def __init__(
        self,
        root_dir,
        *,
        demodulator: Demodulator = None,
        code_map_offset: int = 1,
        mod_uniq_symbol: bool = False,
        data_aug: bool = False,
        is_test: bool = False,
        sample_rate: float = 1.0,
    ):
        """
        Args:
            root_dir (str): 数据集根目录。
            code_map_offset (int): 符号到整数的映射偏移量。
                e.g.
                    - 0 -> <PAD> (填充符号)
                    - 1 -> <UNK> (未知符号)
                    - 2 -> <SOS> (起始符号)
                    - 3 -> <EOS> (终止符号)
            mod_uniq_symbol (bool): 是否对码元符号进行唯一化。
            data_aug (bool): 是否进行数据增强。
            is_test (bool): 是否为测试评估场景，负责为 train val
        """
        self.root_dir = root_dir
        self.demodulator = demodulator
        self.samples = None
        self.code_map_offset = code_map_offset
        self.mod_uniq_symbol = mod_uniq_symbol
        self.data_aug = data_aug
        self.sample_rate = sample_rate

        # 符号类别数
        if mod_uniq_symbol:
            self.num_code_classes = EBDSC3rdLoader.START_OFFSET[-1] + 1
        else:
            self.num_code_classes = 32
        self.num_code_classes = self.num_code_classes + code_map_offset

        # mod 类别数
        self.num_mod_classes = self.MOD_TYPE_NUM

        if is_test:
            self._test_loader()
        else:
            self._train_loader()
        self.is_test = is_test

    def _train_loader(self):
        self.samples = []
        for label in range(11):
            folder_path = os.path.join(self.root_dir, EBDSC3rdLoader.MOD_TYPE[label])
            if os.path.isdir(folder_path):
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
                self.samples.extend(iter(files))

    def _test_loader(self):
        test_file_lst = [name for name in os.listdir(self.root_dir) if name.endswith(".csv")]
        test_file_lst = [os.path.join(self.root_dir, name) for name in test_file_lst]
        self.samples = test_file_lst
        self.symbol_widths = [None] * len(self.samples)

    def test_update_demod(self, demodulator, symbol_widths):
        self.demodulator = demodulator
        self.symbol_widths = symbol_widths
        print("Demodulator updated.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ans = dict()

        file_path = self.samples[idx]
        df = pd.read_csv(file_path, names=EBDSC3rdLoader.DATA_COLUMNS, header=None)

        # - 提取 I/Q 数据
        IQ_data = df[["I", "Q"]].dropna().to_numpy().astype(np.float32)

        if not self.is_test:
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
                symbol_width * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT,
                len(IQ_data),
                mapped_code_sequence,
                EBDSC3rdLoader.PAD,
                self.sample_rate,
            )

            # 数据增强
            if self.data_aug:
                # TODO 截断

                # - 随机镜像
                if np.random.random() < 0.5:
                    IQ_data = np.flip(IQ_data, axis=0).copy()
                    code_sequence_aligned = np.flip(code_sequence_aligned, axis=0).copy()
                    mapped_code_sequence = np.flip(mapped_code_sequence, axis=0).copy()

            # - 带通信号解调
            if self.demodulator is not None:
                IQ_data = self.demodulator.demod(
                    IQ_data, symbol_width * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT
                )  # TODO 此处先验;测试集

            ans = {
                "code_sequence_aligned": torch.tensor(code_sequence_aligned, dtype=torch.long),  # [code_len]
                "mod_type": torch.tensor(mod_type, dtype=torch.long),  # scalar
                "symbol_width": torch.tensor(symbol_width, dtype=torch.float32),  # scalar
                "code_sequence": torch.tensor(mapped_code_sequence, dtype=torch.long),  # [code_len]
            }
        else:   # is_test
            if self.demodulator is not None:
                if self.demodulator.step == 1:
                    IQ_data = self.demodulator.demod(IQ_data, None)
                elif self.demodulator.step == 2:
                    IQ_data = self.demodulator.demod(
                        IQ_data, self.symbol_widths[idx] * EBDSC3rdLoader.SYMBOL_WIDTH_UNIT
                    )
                    # ans = {
                    #     "symbol_width_pre": torch.tensor(self.symbol_widths[idx], dtype=torch.float32),  # scalar
                    # }
        # - 标准化
        # IQ_data = (IQ_data - IQ_data.mean(axis=0)) / IQ_data.std(axis=0)

        return {
            "idx": idx,
            "filename": os.path.basename(file_path),  # str 文件名，用于测试评估
            "IQ_data": torch.from_numpy(IQ_data).float(),  # [IQ_len, 2]
            "IQ_length": torch.tensor(len(IQ_data), dtype=torch.long),  # scalar 备用
            **ans,
        }


def unique_symbol(code_sequence: np.ndarray, mod_type: int):
    """
    码元符号根据 mod_type 唯一化
    modtpye: 0 ~ 10
    """
    start = EBDSC3rdLoader.START_OFFSET[mod_type]
    return code_sequence + start


def _repeat_and_pad_sequence_batch(
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


def un_unique_symbol(code_sequence, mod_type):
    """
    码元符号根据 mod_type 反唯一化
    modtpye: 0 ~ 10
    """
    start = EBDSC3rdLoader.START_OFFSET[mod_type]
    return code_sequence - start


# TODO 根据 ratio 只保留中心的采样（请不要均匀分布，而是优先采样靠近中间的码元），但至少采样一次，额外返回 mask 提供掩码的函数
def _sample_masking(symbol_width_absl: float, length: int, pad: int = 0, sample_rate: float = 1.0):
    """根据 ratio 采样码序列 (中心优先的采样)，并返回掩码
    e.g.
        |<------->| = symbol_width_absl
        [1, 1, 1, 2, 2, 2, 3... 1] ->
        [0, 1, 0, 0, 1, 0,  ...  ] (mask based on ratio e.g. 1/3)
        |<---------------------->| = length == len(IQ_data)

    Args:
        symbol_width_absl (float): 码元宽度
        length (int): IQ 数据长度
        pad (int): 填充值
        ratio (float): 采样比例
    Returns:
        np.ndarray: 采样后的码序列
        np.ndarray: 掩码
    """
    repeat_count = int(symbol_width_absl)
    # 生成采样掩码
    mask = np.ones(length, dtype=np.int64) * pad
    mask[repeat_count // 2 :: repeat_count] = 1
    return mask


def sample_masking(symbol_width_absl: float, length: int, pad: int = 0, sample_rate: float = 1.0):
    """
    根据 ratio 采样码序列 (中心优先的采样)，并返回掩码。
    根据比例 ratio 选择要采样的码元区域，并确保采样从序列中心开始向外扩展。
    保证至少采样一次中心点。

    e.g.
        |<------->| = symbol_width_absl
        [1, 1, 1, 2, 2, 2, 3... 1] ->
        [0, 1, 0, 0, 1, 0,  ...  ] (mask based on ratio e.g. 1/3)
        |<---------------------->| = length == len(IQ_data)

    Args:
        symbol_width_absl (float): 码元宽度
        length (int): IQ 数据长度
        pad (int): 填充值
        ratio (float): 采样比例（0 到 1 之间，表示采样比例）

    Returns:
        np.ndarray: 采样后的码序列掩码
        np.ndarray: 掩码
    """
    if sample_rate == 1.0:
        return np.ones(length, dtype=np.int64)

    # 计算每个码元内的采样数量
    num_samples = max(1, int(symbol_width_absl * sample_rate))  # 每个码元的采样点数，确保至少一个
    repeat_count = int(symbol_width_absl)  # 每个码元的宽度（取整）

    # 计算码元的总数
    num_symbols = length // repeat_count

    # 初始化掩码，默认填充值
    mask = np.full(length, pad, dtype=np.int64)

    # 计算偏移量范围
    half_samples = num_samples // 2  # 采样点偏移的半宽度

    # 批量计算所有采样点
    sample_indices = []
    for symbol_idx in range(num_symbols):
        center_idx = symbol_idx * repeat_count + repeat_count // 2  # 中心点索引
        # 计算当前码元的采样点索引范围
        start_idx = max(0, center_idx - half_samples)  # 确保不越界
        end_idx = min(length, center_idx + half_samples + 1)  # 确保不越界

        # 生成当前码元的采样点
        sample_indices.extend(range(start_idx, end_idx))

    # 使用 NumPy 设置采样点为 1（避免逐一访问）
    mask[np.array(sample_indices)] = 1

    return mask


def repeat_and_pad_sequence(
    symbol_width_absl: float, length: int, code_sequence: np.ndarray, pad: int = 0, sample_rate: float = 1.0
):
    """code_sequence 拓展为 IQ 数据长度 len(mapped_code_sequence) = len(IQ_data)
    e.g.
        |<------->| = symbol_width_absl
        [1,       2,       3... 1] ->
        [1, 1, 1, 2, 2, 2, 3... 1]
        |<---------------------->| = length == len(IQ_data)
        [0, 1, 0, 0, 1, 0,  ...  ] (mask based on ratio e.g. 1/3)
    计算每个码元需要重复的次数

    TODO 根据 ratio 只保留中心的采样，但至少采样一次，额外返回 mask 提供掩码

    Args:
        symbol_width_absl (float): 码元宽度。
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

    return s * sample_masking(symbol_width_absl, length, pad, sample_rate)


def _reverse_sequence(symbol_width_absl: float, expanded_sequence: np.ndarray, pad: int = 0, sample_rate: float = 1.0):
    """从展开的序列恢复原始码序列
    e.g.
        [?, 1, ?, ?, 2, ?,  ...  ] (ratio <= 1/3)
        [1, 1, 1, 2, 2, 2, 3... 1] (ratio == 1.) ->
        [1,       2,       3... 1]
        |<------->| = symbol_width_absl

    TODO 根据 ratio 只根据中心的采样恢复

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
    # TODO 通过查找最后一个非 PAD 值的位置，需要确定下正确性，因为 `?` 可能是 PAD
    last_non_pad = np.where(original_sequence != pad)[0][-1] if len(original_sequence) > 0 else -1
    return original_sequence[: last_non_pad + 1] if last_non_pad >= 0 else np.array([])


def _reverse_sequence_from_logits(symbol_width_absl: float, expanded_logits: np.ndarray, pad: int = 0):
    """从展开的 logits 序列恢复原始码序列，通过选择窗口内累计概率最大的类别

    TODO 根据 ratio 只根据中心的采样恢复

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
    symbol_width_absl: torch.Tensor, expanded_logits: torch.Tensor, pad: int = 0, sample_rate: float = 1.0
) -> torch.LongTensor:
    """从展开的 logits 序列恢复原始码序列，支持批处理

    TODO 根据 ratio 只根据中心的采样恢复

    先验 set(symbol_widths_absl)：
    {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

    Args:
        symbol_width_absl (Tensor): [batch_size] 每个样本的码元宽度
        expanded_logits (Tensor): [batch_size, seq_len, num_classes] logits 序列
        pad (int): 填充值

    Returns:
        Tensor: [batch_size, orig_len] 原始码序列
    """
    # - Softmax 转换为概率
    expanded_probs = F.softmax(expanded_logits, dim=-1)
    batch_size, seq_len, num_classes = expanded_probs.shape

    # - 根据先验选择最近的宽度值
    allowed_widths = torch.tensor(
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], device=expanded_logits.device
    )
    # 计算与每个允许值的距离
    distances = torch.abs(symbol_width_absl.unsqueeze(1) - allowed_widths)
    # 选择最近的允许值
    repeat_counts = allowed_widths[torch.argmin(distances, dim=1)]

    # - 计算每个样本的原始序列长度
    # TODO seq_len 这边依据的 IQ 实际上被填充了，所以这边的长度是填充后的长度
    orig_lens = torch.div(seq_len, repeat_counts, rounding_mode="floor")
    max_orig_len = orig_lens.max().item()

    # 初始化输出
    # TODO 注意这边的填充了 0
    original_sequences = torch.full((batch_size, max_orig_len), pad, device=expanded_probs.device, dtype=torch.long)

    # 对每个样本处理
    for b in range(batch_size):
        # sample_masking
        mask = sample_masking(repeat_counts[b], seq_len, pad, sample_rate)
        expanded_prob = expanded_probs[b] * torch.tensor(mask, device=expanded_probs.device).unsqueeze(-1)
        # 使用 unfold 切分窗口
        windows = expanded_prob.unfold(
            0, repeat_counts[b], repeat_counts[b]
        )  # [num_windows, repeat_count, num_classes]

        # 在窗口内求和并取最大值索引
        summed_logits = windows.sum(dim=-1)  # [orig_len, num_classes]
        pred_codes = summed_logits.argmax(dim=-1)  # [orig_len]

        # 填充结果
        orig_len = orig_lens[b]
        original_sequences[b, :orig_len] = pred_codes[:orig_len]

    return original_sequences


def make_collate_fn(*, data_aug: bool = False, code_map_offset: int = 1, pad_idx: int = 0, is_test: bool = False):
    def collate_fn(batch):
        """
        自定义的 collate 函数，用于处理可变长度的 I/Q 数据和 code_sequence。
        TODO 限制 IQ 与 code_sequence 的长度 | 并加入数据增强
        Args:
            batch (list): 包含多个样本，每个样本是一个字典。
        Returns:
            dict: 包含批处理后的 I/Q 数据、code_sequence、掩码、mod_type 和 symbol_width。
        """
        # 提取各个字段 - test
        filenames = [item["filename"] for item in batch]
        IQ_data_list = [item["IQ_data"] for item in batch]
        IQ_length_list = [item["IQ_length"] for item in batch]

        # 填充 I/Q 数据
        IQ_padded = torch.nn.utils.rnn.pad_sequence(
            IQ_data_list, batch_first=True, padding_value=pad_idx
        )  # [batch_size, max_IQ_len, 2]
        IQ_lengths = torch.stack(IQ_length_list)  # [batch_size]
        if is_test:
            return {
                "idx": [item["idx"] for item in batch],
                "filename": filenames,
                "IQ_data": IQ_padded,  # [batch_size, max_IQ_len, 2]
                "IQ_length": IQ_lengths,  # [batch_size]
            }

        # 提取各个字段 - train
        code_seq_list = [item["code_sequence"] for item in batch]
        code_seq_aligned_list = [item["code_sequence_aligned"] for item in batch]
        mod_type_list = [item["mod_type"] for item in batch]
        symbol_width_list = [item["symbol_width"] for item in batch]

        # 填充 code_sequence_aligned
        code_seq_aligned_list = torch.nn.utils.rnn.pad_sequence(
            code_seq_aligned_list, batch_first=True, padding_value=pad_idx
        )  # [batch_size, max_code_len]

        # 填充 code_sequence
        code_padded = torch.nn.utils.rnn.pad_sequence(
            code_seq_list, batch_first=True, padding_value=pad_idx
        )  # [batch_size, max_code_len]

        # 创建 code_sequence 的掩码（1 表示有效，0 表示填充）
        code_mask = (code_seq_aligned_list != pad_idx).long()  # [batch_size, max_code_len]

        return {
            "idx": [item["idx"] for item in batch],
            "IQ_data": IQ_padded,  # [batch_size, max_IQ_len, 2]
            "IQ_length": IQ_lengths,  # [batch_size]
            "code_sequence_aligned": code_seq_aligned_list,  # [batch_size, max_code_len]
            "code_mask": code_mask,  # [batch_size, max_code_len]
            "mod_type": torch.stack(mod_type_list),  # [batch_size]
            "symbol_width": torch.stack(symbol_width_list),  # [batch_size]
            "code_sequence": code_padded,  # [batch_size, code_len]
        }

    return collate_fn


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
    code_seq_preds: torch.Tensor,
    code_seq_labels: torch.Tensor,
    pad_idx: int = 0,
    code_map_offset: int = 1,
    mod_uniq_symbol: tuple = (False, None, None),
):
    """
    计算码序列解调余弦相似度得分 (CQ_score)。
    NOTE 计算时需减去 code_map_offset
    TODO 这个函数的逻辑有点繁杂，需要重构：
        1. 根据向量中的向量中的零元素 不会对余弦相似度产生影响，去除截断的操作
        2. 不能根据 max_true_length 掩码，而是根据每个样本的实际的长度进行掩码
        3. 解决以下 bug：
        File "/ebdsc3rd_datatools.py", line 660, in compute_CQ_score
            mask = mask_1 & mask_2 & mask_3
                ~~~~~~~~~~~~~~~~^~~~~~~~
        RuntimeError: The size of tensor a (396) must match the size of tensor b (330) at non-singleton dimension 1

    Args:
        code_seq_preds (Tensor): [batch_size, tgt_seq_len] 预测的码序列
        code_seq_labels (Tensor): [batch_size, tgt_seq_len] 真实的码序列
        pad_idx (int): 填充符号的索引
        code_map_offset (int): 码映射偏移量
        mod_uniq_symbol (tuple, optional):
            A tuple indicating whether to modify unique symbols and the related parameters.
            Expected format: (bool, preds_mod_tensor, labels_mod_tensor).
            Defaults to (False, None, None).
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
        code_seq_preds = torch.where(
            code_seq_preds != 0, code_seq_preds - offset[mod_preds].unsqueeze(1), code_seq_preds
        )
        code_seq_preds = code_seq_preds.clamp(min=0 + code_map_offset)  # TODO

        mod_labels = mod_uniq_symbol[2]
        code_seq_labels = torch.where(
            code_seq_labels != 0, code_seq_labels - offset[mod_labels].unsqueeze(1), code_seq_labels
        )
        assert torch.sum(code_seq_labels < 0) == 0
        assert torch.sum(code_seq_labels > 32) == 0

    # 计算每个样本的真实长度（非 pad 的长度）
    true_lengths = (code_seq_labels != pad_idx).sum(dim=1)  # [batch_size]

    # 确定每个样本的最大真实长度，以便截断预测序列
    # TODO 这实际上只是对最大长度截取。向量填充 0 是不是不影响余弦相似度计算？
    # - 还是得针对每一个样本截取，通过 mask 实现
    max_true_length = true_lengths.max().item()

    # 如果预测序列长度超过最大真实长度，进行截断；否则，进行填充
    if pred_seq_len > max_true_length:
        # 截断预测序列
        pred_seq_truncated = code_seq_preds[:, :max_true_length]
    else:
        # 填充预测序列，使其长度与最大真实长度一致
        padding_size = max_true_length - pred_seq_len

        padding = torch.full(
            (batch_size_pred, padding_size),
            pad_idx,  # NOTE 此处 pad_idx 而非 code_map_offset 填充是为了强调
            dtype=code_seq_preds.dtype,
            device=device,
        )
        pred_seq_truncated = torch.cat([code_seq_preds, padding], dim=1)

    # 创建掩码，表示每个位置是否在真实长度内
    mask_1 = torch.arange(max_true_length, device=device).unsqueeze(0).expand(
        batch_size, max_true_length
    ) < true_lengths.unsqueeze(
        1
    )  # [batch_size, max_true_length]

    mask_2 = (code_seq_labels != pad_idx)[:, :max_true_length]  # [batch_size, max_true_length]
    mask_3 = (code_seq_preds != pad_idx)[:, :max_true_length]  # [batch_size, max_true_length]
    mask = mask_1 & mask_2 & mask_3

    # 将序列转换为向量，并减去 code_map_offset
    true_vec = (
        code_seq_labels[:, :max_true_length]
        - code_map_offset  # TODO 原本作为填充的 0 会被减去 code_map_offset = -1 不过被 mask 掉了
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

    return CQ_score, cosine_similarity


def compute_CQ_score(
    code_seq_preds: torch.Tensor,
    code_seq_labels: torch.Tensor,
    pad_idx: int = 0,
    code_map_offset: int = 1,
    uniq_symbol_args: dict = {
        "enable": False,
        "mod_preds": None,
        "mod_labels": None,
    },
):
    """
    计算码序列解调余弦相似度得分 (CQ_score)。

    Args:
        code_seq_preds (Tensor): [batch_size, pred_seq_len] 预测的码序列。
        code_seq_labels (Tensor): [batch_size, label_seq_len] 真实的码序列。
        pad_idx (int): 填充符号的索引。
        code_map_offset (int): 码映射偏移量。
        mod_uniq_symbol (dict, optional): 是否对码元符号进行唯一化。默认为 False。
            - enable (bool): 是否启用唯一化。
            - mod_preds (Tensor): [batch_size] 预测的 mod 类别。
            - mod_labels (Tensor): [batch_size] 真实的 mod 类别。

    Returns:
        Tuple[Tensor, Tensor]:
            - CQ_score: [batch_size] 每个样本的 CQ_score。
            - cosine_similarity: [batch_size] 每个样本的余弦相似度。
    """
    device = code_seq_preds.device
    batch_size_pred, pred_seq_len = code_seq_preds.size()
    batch_size_label, label_seq_len = code_seq_labels.size()

    # 如果预测序列和标签序列的长度不一致，进行填充以匹配长度
    if pred_seq_len > label_seq_len:
        padding_size = pred_seq_len - label_seq_len
        padding = torch.full(
            (batch_size_label, padding_size),
            pad_idx,
            dtype=code_seq_labels.dtype,
            device=device,
        )
        code_seq_labels = torch.cat([code_seq_labels, padding], dim=1)
    elif label_seq_len > pred_seq_len:
        padding_size = label_seq_len - pred_seq_len
        padding = torch.full(
            (batch_size_pred, padding_size),
            pad_idx,
            dtype=code_seq_preds.dtype,
            device=device,
        )
        code_seq_preds = torch.cat([code_seq_preds, padding], dim=1)

    # 处理唯一符号修改（如果需要）
    if uniq_symbol_args["enable"]:
        offset = torch.tensor(EBDSC3rdLoader.START_OFFSET, device=device, dtype=torch.long)

        # 修改预测序列
        mod_preds = uniq_symbol_args["mod_preds"]
        c = code_seq_preds
        # 先限制在 [offset[mod_preds], offset[mod_preds+1]) 范围内
        c = c.clamp(
            min=offset[mod_preds].unsqueeze(1) + code_map_offset,
            max=offset[mod_preds + 1].unsqueeze(1) - 1 + code_map_offset,
        )
        # 去除偏置
        c = c - offset[mod_preds].unsqueeze(1)
        code_seq_preds = torch.where(code_seq_preds != pad_idx, c, code_seq_preds)
        # assert torch.sum(code_seq_preds < 0) == 0, "预测在修改后存在负值。"

        # 修改标签序列
        mod_labels = uniq_symbol_args["mod_labels"]
        code_seq_labels = torch.where(
            code_seq_labels != pad_idx, code_seq_labels - offset[mod_labels].unsqueeze(1), code_seq_labels
        )
        assert torch.sum(code_seq_labels < 0) == 0, "标签在修改后存在负值。"
        assert torch.sum(code_seq_labels > 32) == 0, "标签在修改后超过预期最大值。"

    # 创建掩码，表示每个位置是否在真实长度内且预测和标签不为 pad
    mask_labels = code_seq_labels != pad_idx  # [batch_size, max_seq_len]
    mask_preds = code_seq_preds != pad_idx  # [batch_size, max_seq_len]
    combined_mask = mask_labels & mask_preds  # [batch_size, max_seq_len]

    # 将序列转换为向量，并减去 code_map_offset
    true_vec = (code_seq_labels - code_map_offset).float() * combined_mask.float()  # [batch_size, max_seq_len]
    pred_vec = (code_seq_preds - code_map_offset).float() * combined_mask.float()  # [batch_size, max_seq_len]

    # acc
    f1 = code_seq_labels == code_seq_preds
    f2 = code_seq_labels != pad_idx
    acc = (f1 & f2).sum(dim=1).float() / f2.sum(dim=1).float()

    # 计算余弦相似度
    dot_product = (true_vec * pred_vec).sum(dim=1)  # [batch_size]
    norm_true = true_vec.norm(p=2, dim=1)  # [batch_size]
    norm_pred = pred_vec.norm(p=2, dim=1)  # [batch_size]

    # 处理范数为零的情况，避免除以零
    cosine_similarity = torch.where(
        (norm_true == 0) | (norm_pred == 0), torch.zeros_like(dot_product), dot_product / (norm_true * norm_pred)
    )

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

    return CQ_score, cosine_similarity, acc


# 示例用法
if __name__ == "__main__":
    # 示例数据
    code_seq_preds = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7],  # 预测序列长度 6
            [1, 2, 3, 4, 0, 0],  # 预测序列长度 4
            [3, 3, 3, 3, 3, 3],  # 预测序列长度 6
            [0, 0, 0, 0, 0, 0],  # 预测序列全是 pad
        ]
    )

    code_seq_labels = torch.tensor(
        [
            [2, 3, 4, 5, 0, 0],  # 真实序列长度 4
            [1, 2, 3, 4, 5, 0],  # 真实序列长度 5
            [3, 3, 3, 3, 3, 3],  # 真实序列长度 6
            [0, 0, 0, 0, 0, 0],  # 真实序列全是 pad
        ]
    )

    pad_idx = 0
    code_map_offset = 1

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
    cq_scores, cs, acc = compute_CQ_score(code_seq_preds, code_seq_labels, pad_idx, code_map_offset)
    print(cq_scores)
    print(cs)
    print(acc)

    # - sample_masking 示例使用
    symbol_width_absl = 7  # 例如每个码元宽度为 6
    length = 60  # 数据长度
    pad = 0  # 填充值
    ratio = 0.5  # 采样比例

    mask = sample_masking(symbol_width_absl, length, pad, ratio)
    print(mask)
