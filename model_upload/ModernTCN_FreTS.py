import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from models.FreTS import FreTS_Block

# Attention Pooling
class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        scores = self.attention(x)  # [batch_size, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1] 
        global_feat = torch.sum(x * weights, dim=1)  # [batch_size, d_model]
        return global_feat

# CNN + Pooling
class CNNPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.conv(x)
        global_feat = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # [batch_size, d_model]
        return global_feat

class MeanPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class FrequencyOffsetEstimator(nn.Module):
    """时变相位生成器"""

    def __init__(self, embed_size) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.pool = AttentionPool(embed_size * 2)
        self.regressor = nn.Sequential(
            nn.Linear(embed_size * 2, 1),  # 输出Δf 估计值
        )

    def forward(self, x):
        T = x.size(-1)
        x = rearrange(x, "b m d t -> b t (m d)")
        x = self.pool(x)  # [B, 2*D]
        delta_f = self.regressor(x)  # [B, 1]
        return delta_f * torch.arange(T).to(x.device) / T  # 生成线性相位


class ComplexPhaseCorrector(nn.Module):
    """复数域相位校正层"""
    def __init__(self):
        super().__init__()
        self.fc_real = nn.Linear(1, 1, bias=False)
        self.fc_imag = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, D, N]
        x = rearrange(x, "b m d t -> b d t m")  # [B, D, T, 2] (实部 + 虚部)
        # phi: [B,T]
        # cos_phi = torch.cos(phi).unsqueeze(-1)  # [B,T,1]
        cos_phi = rearrange(torch.cos(phi), "b t -> b 1 t 1")  # [B, 1, T, 1]
        sin_phi = rearrange(torch.sin(phi), "b t -> b 1 t 1")
        x_real = x[..., 0:1] * self.fc_real(cos_phi) - x[..., 1:2] * self.fc_imag(sin_phi)
        x_imag = x[..., 0:1] * self.fc_imag(sin_phi) + x[..., 1:2] * self.fc_real(cos_phi)
        x = torch.cat([x_real, x_imag], dim=-1)
        x = rearrange(x, "b d t m -> b m d t")
        return x


class PhaseCorrectorBlock(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.frequency_offset_estimator = FrequencyOffsetEstimator(embed_size)
        self.complex_phase_corrector = ComplexPhaseCorrector()

    def forward(self, x):
        delta_f = self.frequency_offset_estimator(x)
        x_corrected = self.complex_phase_corrector(x, delta_f)
        return x_corrected + x


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left).cuda()
            pad_right = torch.zeros(D_out, D_in, pad_length_right).cuda()
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values

        x = torch.cat([pad_left, x], dim=-1)
        x = torch.cat([x, pad_right], dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(
                self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
        
        self.freTS_block = FreTS_Block(embed_size=dmodel, hidden_size=dff)
        
        
    def forward(self, x):
        
        # freTS 块位置备选 1 - default
        x = x + self.freTS_block(x)

        # = modern TCN
        input = x
        B, M, D, N = x.shape
        
        # freTS 块位置备选 2
        # x = self.freTS_block(x)
        
        # - DWConv
        x = x.reshape(B, M*D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        
        # - BN
        x = x.reshape(B*M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        
        # freTS 块位置备选 4        
        # x = self.freTS_block(x)
        
        # - ConvFFN1 Groups: M
        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        # - ConvFFN2 Groups: D
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        # - Residual
        x = input + x        
        
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars,
                 small_kernel_merged=False, drop=0.):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for _ in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel,
                        dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


import math
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(-1)]

class myEmbedding(nn.Module):
    """
    模拟以下操作的固定嵌入层:
    x: [B, L, M=2] -> emb [B, L, M=2, D]
    M = 2 存储 IQ 分量
    在嵌入维度 (D) 生产 `carrier_freq = torch.linspace(-0.1, 0.1, d_model)` 的频移信号
    即在 D 的每一个维度:
    iq_complex = iq_filtered[:, 0] + 1j * iq_filtered[:, 1]
    sampling_rate = 1
    t = np.arange(len(iq_complex)) / sampling_rate
    local_carrier = np.exp(-2j * np.pi * carrier_freq * t)
    demodulated_signal = iq_complex * local_carrier
    """
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        carrier_freq = torch.linspace(-0.1, 0.1, d_model).view(1, 1, d_model)
        t = torch.arange(max_len, dtype=torch.float).view(max_len, 1, 1)
        carrier_real = torch.cos(-2 * math.pi * carrier_freq * t)
        carrier_imag = torch.sin(-2 * math.pi * carrier_freq * t)
        carrier = torch.stack([carrier_real, carrier_imag], dim=2)  # [max_len, 1, 2, d_model]
        self.register_buffer('carrier', carrier.permute(1, 0, 2, 3))  # [1, max_len, 2, d_model]

    def forward(self, x):
        # x: [B, L, 2]
        L = x.size(1)
        c = self.carrier[:, :L]  # [1, L, 2, d_model]
        x_real = x[..., 0].unsqueeze(-1)
        x_imag = x[..., 1].unsqueeze(-1)
        c_real = c[..., 0, :]
        c_imag = c[..., 1, :]
        emb_real = x_real * c_real - x_imag * c_imag
        emb_imag = x_real * c_imag + x_imag * c_real
        emb = torch.stack([emb_real, emb_imag], dim=2)  # [B, L, 2, d_model]
        return emb
    

class ModernTCN_MutiTask(nn.Module):  # T 在预测任务当中为预测的长度，可以更换为输出的种类 num_classes
    def __init__(self, *, M, num_code_classes, num_mod_classes, D=128, large_sizes=51, ffn_ratio=2, num_layers=24, 
                 small_size=5, small_kernel_merged=False, backbone_dropout=0., head_dropout=0., stem=1, mean_pool=False
                 ):  # 如果能收敛就一点一点增加，在原来跑通的里面层数为
        # M, L, num_classes,
        super(ModernTCN_MutiTask, self).__init__()
        self.num_layers = num_layers

        # stem layer
        # emb_type 0: fixed, 1: learnable, 2: learnable + pos
        self.emb_type = stem
        if self.emb_type == 1:
            self.value_embedding = nn.Sequential(
                nn.Conv1d(1, D, kernel_size=1, bias=True),
                nn.BatchNorm1d(D)
            )
        elif self.emb_type == 2:
            self.value_embedding = nn.Sequential(
                nn.Conv1d(1, D, kernel_size=1, bias=False),
                nn.BatchNorm1d(D)
            )
            self.position_embedding = PositionalEmbedding(d_model=D)
        elif self.emb_type == 3:
            self.value_embedding = myEmbedding(D)
            
        # backbone
        self.stages = Stage(ffn_ratio, num_layers, large_size=large_sizes, small_size=small_size, dmodel=D,
                            nvars=M, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)

        # w/o pool
        # self.classificationhead = nn.Linear(D * M, num_classes)
        self.sortinghead = nn.Sequential(
            nn.Linear(D * M, D * M * 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(D * M * 2, num_code_classes)
        )
        
        # 分类头：调制类型
        self.mod_classifier = nn.Sequential(
            AttentionPool(D * M) if not mean_pool else MeanPool(dim=1),
            nn.Linear(D * M, D * M),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(D * M, num_mod_classes)
        )

        # 回归头：码元宽度
        self.symbol_width_regressor = nn.Sequential(
            AttentionPool(D * M) if not mean_pool else MeanPool(dim=1),
            nn.Linear(D * M, D * M),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(D * M, 1),
        )

        # # with pool
        # self.classificationhead = nn.Linear(D, num_classes)
        
        # - 相位校正
        # self.phase_corrector = PhaseCorrectorBlock(embed_size=D)
        
    def forward(self, x: torch.Tensor):
        # x: [B, L, M=2]
        # L = N = 1024 序列长 (P=1, S=1 时)
        # B = batch size
        # M = 2 IQ 分量
        if self.emb_type == 3:
            x_emb = self.value_embedding(x) # [B, L, M=2, D]
            x_emb = rearrange(x_emb, 'b l m d -> b m d l')  # [B, L, M=2, D] -> [B, M=2, D, L]
            
        elif self.emb_type >= 1:
            # x: [B, L, M=2] -> [B, M=2, L]
            B = x.shape[0]
            x = rearrange(x, 'b l m -> b m l')
            x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
            x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
            x_emb = self.value_embedding(x)
            
            if self.emb_type == 2:
                x_pos = self.position_embedding(x).transpose(1, 2)
                x_emb = x_pos * x_emb
            x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]
        else:            
            # x: [B, L, M=2, pos_D=128] -> [B, M=2, D=128, L]
            x_emb = rearrange(x, 'b l m d -> b m d l')
        
        x_emb = self.stages(x_emb)
        # x_emb = self.phase_corrector(x_emb)  # [batch_size, seq_len, M, num_classes]
        
        # Flatten 将预测的长度拉开，把嵌入的维度拉开
        # [B, M, D, N] -> [B, M*D, N]
        cls1 = rearrange(x_emb, 'b m d n -> b (m d) n')

        # maxpool
        # cls1 = torch.max(x_emb, dim=1)[0]    # [B, M, D, N] -> [B, D, N]

        encoder_output = cls1.permute(0, 2, 1)

        code_seq_logits = self.sortinghead(encoder_output) # [batch_size, seq_len, 1, num_classes]

        # 全局特征用于分类和回归
        # TODO mean 池化待验证
        global_feat = encoder_output # .mean(dim=1)  # [batch_size, d_model]

        # = 调制类型分类
        mod_logits = self.mod_classifier(global_feat)  # [batch_size, num_mod_classes]

        # = 码元宽度回归
        symbol_width = self.symbol_width_regressor(global_feat).squeeze(-1)  # [batch_size]
        
        return mod_logits, symbol_width, code_seq_logits

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

if __name__ == '__main__':
    from time import time

    past_series = torch.rand(1, 1024, 2).cuda()
    model = ModernTCN_MutiTask(M=2, num_code_classes=32, num_mod_classes=12, stem=3).cuda()
    
    pred_series = model(past_series)

    model.structural_reparam()

    start = time()
    pred_series = model(past_series)
    end = time()

    print(f"time {end - start}")


