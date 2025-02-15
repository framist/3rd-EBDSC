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


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

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
        # - freTS
        x = self.freTS_block(x)

        # - modern TCN
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M*D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B*M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

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
    def __init__(self, d_model, max_len=5000):
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
        if self.emb_type >= 1:
            self.value_embedding = nn.Sequential(
                nn.Conv1d(1, D, kernel_size=1, bias=False),
                nn.BatchNorm1d(D)
            )
        if self.emb_type == 2:
            self.position_embedding = PositionalEmbedding(d_model=D)

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

    def forward(self, x: torch.Tensor):
        # L = N = 1024 序列长 (P=1, S=1 时)
        # B = batch size
        
        if self.emb_type >= 1:
            # x: [B, L=1024, M=5] -> [B, M=5, L]
            B = x.shape[0]
            x = rearrange(x, 'b l m -> b m l')
            x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
            x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
            x_emb = self.value_embedding(x)
            
            if self.emb_type == 2:
                x_pos = self.position_embedding(x).transpose(1, 2)
                x_emb = x_pos + x_emb
            x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]
        else:            
            # x: [B, L=1024, M=5, pos_D=128] -> [B, M=5, D=128, L=1024]
            x_emb = rearrange(x, 'b l m d -> b m d l')
        
        x_emb = self.stages(x_emb)

        # 在展平之前，[64, 5, 64, 1024] 要做序列标注任务 则 [64,5,1024,12] 将 5 个特征维度聚合得到 [64,1024,12]
        # 本质是 [B, M, D, N] -> [B, L, classes],其中 L 为 1024，classes 为 12，且 N = L // S
        # 可以考虑使用更复杂的池化方式、添加 dropout 等来增强模型的表达能力。

        # Flatten 将预测的长度拉开，把嵌入的维度拉开
        # [B, M, D, N] -> [B, M*D, N]
        cls1 = rearrange(x_emb, 'b m d n -> b (m d) n')

        # maxpool
        # cls1 = torch.max(x_emb, dim=1)[0]    # [B, M, D, N] -> [B, D, N]

        # 转换为 [64, 1024, 64]
        encoder_output = cls1.permute(0, 2, 1)

        code_seq_logits = self.sortinghead(encoder_output) # [batch_size, seq_len, 1, num_classes]

        # 全局特征用于分类和回归
        # TODO mean 池化待验证
        global_feat = encoder_output # .mean(dim=1)  # [batch_size, d_model]

        # 调制类型分类
        mod_logits = self.mod_classifier(global_feat)  # [batch_size, num_mod_classes]

        # 码元宽度回归
        symbol_width = self.symbol_width_regressor(global_feat).squeeze(-1)  # [batch_size]
        
        return mod_logits, symbol_width, code_seq_logits

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

if __name__ == '__main__':
    from time import time

    past_series = torch.rand(10, 1024, 2).cuda()
    # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    model = ModernTCN_MutiTask(M=2, num_code_classes=32, num_mod_classes=12, stem=2).cuda()
    
    pred_series = model(past_series)

    # model.structural_reparam()

    # start = time()
    # pred_series = model(past_series)
    # end = time()

    # print(pred_series.shape, f"time {end - start}")


    # past_series2 = torch.rand(10, 1024, 5).cuda()
    # # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    # model = ModernTCN_MutiTask(5, 12, stem=True).cuda()

    # start = time()
    # pred_series = model(past_series2)
    # end = time()
    # print(pred_series.shape, f"time {end - start}")

    # model.structural_reparam()

    # start = time()
    # pred_series = model(past_series2)
    # end = time()

    # print(pred_series.shape, f"time {end - start}")
