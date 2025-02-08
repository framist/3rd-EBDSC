import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreTS_Block(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """

    def __init__(self, embed_size, hidden_size, sparsity_threshold=0.01, scale=0.02):
        super(FreTS_Block, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))


    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=L, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=N, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y


    # def forecast(self, x_enc):
    #     # x: [Batch, Input length, Channel]
    #     B, T, N = x_enc.shape
    #     # embedding x: [B, N, T, D]
    #     x = self.tokenEmb(x_enc)
    #     bias = x
    #     # [B, N, T, D]
    #     if self.channel_independence == '0':
    #         x = self.MLP_channel(x, B, N, T)
    #     # [B, N, T, D]
    #     x = self.MLP_temporal(x, B, N, T)
    #     x = x + bias
    #     x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
    #     return x

        
    def block(self, x):        
        # embedding x: [B, N, T, D]
        B, N, T, D = x.shape
        
        bias = x
        # [B, N, T, D]
        x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        
        x = x + bias
        return x
    
    def forward(self, x):
        # x [Batch, Channel, D, Input length] ->
        # x [Batch, Channel, Input length, D]
        x = x.permute(0, 1, 3, 2)
        x = self.block(x)
        x = x.permute(0, 1, 3, 2) # [Batch, Channel, D, Input length]
        return x
