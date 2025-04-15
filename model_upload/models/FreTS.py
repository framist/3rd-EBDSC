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
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.hidden_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.hidden_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))


    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=L, dim=2, norm="ortho")
        return x
    
    # frequency temporal learner 复数输入
    def MLP_temporal_complex(self, x, B, L):
        # [B, T, D]
        x = torch.fft.fft(x, dim=1, norm='ortho')  # FFT on L dimension
        y = self.FreMLP_complex1(B, L, x, self.r1, self.i1, self.rb1, self.ib1)
        y = self.FreMLP_complex2(B, L, y, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.ifft(y, n=L, dim=1, norm="ortho")
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

    def FreMLP_complex1(self, B, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, dimension, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, dimension, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bid,dh->bih', x.real, r) - \
            torch.einsum('bid,dh->bih', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bid,dh->bih', x.imag, r) + \
            torch.einsum('bid,dh->bih', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def FreMLP_complex2(self, B, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, dimension, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, dimension, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bih,hd->bid', x.real, r) - \
            torch.einsum('bih,hd->bid', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bih,hd->bid', x.imag, r) + \
            torch.einsum('bih,hd->bid', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
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
        
        # bias = x
        # [B, N, T, D]
        x = x.to(torch.float32)
        # x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        
        # x = x + bias
        return x
    

    def block_complex(self, x: torch.FloatTensor):        
        # embedding x: [B, N=2, T, D]
        B, N, T, D = x.shape
        # 转为复数 N=2 as complex number's real and imaginary part
        x = x.permute(0, 2, 3, 1)
        x = torch.complex(x[..., 0], x[..., 1])
        
        # amp 时 T 不是 2 的幂次方时，cuFFT 会报错，需要 x.to(torch.complex64)
        # cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
        # x = x.to(torch.complex64)
        
        x = self.MLP_temporal_complex(x, B, T)
        
        # 转为实数
        x = torch.stack([x.real, x.imag], dim=-1)
        x = x.permute(0, 3, 1, 2)
        return x
    
    def forward(self, x):
        # x [Batch, Channel, D, Input length] ->
        # x [Batch, Channel, Input length, D]
        x = x.permute(0, 1, 3, 2)
        x = self.block_complex(x)
        x = x.permute(0, 1, 3, 2) # [Batch, Channel, D, Input length]
        return x
