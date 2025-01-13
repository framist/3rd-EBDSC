import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from einops import rearrange

class Configs():
    def __init__(self):
        self.task_name = "muti_tasks"
        self.pred_len = None
        # self.seq_len = 1024
        self.output_attention = False
        self.enc_in = 2
        self.d_model = 128 * 2
        self.embed = 'fixed' # 不用
        self.freq = 'h' # 不用
        # self.use_norm = False
        self.dropout = 0.
        self.n_heads = 2
        self.e_layers = 8
        self.d_ff = self.d_model
        self.activation = 'relu'
        
        self.head_dropout = 0.5
        self.num_code_classes = None
        self.num_mod_classes = None
        
        
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


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs: Configs=None, wide_value_emb = False):
        super(Model, self).__init__()
        if configs is None:
            configs = Configs()
        self.wide_value_emb = wide_value_emb
            
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        if not self.wide_value_emb:
            # Embedding
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        if self.task_name == 'muti_tasks':            
            self.sortinghead = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model * 2),
                nn.GELU(),
                nn.Dropout(configs.head_dropout),
                nn.Linear(configs.d_model * 2, configs.num_code_classes)
            )
            
            # 分类头：调制类型
            self.mod_classifier = nn.Sequential(
                AttentionPool(configs.d_model),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.head_dropout),
                nn.Linear(configs.d_model, configs.num_mod_classes)
            )

            # 回归头：码元宽度
            self.symbol_width_regressor = nn.Sequential(
                AttentionPool(configs.d_model),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.head_dropout),
                nn.Linear(configs.d_model, 1),
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    
    def muti_tasks(self, x_enc):
        """
        TODO Mask
        """
        if self.wide_value_emb:
            # x: [B, L=1024, M=5, pos_D=128] ->
            enc_out = rearrange(x_enc, 'b l m d -> b l (m d)')
        else:
            # Embedding
            enc_out = self.enc_embedding(x_enc, None)
            
        encoder_output, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        code_seq_logits = self.sortinghead(encoder_output) # [batch_size, seq_len, 1, num_classes]

        # 全局特征用于分类和回归
        global_feat = encoder_output # .mean(dim=1)  # [batch_size, d_model]

        # 调制类型分类
        mod_logits = self.mod_classifier(global_feat)  # [batch_size, num_mod_classes]

        # 码元宽度回归
        symbol_width = self.symbol_width_regressor(global_feat).squeeze(-1)  # [batch_size]
        
        return mod_logits, symbol_width, code_seq_logits


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'muti_tasks':
            dec_out = self.muti_tasks(x_enc)
            return dec_out  # [B, L, N]
        return None


if __name__ == '__main__':
    from time import time

    # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    # input = torch.rand(10, 1024, 5, 128).cuda()
    input = torch.rand(10, 1024, 5).cuda()
    model = Model().cuda()
    
    print("模型参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    start = time()
    pred_series = model(input)
    end = time()
    print(pred_series.shape, f"time {end - start}")
