import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=2, pad_idx=0, code_map_offset=1):
        """
        基于余弦相似度的损失函数。

        Args:
            num_classes (int): 类别数量。
            embedding_dim (int): 嵌入维度。
            pad_idx (int): 填充符号的索引。
            code_map_offset (int): 码映射偏移量。
            mod_uniq_symbol (tuple, optional):
                一个元组，指示是否修改唯一符号及相关参数。
                格式：(bool, preds_mod_tensor, labels_mod_tensor)。
                默认为 (False, None, None)。
        """
        super(CosineSimilarityLoss, self).__init__()
        self.pad_idx = pad_idx
        self.code_map_offset = code_map_offset
        self.embedding_dim = embedding_dim

        # 定义嵌入层，将类别索引映射为向量
        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx=pad_idx)

    def forward(self, logits, code_seq_labels):
        """
        前向传播计算损失。

        Args:
            logits (Tensor): [batch_size, seq_len, num_classes] 模型输出的 logits。
            code_seq_labels (Tensor): [batch_size, seq_len] 真实的码序列。

        Returns:
            Tensor: 标量损失值。
        """

        # 创建掩码，忽略填充位置
        mask = code_seq_labels != self.pad_idx  # [batch_size, seq_len]

        # 将真实标签转换为嵌入向量
        true_embeds = self.embedding(code_seq_labels)  # [batch_size, seq_len, embedding_dim]

        # 计算预测的嵌入向量
        # 使用 softmax 获取概率分布
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]

        # 计算预测的嵌入向量为概率加权的嵌入
        pred_embeds = torch.matmul(probs, self.embedding.weight)  # [batch_size, seq_len, embedding_dim]

        # # 减去 code_map_offset
        # true_embeds = (true_embeds - self.code_map_offset).float()
        # pred_embeds = (pred_embeds - self.code_map_offset).float()

        # 将掩码扩展到嵌入维度
        mask = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]

        # 仅计算有效位置的嵌入向量
        true_embeds = true_embeds * mask  # [batch_size, seq_len, embedding_dim]
        pred_embeds = pred_embeds * mask  # [batch_size, seq_len, embedding_dim]

        true_embeds = true_embeds.reshape(-1, self.embedding_dim)  # [batch_size * seq_len, embedding_dim]
        pred_embeds = pred_embeds.reshape(-1, self.embedding_dim)  # [batch_size * seq_len, embedding_dim]

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(true_embeds, pred_embeds, dim=1)  # [batch_size]

        # 计算余弦距离作为损失
        cosine_distance = 1 - cosine_similarity  # [batch_size]

        # 计算平均损失
        loss = cosine_distance.mean()

        return loss
    
    
class SeqCosineSimilarityLoss(nn.Module):
    def __init__(self, pad_idx=0, code_map_offset=1):
        """
        基于余弦相似度的损失函数。

        Args:
            num_classes (int): 类别数量。
            embedding_dim (int): 嵌入维度。
            pad_idx (int): 填充符号的索引。
            code_map_offset (int): 码映射偏移量。
            mod_uniq_symbol (tuple, optional):
                一个元组，指示是否修改唯一符号及相关参数。
                格式：(bool, preds_mod_tensor, labels_mod_tensor)。
                默认为 (False, None, None)。
        """
        super(SeqCosineSimilarityLoss, self).__init__()
        self.pad_idx = pad_idx
        self.code_map_offset = code_map_offset

    def forward(self, logits, code_seq_labels):
        """
        前向传播计算损失。

        Args:
            logits (Tensor): [batch_size, seq_len, num_classes] 模型输出的 logits。
            code_seq_labels (Tensor): [batch_size, seq_len] 真实的码序列。

        Returns:
            Tensor: 标量损失值。
        """

        # 创建掩码，忽略填充位置
        mask = code_seq_labels != self.pad_idx
        
        