"""
序列推荐 Decoder 模型

核心思想：
- 将用户的历史交互序列（物品的Semantic ID序列）作为输入
- 使用Transformer Decoder自回归地预测下一个物品的Semantic ID
- Semantic ID是多层code的序列 [code_0, code_1, ..., code_{N-1}]
  用分隔符SEP连接: [BOS, item1_code0, item1_code1, SEP, item2_code0, item2_code1, SEP, ..., EOS]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, d_model]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SemanticIDDecoder(nn.Module):
    """
    序列推荐的Decoder模型

    输入: 用户历史交互的Semantic ID序列
    输出: 预测下一个物品的Semantic ID（逐层预测）

    特殊Token设计:
    - PAD = 0: 填充
    - BOS = 1: 序列开始
    - EOS = 2: 序列结束
    - SEP = 3: 物品间分隔符
    实际code从4开始（code_value + special_token_size）
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        nhead = config["nhead"]
        num_layers = config["num_layers"]
        dim_feedforward = config["dim_feedforward"]
        dropout = config["dropout"]
        num_codebooks = config["num_codebooks"]
        codebook_size = config["codebook_size"]
        special_token_size = config["special_token_size"]

        # 用户特征类别数（可选，不配则不使用用户特征）
        self.use_user_features = config.get("use_user_features", False)
        self.num_user_tokens = 0  # 用户特征前缀token数

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.special_token_size = special_token_size

        # 词表大小 = 特殊token + code值
        vocab_size = special_token_size + codebook_size

        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # 用户特征Embedding（可选）
        if self.use_user_features:
            num_genders = config.get("num_genders", 2)
            num_ages = config.get("num_ages", 7)
            num_occupations = config.get("num_occupations", 21)

            # 每个特征维度一个独立的Embedding
            self.gender_embedding = nn.Embedding(num_genders, d_model)
            self.age_embedding = nn.Embedding(num_ages, d_model)
            self.occupation_embedding = nn.Embedding(num_occupations, d_model)

            # 3个用户token
            self.num_user_tokens = 3

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 输出层：预测每个位置的token
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码（下三角矩阵），确保每个位置只能看到之前的token
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor = None,
        user_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播
        输入:
            input_ids [B, seq_len]: 输入token序列
            padding_mask [B, seq_len]: True表示填充位置
            user_features [B, 3]: 用户特征 [gender_idx, age_idx, occ_idx]（可选）
        返回:
            logits [B, total_len, vocab_size]: 每个位置的token预测概率
            (total_len = seq_len + num_user_tokens 如果使用用户特征)
        """
        # Token嵌入
        x = self.token_embedding(input_ids) * math.sqrt(
            self.d_model
        )  # [B, seq_len, d_model]

        # 可选：拼接用户特征前缀
        if self.use_user_features and user_features is not None:
            # user_features: [B, 3] 分别为 gender_idx, age_idx, occ_idx
            gender_emb = self.gender_embedding(user_features[:, 0])  # [B, d_model]
            age_emb = self.age_embedding(user_features[:, 1])  # [B, d_model]
            occ_emb = self.occupation_embedding(user_features[:, 2])  # [B, d_model]

            # 拼接为前缀 [B, 3, d_model]
            user_emb = torch.stack([gender_emb, age_emb, occ_emb], dim=1)

            # 拼接到序列前面: [B, 3+seq_len, d_model]
            x = torch.cat([user_emb, x], dim=1)

            # 扩展padding_mask（用户token不填充）
            if padding_mask is not None:
                user_pad = torch.zeros(
                    padding_mask.size(0),
                    self.num_user_tokens,
                    dtype=padding_mask.dtype,
                    device=padding_mask.device,
                )
                padding_mask = torch.cat([user_pad, padding_mask], dim=1)

        # 位置编码
        x = self.pos_encoding(x)
        seq_len = x.size(1)

        # 因果掩码
        causal_mask = self._generate_causal_mask(seq_len, input_ids.device)

        # Transformer Decoder
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # 输出投影
        logits = self.output_proj(output)  # [B, total_len, vocab_size]

        return logits

    def predict_next_token(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor = None,
        user_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        推理时：预测序列下一个token的logits
        输入: input_ids [B, seq_len], padding_mask [B, seq_len]（可选）, user_features [B, 3]（可选）
        返回: next_token_logits [B, vocab_size]
        """
        logits = self.forward(input_ids, padding_mask, user_features=user_features)
        return logits[:, -1, :]  # 只取最后一个位置
