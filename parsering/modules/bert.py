# -*- coding: utf-8 -*-

"""
Module định nghĩa tầng BertEmbedding.
Sử dụng thư viện transformers để tải và trích xuất đặc trưng (embedding) ngữ cảnh
từ các mô hình ngôn ngữ dựa trên kiến trúc BERT/RoBERTa.
Hỗ trợ QLoRA: load model ở dạng 4-bit quantization khi được chỉ định.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, logging
logging.set_verbosity_error()  # 忽略 bert 警告


class BertEmbedding(nn.Module):
    """
    Lớp nhúng (embedding) từ mô hình ngôn ngữ cài sẵn họ BERT/RoBERTa.
    Sử dụng thư viện transformers để tải và xử lý tự động (AutoModel).
    Lớp này dùng để trích xuất đặc trưng (embedding) ngữ cảnh của các subwords.
    
    Khi quantization_config được cung cấp, model sẽ được load ở dạng 4-bit
    quantization (QLoRA) để tiết kiệm VRAM.
    """

    def __init__(self, model, n_layers, n_out, requires_grad=False, quantization_config=None):
        super(BertEmbedding, self).__init__()

        self.config = AutoConfig.from_pretrained(model)
        
        # Load model với quantization config nếu có (QLoRA)
        if quantization_config is not None:
            self.bert = AutoModel.from_pretrained(
                model,
                config=self.config,
                quantization_config=quantization_config,
            )
        else:
            self.bert = AutoModel.from_pretrained(model, config=self.config)
        
        # self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers
        self.n_out = n_out
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size

        if self.hidden_size != n_out:
            self.projection = nn.Linear(self.hidden_size, n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, subwords, bert_mask):
        """
        Lan truyền tiến (Forward). Đưa mask và subwords (mã token) vào BERT để lấy vector nhúng ngữ cảnh.
        """
        if not self.requires_grad:
            self.bert.eval()
        embed = self.bert(subwords, attention_mask=bert_mask).last_hidden_state

        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed
