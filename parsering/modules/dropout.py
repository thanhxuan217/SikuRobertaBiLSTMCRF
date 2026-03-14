# -*- coding: utf-8 -*-

"""
Module định nghĩa các kỹ thuật Dropout chuyên biệt như SharedDropout và IndependentDropout.
Giúp giữ ổn định quá trình học lâu dài của chuỗi tuần tự và giảm thiểu hiện tượng quá khớp (overfitting).
"""

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    """
    Lớp Variational Dropout (Dropout chia sẻ).
    Sử dụng cùng 1 mặt nạ dropout (mask) cho tất cả các bước thời gian (time-steps) của dữ liệu chuỗi RNN.
    Giúp giữ được sự ổn định của chuỗi dài hạn, tránh mất thông tin do rớt nơ-ron không đồng đều.
    """

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class IndependentDropout(nn.Module):
    """
    Kỹ thuật Independent Dropout.
    Thay vì áp dụng chung một phân phối mask, hàm này tính toán mask độc lập cho nhiều tensor (items) đầu vào.
    Áp dụng tỷ lệ scale để giữ cân bằng kỳ vọng của đầu ra.
    """

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        """
        Các items có thể là các vector đặc trưng đầu vào khác nhau như word embeddings, char embeddings, tag embeddings...
        """
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items
