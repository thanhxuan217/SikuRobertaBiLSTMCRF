# -*- coding: utf-8 -*-

"""
Module định nghĩa mạng nơ-ron truyền thẳng đa tầng (Multi-Layer Perceptron - MLP).
Thường được đóng vai trò là tầng tinh chế trung gian hoặc bộ phân loại chính (Classification Head)
để chuyển tiếp các vector đặc trưng ẩn đầu ra từ các tầng trước đó.
"""

from parsering.modules.dropout import SharedDropout

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Mạng nơ-ron truyền thẳng đa tầng).
    Được sử dụng như bộ phân loại chính (Classification Head) chuyển tiếp đặc trưng LSTM ra các vector nhãn.
    """

    def __init__(self, n_in, n_out, dropout=0):
        super(MLP, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        # Tầng biến đổi tuyến tính (Fully Connected Layer)
        self.linear = nn.Linear(n_in, n_out)
        # Hàm kích hoạt phi tuyến (LeakyReLU)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        Quá trình Feed-forward: Biến đổi tuyến tính -> Kích hoạt phi tuyến -> Dropout
        """
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
