# -*- coding: utf-8 -*-

"""
Module định nghĩa các tầng Biaffine (Bilinear Affine Attention) và Element-wise Biaffine.
Dùng để tính toán ma trận điểm số tương tác trực tiếp giữa hai tập vector biểu diễn,
thường được sử dụng trong các bài toán Dependency Parsing và Attention.
"""

import torch
import torch.nn as nn


class Biaffine(nn.Module):
    """
    Lớp Biaffine (Bilinear Affine Attention).
    Sử dụng ma trận trọng số 3 chiều (weight tensor) tương tác trực tiếp 2 vector x và y.
    Công thức toán học: s = x^T U y + W(x \\oplus y) + b (Áp dụng trong Dependency Parsing). 
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

class ElementWiseBiaffine(nn.Module):
    """
    Biến thể Element-wise Biaffine.
    Chỉ thực hiện phép nhân độc lập giữa từng phần tử x_i và y_i của chuỗi thay vì tính tương quan chéo (cross-correlation).
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(ElementWiseBiaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        """
        
        (x1, x2, x3, ...)W(y1, y2, y3, ...) -> (z1, z2, z3),
        z1 = x1 W y1

        Args:
            x (Tensor(B, L-1, D)))
            y (Tensor(B, L-1, D)))

        Returns:
            Tensor(B, L-1, 1))
        """
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len]
        s = torch.einsum('bni,oij,bnj->bon', x, self.weight, y)
        # remove dim 1 if n_out == 1\
        s = s.squeeze(1)
        # (B, N)
        
        return s