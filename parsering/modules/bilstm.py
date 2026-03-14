# -*- coding: utf-8 -*-

"""
Module định nghĩa tầng mạng nơ-ron hồi quy LSTM hai chiều (BiLSTM) tùy chỉnh.
Hỗ trợ áp dụng Shared Dropout (Variational Dropout) một cách nhất quán trên các bước thời gian (time-steps)
và tùy biến quá trình lướt xuôi/ngược (forward/backward pass) của chuỗi.
"""

from parsering.modules.dropout import SharedDropout

import torch
import torch.nn as nn
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence


class BiLSTM(nn.Module):
    """
    Tầng mạng nơ-ron hồi quy LSTM hai chiều (Bidirectional LSTM) tùy chỉnh.
    Sử dụng ModuleList chứa các LSTMCell thay vì nn.LSTM mặc định để dễ dàng can thiệp
    vào quá trình forward (ví dụ: áp dụng dropout tùy biến cho từng bước thời gian).
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Danh sách các ô tính toán theo lướt xuôi (Forward Cell)
        self.f_cells = nn.ModuleList()
        # Danh sách các ô tính toán theo lướt ngược (Backward Cell)
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            input_size = hidden_size * 2 # LSTM 2 chiều nên output của lớp trước sẽ nhân đôi chiều đưa vào lớp sau

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += ')'

        return s

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        """
        Quá trình lan truyền tiến cho một tầng LSTM cụ thể.
        Nhận vào chuỗi đã được loại bỏ padding bằng PackedSequence.
        """
        hx_0 = hx_i = hx
        hx_n, output = [], []
        # Quyết định chạy tiến hay chạy lùi của một chiều (forward/backward)
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            # Shared dropout đảm bảo cùng một mặt nạ rớt nơ-ron được áp dụng trên tất cả các bước thời gian (step)
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
                        for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        """
        Đầu vào là PackedSequence, thực hiện lần lượt qua từng lớp layer LSTM.
        """
        # sequence.data: vector các tokens (không tính padding)
        # batch_sizes: chứa kích thước batch ở mỗi time-step để LSTM biết lấy bao nhiêu mẫu tính toán
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            # Khởi tạo vector State ẩn (hidden) và State tế bào (cell)
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        # Chạy từng lớp Layer
        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            
            # Forward pass: lướt từ đầu -> cuối
            x_f, (h_f, c_f) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            # Backward pass: lướt từ cuối -> đầu
            x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                 hx=(h[i, 1], c[i, 1]),
                                                 cell=self.b_cells[i],
                                                 batch_sizes=batch_sizes,
                                                 reverse=True)
            # Ghép output ở 2 chiều lại
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
            
        # Đóng gói vector trở lại format PackedSequence để đưa ra khỏi mô hình
        x = PackedSequence(x,
                           sequence.batch_sizes,
                           sequence.sorted_indices,
                           sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx
