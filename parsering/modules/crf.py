# -*- coding: utf-8 -*-

"""
Module định nghĩa tầng Conditional Random Field (CRF) tùy chỉnh, dành cho bài toán Gán nhãn chuỗi (Sequence Labeling).
Hỗ trợ tính toán hàm mất mát (Negative Log-Likelihood) khi huấn luyện và
giải mã tìm ra chuỗi nhãn tối ưu nhất bằng thuật toán Viterbi lúc dự đoán.
"""

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Tầng Conditional Random Field (CRF) tùy chỉnh.
    Sử dụng để gán nhãn chuỗi (Sequence Labeling) trong bài toán phân mảng / ngắt câu.
    """
    def __init__(self, n_labels, batch_first=True):
        super(CRF, self).__init__()

        self.n_labels = n_labels
        self.batch_first = batch_first
        # Ma trận học chuyển tiếp (transition matrix) từ nhãn i -> nhãn j
        self.trans = nn.Parameter(torch.Tensor(n_labels, n_labels))
        # Ma trận điểm số khởi đầu (Start Transition Score) của một chuỗi
        self.strans = nn.Parameter(torch.Tensor(n_labels))
        # Ma trận điểm số kết thúc (End Transition Score) của một chuỗi
        self.etrans = nn.Parameter(torch.Tensor(n_labels))

        self.reset_parameters()

    def extra_repr(self):
        s = f"n_labels={self.n_labels}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def reset_parameters(self):
        # Khởi tạo ma trận bằng 0 ban đầu
        nn.init.zeros_(self.trans)
        nn.init.zeros_(self.strans)
        nn.init.zeros_(self.etrans)

    def forward(self, emit, target, mask):
        """
        Tính toán NLL (Negative Log-Likelihood) Loss để huấn luyện mô hình.
        Loss = log(Z) - Thực tế v(True path score)
        """
        logZ = self.get_logZ(emit, mask)
        score = self.get_score(emit, target, mask)

        return logZ - score

    def get_logZ(self, emit, mask):
        """
        Tính toán hằng số tổng quát Log Partition Function (log Z) của CRF.
        Sử dụng thuật toán Forward Algorithm với kỹ thuật Dynamic Programming.
        """
        if self.batch_first:
            emit, mask = emit.transpose(0, 1), mask.t()
        seq_len, batch_size, n_labels = emit.shape

        alpha = self.strans + emit[0]  # [batch_size, n_labels]

        for i in range(1, seq_len):
            scores = self.trans + alpha.unsqueeze(-1)
            scores = torch.logsumexp(scores + emit[i].unsqueeze(1), dim=1)
            alpha[mask[i]] = scores[mask[i]]
        logZ = torch.logsumexp(alpha + self.etrans, dim=1).sum()

        return logZ / batch_size

    def get_score(self, emit, target, mask):
        """
        Tính điểm (score) của chuỗi nhãn đích đúng (ground truth sequence).
        Điểm = Điểm phát xạ (Emission) + Điểm chuyển tiếp (Transition).
        """
        if self.batch_first:
            emit, target, mask = emit.transpose(0, 1), target.t(), mask.t()
        seq_len, batch_size, n_labels = emit.shape
        scores = emit.new_zeros(seq_len, batch_size)

        # plus the transition score
        # Cộng điểm chuyển tiếp giữa các nhãn liền kề
        scores[1:] += self.trans[target[:-1], target[1:]]
        # plus the emit score
        # Cộng điểm phát xạ tại các bước thời gian tương ứng với nhãn đúng
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # filter some scores by mask
        score = scores.masked_select(mask).sum()

        ends = mask.sum(dim=0).view(1, -1) - 1
        # plus the score of start transitions
        # Cộng điểm khởi tạo bắt đầu
        score += self.strans[target[0]].sum()
        # plus the score of end transitions
        # Cộng điểm chặn cuối
        score += self.etrans[target.gather(dim=0, index=ends)].sum()

        return score / batch_size

    def viterbi(self, emit, mask):
        """
        Thuật toán Viterbi: Giải mã chuỗi nhãn tốt nhất (có tổng điểm số cao nhất) cho chuỗi dự đoán ở thời điểm evaluation phase.
        Return: List of Predicted Labels (Ví dụ: [O, O, Q_SY, H_SY])
        """
        if self.batch_first:
            emit, mask = emit.transpose(0, 1), mask.t()
        seq_len, batch_size, n_labels = emit.shape
        lens = mask.sum(0)

        delta = emit.new_zeros(seq_len, batch_size, n_labels)
        paths = emit.new_zeros(seq_len, batch_size, n_labels, dtype=torch.long)

        # Bước 1: Suy diễn xuôi (Forward pass) để lưu ma trận điểm (delta) và dấu vết đường đi (paths)
        delta[0] = self.strans + emit[0]  # [batch_size, n_labels]

        for i in range(1, seq_len):
            scores = self.trans + delta[i - 1].unsqueeze(-1)
            scores, paths[i] = scores.max(1)
            delta[i] = scores + emit[i]

        preds = []
        # Bước 2: Truy vết ngược (Backtracking) để tìm ra chuỗi path tối ưu nhất
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + self.etrans)

            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            # flip the predicted sequence before appending it to the list
            # Lật ngược chuỗi vì ta Backtracking từ cuối lên đầu
            preds.append(paths.new_tensor(predict).flip(0).cpu().tolist())

        return preds
