# -*- coding: utf-8 -*-

"""
metric.py: Định nghĩa các class dùng để tính toán các độ đo (metrics) đánh giá chất lượng mô hình dự đoán.
- Metric: Class cơ sở định nghĩa các tính chất so sánh score.
- PosMetric: Tính toán Accuracy, Precision, Recall, F1-score tổng hợp cho các task phân loại nhãn (vd: POS Tagging).
- SegF1Metric: Tính toán Precision, Recall, F1 cho các tập hợp phân đoạn (spans), dùng cho thao tác ghép ngoặc hoặc cắt từ học sâu (deep segmentation).
"""

import sys

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from collections import Counter


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class PosMetric(Metric):

    def __init__(self, eps=1e-8):
        super(PosMetric, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

        self.gold_ls = []
        self.pred_ls = []

    def __call__(self, preds, golds, lens, word_ids=None):
        """
        Gộp chuỗi dự đoán (preds) và nhãn đúng (golds) lại để chuẩn bị tính điểm đánh giá F1.

        Args:
            preds: List[List[int]] (Danh sách chuỗi nhãn dự đoán)
            golds: List[List[int]] (Danh sách chuỗi nhãn đúng ground truth)
            lens: Độ dài thực của câu (để bỏ qua phần tử padding)
        """

        golds = golds.tolist()
        # preds = preds.tolist()
        if word_ids:
            self.back_to_original_state(preds, golds, lens, word_ids)
        else:
            self.aligned_state(preds, golds, lens)

    def aligned_state(self, preds, golds, lens):
        for pred, gold, l in zip(preds, golds, lens):
            self.gold_ls.extend(gold[: l])
            self.pred_ls.extend(pred[: l])

    def back_to_original_state(self, preds, golds, lens, word_ids):
        for pred, gold, l, index in zip(preds, golds, lens, range(len(lens))):
            previous = -1
            for i, word_id in enumerate(word_ids(batch_index=index)[:l]):
                if word_id is not None and word_id != previous:
                    self.gold_ls.append(gold[i])
                    self.pred_ls.append(pred[i])
                    previous = word_id
        # print(self.pred_ls)
        # print(self.gold_ls)

    def __repr__(self):
        r = f"Acc: {self.acc:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} "
        r += f"weighted-F: {self.f:6.2%}"
        # r += f"weighted-F: {self.f:6.2%} macro-F: {self.macro_f:6.2%} "
        # r += f"micro-F: {self.micro_f:6.2%}"
        return r

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return precision_score(self.gold_ls, self.pred_ls,
                               average='weighted', zero_division=1)

    @property
    def r(self):
        return recall_score(self.gold_ls, self.pred_ls,
                            average='weighted', zero_division=1)

    @property
    def f(self):
        # Tính F1-score có trọng số (weighted F1) dựa trên tất cả các nhãn
        return f1_score(self.gold_ls, self.pred_ls,
                        average='weighted', zero_division=1)

    @property
    def macro_f(self):
        # Tính F1-score bình quân (macro F1 - tất cả các class đều xem trọng như nhau)
        return f1_score(self.gold_ls, self.pred_ls,
                        average='macro', zero_division=1)

    @property
    def micro_f(self):
        # Tính F1-score gộp chung số mẫu (micro F1)
        return f1_score(self.gold_ls, self.pred_ls,
                        average='micro', zero_division=1)

    @property
    def acc(self):
        return accuracy_score(self.gold_ls, self.pred_ls)

class SegF1Metric(Metric):

    def __init__(self, eps=1e-8):
        super(SegF1Metric, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        """[summary]

        Args:
            preds (): List[List[tuple(i, j)]]
            golds ([type]): [description]
        """
        for pred, gold in zip(preds, golds):
            tp = list((Counter(pred) & Counter(gold)).elements())
            self.tp += len(tp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

