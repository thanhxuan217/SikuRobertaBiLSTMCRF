# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 20:07
# @Author  : wxb
# @File    : cmd_bigram.py
"""
File này chứa lớp `CMD` cơ sở cấu hình cho kiến trúc mô hình Single-CRF (RoBERTa BiLSTM CRF) 
chỉ phục vụ một luồng tác vụ gộp duy nhất. 
Cung cấp các luồng chức năng tiêu chuẩn: `train()`, đo lường chỉ số qua `evaluate()` và chạy dự đoán với `predict()`.
"""

import os
import sys
from typing import Any
from copy import deepcopy

from ..BasePlusModel import roberta_bilstm_crf

from ..utils.metric import PosMetric

import torch
import torch.nn as nn


class CMD(object):
    """
    Lớp CMD dành riêng cho Single-Task CRF.
    (Khác với cmd_gram.py giải quyết phân mảng & ngắt câu tách biệt bằng 2 luồng CRF, file này thiết lập mô hình BasePlusModel chỉ có 1 CRF duy nhất).
    """

    def __call__(self, args) -> Any:
        self.args = args
        # Tạo file/thư mục lưu cấu hình nếu chưa có
        if not os.path.exists(args.file):
            os.mkdir(args.file)

        self.model_check = args.base_model
        
        # Chỉ định chạy mô hình roberta_bilstm_crf (1 nhánh MLP và CRF)
        self.model_cl = roberta_bilstm_crf

        args.update({
            'model_check': self.model_check,
            'model_cl': self.model_cl,
        })

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def train(self, loader):
        """
        Huấn luyện mô hình một epoch (luồng Single Task).
        """
        self.model.train()
        torch.set_grad_enabled(True)
        for data in loader:
            # Nhận data chỉ có chung 1 loại 'tags' duy nhất (thay vì stop_tags và non_stop_tags như gram model)
            ((chars, bi_chars, bert_input, attention_mask, mask), tags) = data
            self.optimizer.zero_grad()

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}

            # Forward mô hình, do chỉ có 1 task nên biến trả về duy nhất một dictionary chứa tổng Loss
            ret = self.model(feed_dict, tags)
            loss = ret['loss']
            
            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Hàm đánh giá mô hình Single Task.
        Ở đây dùng duy nhất PosMetric (Đo Precision/Recall/F1 cho nhãn gộp chung phân mảng & ngắt câu).
        """
        print('evaluate...')
        self.model.eval()
        total_loss, metric_pos = 0, PosMetric()
        total_re, total_num = 0, 0

        for data in loader:
            ((chars, bi_chars, bert_input, attention_mask, mask), tags) = data
            self.optimizer.zero_grad()

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}
                         
            # Bật do_predict=True để giải mã thuật toán viterbi lấy nhãn dự đoán (predict vector)
            ret = self.model(feed_dict, tags, do_predict=True)
            loss = ret['loss']

            total_loss += loss.item()

            pred = ret['predict']
            # Đánh giá chỉ số chung cho toàn bộ loại nhãn
            metric_pos(pred, tags, mask.sum(dim=-1))

            total_num += mask.sum()

        total_loss /= len(loader)

        return total_loss, metric_pos

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        chars_preds = []
        lens = []
        total_re, total_num = 0, 0
        for data in loader:
            chars, bi_chars, bert_input, attention_mask, mask, str_chars = data

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}

            ret = self.model(feed_dict, do_predict=True)
            for char_line, punc in zip(str_chars, ret['predict']):
                chars_preds.append((char_line, punc))

            lens.append(mask.sum(dim=-1))
            total_num += mask.sum()
        print("Numbers of total chars", total_num)
        return chars_preds, torch.cat(lens)

