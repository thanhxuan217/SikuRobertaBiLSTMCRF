# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 20:07
# @Author  : wxb
# @File    : cmd_bigram.py
"""
File này định nghĩa lớp `CMD` cơ sở dành cho kiến trúc mô hình Hai-CRF (Bigram BERT Model) 
nhằm giải quyết song song hai tác vụ: phân mảng (segmentation) và ngắt câu (punctuation). 
Cung cấp các hàm core pipeline như `train()`, `evaluate()` và `predict()`.
"""

import os
import sys
from typing import Any
from copy import deepcopy

from ..gram_crf_model import bigram_bert_model

from ..utils.metric import PosMetric

import torch
import torch.nn as nn


class CMD(object):

    def __call__(self, args) -> Any:
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)

        self.model_check = args.base_model

        self.model_cl = bigram_bert_model

        args.update({
            'model_check': self.model_check,
            'model_cl': self.model_cl,
        })

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def train(self, loader):
        """
        Hàm thực hiện một epoch huấn luyện.
        """
        self.model.train() # Đưa mô hình về chế độ huấn luyện (kích hoạt dropout, v.v.)
        torch.set_grad_enabled(True) # Bật tính toán đạo hàm
        for data in loader:
            # Dữ liệu đầu vào lấy từ DataLoader
            ((chars, bi_chars, bert_input, attention_mask, mask),
             non_stop_tags, stop_tags) = data
            
            self.optimizer.zero_grad() # Xóa gradient của bước trước đó

            # Cấu trúc dictionary đưa vào mô hình (chứa dữ liệu BERT, word/character, mask CRF)
            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}

            # Truyền qua mô hình. Ở đây trả về hai dictionaries (cho stop và non-stop)
            stop, non_stop_ret = self.model(feed_dict, non_stop_tags, stop_tags)
            
            # Tổng hợp lỗi từ 2 task (classification)
            loss = non_stop_ret['loss'] + stop['loss']
            
            # Lan truyền ngược (Backpropagation) để tính gradient
            loss.backward()
            
            # Cắt bớt gradient (Gradient Clipping) để tránh bùng nổ gradient
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)

            # Cập nhật trọng số của mô hình
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Hàm dùng để đánh giá mô hình trên tập validation hoặc test.
        Vô hiệu hóa đạo hàm để tiết kiệm dung lượng và thời gian thực thi (torch.no_grad).
        """
        print('evaluate...')
        self.model.eval() # Chế độ đánh giá mô hình (không dùng dropout)
        total_loss = 0
        metric_span, metric_pos = PosMetric(), PosMetric()
        total_re, total_num = 0, 0

        for data in loader:
            ((chars, bi_chars, bert_input, attention_mask, mask),
             non_stop_tags, stop_tags) = data
            self.optimizer.zero_grad()

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}
            
            # Lần này thêm tham số `do_predict=True` để chạy thuận toán Viterbi giải mã CRF lấy kết quả dự đoán thay vì chỉ tính Loss
            stopre, non_stop_ret = self.model(feed_dict, non_stop_tags, stop_tags, do_predict=True)
            loss = non_stop_ret['loss'] + stopre['loss']

            total_loss += loss.item()

            pred = non_stop_ret['predict']
            # Đánh giá chỉ số (Accuracy, Precision, Recall, F1) cho phần cắt câu (non_stop)
            metric_span(pred, non_stop_tags, mask.sum(dim=-1))

            pred = stopre['predict']
            # Đánh giá chỉ số cho phần dấu câu (stop/punc)
            metric_pos(pred, stop_tags, mask.sum(dim=-1))

            total_num += mask.sum()

        total_loss /= len(loader)

        return total_loss, metric_span, metric_pos

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        chars_preds = []
        lens = []
        total_re, total_num = 0, 0
        for data in loader:
            chars, bi_chars, bert_input, attention_mask, mask, str_chars = data
            # feed_dict = {'chars': chars, 'bigram': bi_chars,
            #              'bert': [bert_input, attention_mask],
            #              'crf_mask': mask}
            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}

            stopre, non_stop_ret = self.model(feed_dict, do_predict=True)
            for char_line, stop, nonstop in zip(str_chars,
                                                stopre['predict'],
                                                non_stop_ret['predict']):
                chars_preds.append((char_line, stop, nonstop))

            lens.append(mask.sum(dim=-1))
            total_num += mask.sum()
        print("Numbers of total chars", total_num)
        return chars_preds, torch.cat(lens)

