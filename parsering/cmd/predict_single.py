# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
# @Author  : wxb
# @File    : predict_single.sh.py
"""
File này thiết lập lớp `Predict_single` cho quá trình suy luận với mô hình Single-CRF.
Đọc tập test không nhãn, đẩy qua mô hình đã huấn luyện (chạy Viterbi), 
sau đó nối nhãn, khôi phục thành văn bản hoàn chỉnh và lưu kết quả Output ra đĩa.
"""

import os
from datetime import datetime

from .cmd_single import CMD
from .cmd_single import CMD

from torch.utils.data import DataLoader


class Predict_single(CMD):
    """
    Lớp dùng để chạy suy luận (inference) cho mô hình Single-CRF.
    Cấu trúc tương tự như predict_gram, nhưng phục vụ riêng cho mô hình roberta_bilstm_crf duy nhất.
    """
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate a trained model.'
        )
        subparser.add_argument('--data', default='data/ptb_pos',
                               help='path to train file')
        subparser.add_argument('--pred_data', default='../dataset/EvaHan2024_testset.txt',
                               help='path to train file')
        subparser.add_argument('--pred_path', default='TestPredict/result.txt',
                               help='path to save the predict result.') # Nơi lưu dữ liệu output sau khi test
        return subparser

    def __call__(self, args):
        super(Predict_single, self).__call__(args)
        print('Load the dataset.')
        start = datetime.now()

        if getattr(args, 'streaming', False):
            from ..utils.load_pred_streaming import Load_pred_streaming as Load_pred
        else:
            from ..utils.load_pred_single import Load_pred

        # Nạp tệp dữ liệu test
        loader = Load_pred(args)
        
        # Hàm gom data cho prediction
        collate_fn = loader.collate_fn_bigram_pred

        test = DataLoader(loader.test, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn)

        print('Load the model.')
        # Khôi phục (Load) mô hình Single-Task đã huấn luyện 
        self.model = self.model_cl.load(args.save_model)
        print(self.model)

        if getattr(args, 'streaming', False):
            print("Running prediction on streaming dataset...")
            self.model.eval()
            total_num = 0
            with open(args.pred_path, mode='w', encoding='utf-8') as f:
                import torch
                with torch.no_grad():
                    for data in test:
                        # For single CRF prediction, data might be returned by collate_fn without stop/nonstop distinctions.
                        chars, bi_chars, bert_input, attention_mask, mask, str_chars = data
                        feed_dict = {'chars': chars, 'bert': [bert_input, attention_mask], 'crf_mask': mask}
                        ret = self.model(feed_dict, do_predict=True)
                        for i, (char_line, punc) in enumerate(zip(str_chars, ret['predict'])):
                            lens = mask[i].sum(dim=-1).item()
                            if hasattr(loader, 'back_2_sentence_last'):
                                tokens = loader.back_2_sentence_last(char_line, punc, lens)
                            else:
                                tokens = char_line # fallback generic logic
                            f.write("".join(tokens) + '\n')
                        total_num += mask.sum().item()
            print("Numbers of total chars", total_num)
        else:
            sliding_ids = loader.sliding_ids
            enters = loader.enters
            print('enter:', enters)
            
            # Chạy viterbi thu lấy nhãn dự đoán chung
            chars_preds, lens = self.predict(test)
            tokens_punc = []
            for i in range(len(lens)):
                # Cấu trúc Data của model single chỉ trả về mảng list của 1 luồng duy nhất
                char, punc = chars_preds[i]
                
                # Chức năng biến đổi các nhãn dấu punc / token word đã tách lại thành raw string
                tokens = loader.back_2_sentence_last(char, punc, lens[i])

                tokens_punc.append(tokens)
            
            # Hợp nhất các chuỗi con lại thành văn bản (do trước đó DataLoader áp dụng cửa sổ trượt (sliding window) giải quyết quá tải độ dài)
            for each in sliding_ids[::-1]:
                tokens_punc[each[0]] = loader.merge(tokens_punc[each[0]: each[1]])
                tokens_punc[each[0]+1:] = tokens_punc[each[1]:]

            # Ghi nội dung dự đoán xuống .txt
            with open(args.pred_path, mode='w', encoding='utf-8') as f:
                length = len(tokens_punc)
                for i in range(length):
                    f.write("".join(tokens_punc[i]))
                    if i != length - 1:
                        f.write('\n')
                    if temp := enters.get(i):
                        for _ in range(temp):
                            f.write('\n')

                if 'zz' in args.pred_path:
                    f.write('\n')

        print(f'{datetime.now() - start}s elapsed.')
        print(f'Predict result save in {args.pred_path}')
        print('Finish.')
