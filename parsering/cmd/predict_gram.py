# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
# @Author  : wxb
# @File    : predict_single.sh.py
"""
File này định nghĩa lớp `Predict` (kế thừa từ `CMD` của cmd_gram.py). 
Chịu trách nhiệm load dữ liệu test, khôi phục weights cho mô hình Hai-CRF đã huấn luyện, 
chạy lệnh suy luận (inference) và xuất kết quả dự đoán ra file text có kèm cơ chế cửa sổ trượt.
"""

import os
from datetime import datetime

from .cmd_gram import CMD
from .cmd_gram import CMD

from torch.utils.data import DataLoader


class Predict(CMD):
    """
    Lớp dùng để chạy suy luận (inference) cho mô hình Hai-CRF (phân mảng + ngắt câu).
    Dùng để sinh kết quả dự đoán (prediction) trên dữ liệu Test.
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
                               help='path to save the predict result.') # Nơi trút dữ liệu xuất ra
        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)
        print('Load the dataset.')
        start = datetime.now()

        if getattr(args, 'streaming', False):
            from ..utils.load_pred_streaming import Load_pred_streaming as Load_pred
        else:
            from ..utils.load_pred_gram import Load_pred

        # Dùng DataLoader đặc biệt (Load_pred) để chỉ load dữ liệu Test (không có nhãn đúng)
        loader = Load_pred(args)
        
        # Hàm gom data cho prediction
        collate_fn = loader.collate_fn_bigram_pred

        test = DataLoader(loader.test, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn)

        print('Load the model.')
        # Khôi phục (Load) lại mô hình đã được huấn luyện từ đường dẫn đã lưu
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
                        chars, bi_chars, bert_input, attention_mask, mask, str_chars = data
                        feed_dict = {'chars': chars, 'bert': [bert_input, attention_mask], 'crf_mask': mask}
                        stopre, non_stop_ret = self.model(feed_dict, do_predict=True)
                        for i, (char_line, stop, nonstop) in enumerate(zip(str_chars, stopre['predict'], non_stop_ret['predict'])):
                            lens = mask[i].sum(dim=-1).item()
                            tokens = loader.back_2_sentence_count(char_line, stop, nonstop, lens)
                            f.write("".join(tokens) + '\n')
                        total_num += mask.sum().item()
            print("Numbers of total chars", total_num)
        else:
            sliding_ids = loader.sliding_ids
            enters = loader.enters
            print('enter:', enters)
            
            # Trích xuất ký tự dự đoán và độ dài cấu trúc do phương thức predict cung cấp
            chars_preds, lens = self.predict(test)
            tokens_punc = []
            for i in range(len(lens)):
                # Tách ra 3 thành phần: nhóm kí tự, đánh dấu ngắt câu, đánh dấu phân mảng
                char, stop, non_stop = chars_preds[i]  # 分组预测的代码
                # tokens = loader.back_2_sentence(char, stop, non_stop, lens[i])  # 分组预测的代码
                
                # Khôi phục thành câu hoàn chỉnh dựa trên logic token sinh ra ở loader (ví dụ gộp lại từ các nhãn BI)
                tokens = loader.back_2_sentence_count(char, stop, non_stop, lens[i])  # 分组预测根据分布确定 order 的代码

                tokens_punc.append(tokens)

            # Hợp nhất các cửa sổ trượt (sliding windows) để ghép lại với nhau trong trường hợp câu ban đầu quá dài bị cắt dở
            for each in sliding_ids[::-1]:
                tokens_punc[each[0]] = loader.merge(tokens_punc[each[0]: each[1]])
                tokens_punc[each[0] + 1:] = tokens_punc[each[1]:]

            # Mở file và ghi (write) kết quả đầu ra
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
