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

import numpy as np

from .cmd_single import CMD
from .cmd_single import CMD

from torch.utils.data import DataLoader


def _compute_metrics_from_confusion(confusion, label_names=None, ignore_ids=None):
    """
    Tính Precision, Recall, F1 từ confusion matrix (không lưu toàn bộ pred/gold lên RAM).
    
    Args:
        confusion: numpy array (n_labels x n_labels), confusion[gold][pred] = count
        label_names: dict {id: name} hoặc None
        ignore_ids: set of label ids to ignore khi tính macro metrics (vd: 'O' label)
    
    Returns:
        dict chứa per-class metrics và overall weighted/macro metrics
    """
    if ignore_ids is None:
        ignore_ids = set()
    
    n = confusion.shape[0]
    total_correct = 0
    total_samples = 0
    
    per_class = {}
    
    for i in range(n):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp  # Cột i: tất cả pred = i, trừ đúng
        fn = confusion[i, :].sum() - tp  # Hàng i: tất cả gold = i, trừ đúng
        support = confusion[i, :].sum()  # Tổng mẫu thực sự thuộc class i
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        name = label_names.get(i, str(i)) if label_names else str(i)
        per_class[i] = {
            'name': name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support),
        }
        
        total_correct += tp
        total_samples += support
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Weighted average (trọng số theo support, bỏ qua ignore_ids)
    w_p, w_r, w_f = 0.0, 0.0, 0.0
    weighted_total = sum(per_class[i]['support'] for i in range(n) if i not in ignore_ids)
    for i in range(n):
        if i in ignore_ids:
            continue
        w = per_class[i]['support'] / weighted_total if weighted_total > 0 else 0
        w_p += per_class[i]['precision'] * w
        w_r += per_class[i]['recall'] * w
        w_f += per_class[i]['f1'] * w
    
    # Macro average (bỏ qua ignore_ids)
    active = [i for i in range(n) if i not in ignore_ids and per_class[i]['support'] > 0]
    m_p = sum(per_class[i]['precision'] for i in active) / len(active) if active else 0.0
    m_r = sum(per_class[i]['recall'] for i in active) / len(active) if active else 0.0
    m_f = sum(per_class[i]['f1'] for i in active) / len(active) if active else 0.0
    
    # Micro average (tổng TP/FP/FN toàn cục, bỏ qua ignore_ids)
    micro_tp = sum(confusion[i, i] for i in active)
    micro_fp = sum(confusion[:, i].sum() - confusion[i, i] for i in active)
    micro_fn = sum(confusion[i, :].sum() - confusion[i, i] for i in active)
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'weighted_precision': w_p,
        'weighted_recall': w_r,
        'weighted_f1': w_f,
        'macro_precision': m_p,
        'macro_recall': m_r,
        'macro_f1': m_f,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f1': micro_f,
        'per_class': per_class,
        'total_samples': int(total_samples),
        'ignore_ids': ignore_ids,
    }


def _print_metrics(metrics, label_names=None):
    """In kết quả metrics ra console theo dạng bảng."""
    print("\n" + "=" * 72)
    print("  EVALUATION METRICS")
    print("=" * 72)
    
    # Per-class table (bỏ qua padding label)
    ignore_ids = metrics.get('ignore_ids', set())
    print(f"  {'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 72)
    for i, info in sorted(metrics['per_class'].items()):
        if i in ignore_ids:
            continue
        name = info['name']
        print(f"  {name:<12} {info['precision']:>10.4f} {info['recall']:>10.4f} "
              f"{info['f1']:>10.4f} {info['support']:>10d}")
    
    print("-" * 72)
    print(f"  {'Accuracy':<12} {'':>10} {'':>10} {metrics['accuracy']:>10.4f} {metrics['total_samples']:>10d}")
    print(f"  {'Weighted':<12} {metrics['weighted_precision']:>10.4f} {metrics['weighted_recall']:>10.4f} "
          f"{metrics['weighted_f1']:>10.4f} {metrics['total_samples']:>10d}")
    print(f"  {'Macro':<12} {metrics['macro_precision']:>10.4f} {metrics['macro_recall']:>10.4f} "
          f"{metrics['macro_f1']:>10.4f} {metrics['total_samples']:>10d}")
    print(f"  {'Micro':<12} {metrics['micro_precision']:>10.4f} {metrics['micro_recall']:>10.4f} "
          f"{metrics['micro_f1']:>10.4f} {metrics['total_samples']:>10d}")
    print("=" * 72 + "\n")


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
            
            # Khởi tạo confusion matrix (dùng confusion matrix thay vì lưu toàn bộ lên RAM)
            n_labels = getattr(loader, 'n_labels', None)
            has_metrics = n_labels is not None
            if has_metrics:
                confusion = np.zeros((n_labels, n_labels), dtype=np.int64)
                print(f"[Metrics] Confusion matrix initialized: {n_labels} labels")
            
            with open(args.pred_path, mode='w', encoding='utf-8') as f:
                import torch
                with torch.no_grad():
                    for data in test:
                        # Unpack 7 phần tử (thêm tags so với trước)
                        chars, bi_chars, bert_input, attention_mask, mask, str_chars, tags = data
                        feed_dict = {'chars': chars, 'bert': [bert_input, attention_mask], 'crf_mask': mask}
                        ret = self.model(feed_dict, do_predict=True)
                        
                        # Cập nhật confusion matrix batch-by-batch (không lưu toàn bộ pred lên RAM)
                        if has_metrics and tags is not None:
                            preds = ret['predict']  # List[List[int]]
                            for i in range(len(preds)):
                                l = mask[i].sum(dim=-1).item()
                                pred_seq = preds[i][:l]
                                gold_seq = tags[i][:l].tolist()
                                for p, g in zip(pred_seq, gold_seq):
                                    if g < n_labels and p < n_labels:
                                        confusion[g][p] += 1
                        
                        for i, (char_line, punc) in enumerate(zip(str_chars, ret['predict'])):
                            lens = mask[i].sum(dim=-1).item()
                            if hasattr(loader, 'back_2_sentence_last'):
                                tokens = loader.back_2_sentence_last(char_line, punc, lens)
                            else:
                                tokens = char_line # fallback generic logic
                            f.write("".join(tokens) + '\n')
                        total_num += mask.sum().item()
            print("Numbers of total chars", total_num)
            
            # In kết quả metrics
            if has_metrics:
                label_names = getattr(loader, 'id2task_labels', None)
                # Bỏ padding label khi tính macro
                ignore_ids = {n_labels - 1}  # pad label id
                metrics = _compute_metrics_from_confusion(confusion, label_names, ignore_ids)
                _print_metrics(metrics, label_names)
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

