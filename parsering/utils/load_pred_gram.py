# -*- coding: utf-8 -*-
"""
load_pred_gram.py: Chịu trách nhiệm load và tiền xử lý dữ liệu đầu vào trong quá trình dự đoán (inference) bằng mô hình Two-CRF/Gram.
"""
# @Time    : 2024/3/2 11:19
# @Author  : wxb
# @File    : load_pred_gram.py

import torch
from collections import Counter

from .load import Load
from ..utils.common import punctuation, tag_before, tag_after
from ..utils.common import pad, bos, eos


class Load_pred(Load):
    def __init__(self, args):
        super().__init__(args)
        self.test, self.sliding_ids, self.enters = self.read_pred_file()
        self.test = self.pred_2_ids()

    def read_pred_file(self):
        f = open(self.args.pred_data, mode='r', encoding='utf-8')
        lines = [line.strip() for line in f]
        f.close()

        new_lines = []
        repeat_indexes = []  # tuples
        enters = Counter()
        index = -1
        for line in lines:
            if len(line) > 510:
                start = len(new_lines)
                new_lines.extend([list(each) for each in self.sliding_window(line)])
                end = len(new_lines)
                repeat_indexes.append((start, end, len(line)))
                index += 1
                continue
            if line:
                new_lines.append(list(line))
                index += 1
            else:
                enters.update([index])
        print('sliding_ids', repeat_indexes)
        return new_lines, repeat_indexes, enters
    
    def pred_2_ids(self):
        new_dataset = []
        for words in self.test:
            bert_input = self.tokenizer.encode("".join(words),
                                               truncation=True,
                                               max_length=512)
            bert_input = torch.tensor(bert_input)
            length = len(bert_input)

            words_index = torch.tensor([self.chars2ids[e]
                                        if self.chars2ids.get(e)
                                        else self.chars2ids[pad]
                                        for e in [bos] + words[:length - 2] + [eos]])
            bi_words = [bos] + ["".join(words[i: i + 2]) for i in range(length - 3)] + [eos, pad]
            bi_words_index = torch.tensor([self.bichars2ids[e]
                                           if self.bichars2ids.get(e)
                                           else self.bichars2ids[pad]
                                           for e in bi_words])

            attention_mask = torch.ones(length).gt(0)
            mask = torch.tensor([0] + [1] * (length - 2) + [0])
            line = (words_index, bi_words_index, bert_input, attention_mask, mask, words)

            new_dataset.append(line)
        return new_dataset
                
    def collate_fn_bigram_pred(self, batch):
        tokens, bi_chars, bert_input, attention_mask, mask, strwords = zip(*batch)
        tokens = self.pad(tokens, padding_value=self.chars2ids[pad])
        bi_chars = self.pad(bi_chars, padding_value=self.bichars2ids[pad])
        bert_input = self.pad(bert_input, padding_value=self.tokenizer.pad_token_id)
        mask = self.pad(mask, padding_value=0).bool()
        attention_mask = self.pad(attention_mask, padding_value=0).bool()

        assert mask.shape == bert_input.shape == tokens.shape == bi_chars.shape, f"{mask.shape}, {bert_input.shape}, {tokens.shape}, {bi_chars.shape}"
        return (tokens.to(self.args.device), bi_chars.to(self.args.device), bert_input.to(self.args.device),
                attention_mask.to(self.args.device), mask.to(self.args.device), strwords)

    def back_2_sentence(self, tokens, pred_stop, pred_non_stop, length):
        """

        :param length: 
        :param tokens: indexes of tokens in a line
        :param pred_non_stop: non-stop-punc of a line
        :param pred_stop: stop punc in a line
        :return: chars with punc without 'pad'
        """

        # pred_non_stop = pred_non_stop.tolist()
        # pred_stop = pred_stop.tolist()
        # print(len(tokens), len(pred_stop), len(pred_non_stop), length)
        # tokens_ids = tokens_ids.tolist()
        # tokens = [self.id2chars[index] for index in tokens_ids[1: 1+length]]
        # print("".join(tokens))

        # 2024.3.28 尝试第二种后处理：先放stop标点再放别的
        for j, stop in enumerate(pred_stop):
            # 必然表示停顿且放到后面的标点符号
            if (t := self.id2stop_labels.get(stop, 'O')) != 'O':
                tokens[j] += t

        for j, non_s in enumerate(pred_non_stop):
            # 放到前面的标签（不是表示停顿的标点符号）
            for tag in self.id2labels.get(non_s, 'O')[::-1]:   # todo: 这里好像不需要倒序
                if tag == 'O':
                    continue
                elif tag in tag_after:
                    tokens[j] = tokens[j] + tag
                elif tag in tag_before:
                    tokens[j] = tag + tokens[j]
                else:
                    assert False, f'Error.{self.id2labels.get(non_s)}'

        # for j, stop in enumerate(pred_stop):
        #     # 必然表示停顿且放到后面的标点符号
        #     if (t := self.id2stop_labels.get(stop)) != 'O':
        #         tokens[j] += t

        # print("".join(tokens))
        return tokens

    def back_2_sentence_last(self, tokens, pred_punc, length):
        """

        :param length:
        :param tokens: indexes of tokens in a line
        :param pred_non_stop: non-stop-punc of a line
        :param pred_stop: stop punc in a line
        :return: chars with punc without 'pad'
        """

        for j, punc_id in enumerate(pred_punc):
            # 放到前面的标签（不是表示停顿的标点符号）
            for tag in self.id2labels.get(punc_id, 'O'):
                if tag == 'O':
                    continue
                elif tag in tag_after:
                    tokens[j] = tokens[j] + tag
                elif tag in tag_before:
                    tokens[j] = tag + tokens[j]
                else:
                    tokens[j] = tokens[j] + tag
                    # assert False, f'Error. {self.id2labels.get(punc_id)}'

        return tokens

    def back_2_sentence_count(self, tokens, pred_stop, pred_non_stop, length):
        """

        :param length:
        :param tokens: indexes of tokens in a line
        :param pred_non_stop: non-stop-punc of a line
        :param pred_stop: stop punc in a line
        :return: chars with punc without 'pad'
        """

        for j, (stop, non_s) in enumerate(zip(pred_stop, pred_non_stop)):
            stop_punc = self.id2stop_labels.get(stop, "O")
            non_s_punc = self.id2labels.get(non_s, "O")
            if stop_punc == non_s_punc == "O":
                continue
            elif non_s_punc == 'O':
                tokens[j] = tokens[j] + stop_punc   # 必然表示停顿且放到后面的标点符号
            elif stop_punc == "O":
                for tag in non_s_punc:
                    if tag in tag_after:
                        tokens[j] = tokens[j] + tag
                    elif tag in tag_before:
                        tokens[j] = tag + tokens[j]
            else:
                new_tag_1 = stop_punc + non_s_punc
                new_tag_2 = non_s_punc + stop_punc
                new_tag = new_tag_1 if (
                        self.count[new_tag_1]
                        > self.count[new_tag_2]) else new_tag_2
                for tag in new_tag:
                    if tag in self.stop:
                        tokens[j] = tokens[j] + tag
                    else:
                        if tag in tag_after:
                            tokens[j] = tokens[j] + tag
                        elif tag in tag_before:
                            tokens[j] = tag + tokens[j]

        return tokens

    def merge(self, tokens_ls):
        """
        
        :param tokens_ls: several tokens made from sliding window 
        :return: one line
        """
        new_tokens = []
        # for each in tokens_ls:
        #     print(len(each), each)
        win = 100
        start = 0
        for i in range(len(tokens_ls)):
            t = len(tokens_ls[i])
            for j, token in enumerate(tokens_ls[i][-1: -win-1: -1]):
                if token[-1] in self.stop:
                    new_tokens.extend(tokens_ls[i][start: t-j])
                    start = win - j
                    break
        # print(len(new_tokens), "".join(new_tokens))
        return new_tokens
                