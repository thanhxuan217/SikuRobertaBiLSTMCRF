# -*- coding: utf-8 -*-
"""
load_pred_streaming.py: Chịu trách nhiệm load dữ liệu parquet streaming cho prediction.
"""
import os
import glob
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ..utils.common import pad, bos, eos
from .load_pred_gram import Load_pred as Load_pred_gram


def _find_parquet_files(directory):
    pattern = os.path.join(directory, "*.parquet")
    return sorted(glob.glob(pattern))

class ParquetStreamingDataset_pred(torch.utils.data.IterableDataset):
    def __init__(self, data_files, tokenizer, max_length=512):
        self.dataset = load_dataset('parquet', data_files=data_files, split='train', streaming=True)
        self.tokenizer = tokenizer
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.max_length = max_length

    def __iter__(self):
        for item in self.dataset:
            text = item["text"]
            if isinstance(text, str):
                text = list(text)

            bert_input = self.tokenizer.encode("".join(text), truncation=True, max_length=self.max_length)
            bert_input = torch.tensor(bert_input)
            length = len(bert_input)
            
            try:
                words_index = self.tokenizer.convert_tokens_to_ids([self.bos] + text[:length - 2] + [self.eos])
            except:
                words_index = []
                for c in [self.bos] + text[:length - 2] + [self.eos]:
                    words_index.append(self.tokenizer.vocab.get(c, self.tokenizer.unk_token_id))
            words_index = torch.tensor(words_index, dtype=torch.long)
            
            # Predict streaming doesn't really use bigram chars index much if model is single/gram, but fill 0s
            bi_words_index = torch.zeros(length, dtype=torch.long)
            
            attention_mask = torch.ones(length).gt(0)
            mask = torch.tensor([0] + [1] * (length - 2) + [0])
            
            yield (words_index, bi_words_index, bert_input, attention_mask, mask, text)

class Load_pred_streaming(Load_pred_gram):
    def __init__(self, args):
        # Initialize base Load to get vocab stuff
        from .load import Load
        Load.__init__(self, args)
        
        data_path = args.pred_data
        
        # Determine parquet files
        if os.path.isdir(data_path):
            test_files = _find_parquet_files(data_path)
        elif data_path.endswith('.parquet'):
            test_files = [data_path]
        else:
            # fallback glob
            test_files = glob.glob(data_path + "*.parquet") if not data_path.endswith('.parquet') else [data_path]
            # If not found but a directory exists with the name, find in it
            if not test_files and os.path.isdir(data_path):
                 test_files = _find_parquet_files(data_path)
                
        if not test_files:
            raise FileNotFoundError(f"Khong tim thay file .parquet nao phu hop cho prediction tai {data_path}")
            
        print(f"[Load_pred_streaming] Test files: {test_files}")
        
        self.test = ParquetStreamingDataset_pred(test_files, self.tokenizer)
        
    def collate_fn_bigram_pred(self, batch):
        tokens, bi_chars, bert_input, attention_mask, mask, strwords = zip(*batch)
        
        tokens = self.pad(tokens, padding_value=self.chars2ids.get(pad, self.tokenizer.pad_token_id))
        bi_pad = self.bichars2ids.get(pad, 0) if hasattr(self, 'bichars2ids') else 0
        bi_chars = self.pad(bi_chars, padding_value=bi_pad)
        
        bert_input = self.pad(bert_input, padding_value=self.tokenizer.pad_token_id)
        mask = self.pad(mask, padding_value=0).bool()
        attention_mask = self.pad(attention_mask, padding_value=0).bool()

        return (tokens.to(self.args.device), bi_chars.to(self.args.device), bert_input.to(self.args.device),
                attention_mask.to(self.args.device), mask.to(self.args.device), strwords)
