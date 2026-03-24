import torch
import os
import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from ..utils.common import pad, bos, eos
from parsering.task_config import get_task_config


class ParquetStreamingDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset doc du lieu .parquet theo kieu streaming,
    tranh load het vao RAM.
    Ho tro 2 kieu truyen data_files:
      - Duong dan thu muc chua nhieu file .parquet
      - Danh sach cu the cac file .parquet
    """
    def __init__(self, data_files, tokenizer, labels_dic, pad_id, is_crf2=False):
        # data_files co the la glob pattern (str) hoac list cac file paths
        self.dataset = load_dataset('parquet', data_files=data_files, split='train', streaming=True)
        self.tokenizer = tokenizer
        self.labels_dic = labels_dic
        self.pad_id = pad_id
        
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.is_crf2 = is_crf2

    def __iter__(self):
        for item in self.dataset:
            text = item["text"]
            labels = item["labels"]
            
            if isinstance(text, str):
                text = list(text)

            bert_input = self.tokenizer.encode("".join(text), truncation=True, max_length=512)
            bert_input = torch.tensor(bert_input)
            length = len(bert_input)
            
            try:
                words_index = self.tokenizer.convert_tokens_to_ids([self.bos] + text[:length - 2] + [self.eos])
            except:
                words_index = []
                for c in [self.bos] + text[:length - 2] + [self.eos]:
                    words_index.append(self.tokenizer.vocab.get(c, self.tokenizer.unk_token_id))
            words_index = torch.tensor(words_index, dtype=torch.long)
            
            bi_words_index = torch.zeros(length, dtype=torch.long)
            
            attention_mask = torch.ones(length).gt(0)
            mask = torch.tensor([0] + [1] * (length - 2) + [0])
            
            mapped_tags = [self.labels_dic.get(e, self.pad_id) for e in labels[:length-2]]
            tags = torch.tensor(mapped_tags, dtype=torch.long)
            
            if self.is_crf2:
                yield (words_index, bi_words_index, bert_input, attention_mask, mask, tags, tags)
            else:
                yield (words_index, bi_words_index, bert_input, attention_mask, mask, tags)


def _find_parquet_files(directory):
    """Tim tat ca file .parquet trong thu muc (khong de quy)."""
    pattern = os.path.join(directory, "*.parquet")
    return sorted(glob.glob(pattern))


class Load:
    """
    Loader streaming cho parquet dataset.
    Ho tro 3 kich ban:
      1. args.data la thu muc co sub-dir train/ va val/ -> doc rieng
      2. args.data la thu muc chi co train/ (khong co val/) -> tu dong split
      3. args.data la thu muc chua truc tiep cac file .parquet -> tu dong split 95/5
    """
    def __init__(self, args):
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_check)
        self.tokenizer.pad_token = pad
        self.tokenizer.bos_token = bos
        self.tokenizer.eos_token = eos

        task_config = get_task_config(args.task)
        self.labels = task_config.labels
        self.labels_dic = {key: index for index, key in enumerate(self.labels)}
        self.id2labels = {index: key for index, key in enumerate(self.labels)}

        print("Task Labels:", self.labels_dic)

        self.pad_id = len(self.labels_dic)
        
        args.update({
            'pad_index': self.tokenizer.pad_token_id,
            'unk_index': self.tokenizer.unk_token_id,
            'bos_index': self.tokenizer.bos_token_id,
            'eos_index': self.tokenizer.eos_token_id,
            "n_labels": len(self.labels_dic) + 1,
            "n_chars": self.tokenizer.vocab_size + 3,
            "n_bigrams": 3,
            "n_stop_labels": 1,
        })
        
        self.is_crf2 = ('gram' in args.mode)
        
        # --- Tim va gan parquet files cho train/val ---
        train_path = os.path.join(args.data, "train")
        val_path = os.path.join(args.data, "val")
        
        train_files = _find_parquet_files(train_path) if os.path.isdir(train_path) else []
        val_files = _find_parquet_files(val_path) if os.path.isdir(val_path) else []
        
        # Kich ban 3: khong co train/ subdir -> tim .parquet truc tiep trong args.data
        if not train_files:
            root_files = _find_parquet_files(args.data)
            if root_files:
                print(f"[Load] Tim thay {len(root_files)} parquet files truc tiep trong {args.data}")
                # Tu dong split 95/5
                split_idx = max(1, int(len(root_files) * 0.95))
                train_files = root_files[:split_idx]
                val_files = root_files[split_idx:] if split_idx < len(root_files) else root_files[-1:]
                print(f"[Load] Auto-split: {len(train_files)} train files, {len(val_files)} val files")
        
        # Kich ban 2: co train/ nhung khong co val/
        if train_files and not val_files:
            print("[Load] No val/ directory found. Using last train file as validation.")
            val_files = train_files[-1:]       
        
        if not train_files:
            raise FileNotFoundError(
                f"Khong tim thay file .parquet nao trong {args.data}. "
                f"Hay dat cac file .parquet vao {args.data}/ hoac {train_path}/."
            )
        
        print(f"[Load] Train files: {train_files}")
        print(f"[Load] Val files:   {val_files}")

        self.train = ParquetStreamingDataset(
            train_files, self.tokenizer, self.labels_dic, self.pad_id, self.is_crf2
        )
        self.validation = ParquetStreamingDataset(
            val_files, self.tokenizer, self.labels_dic, self.pad_id, self.is_crf2
        )

    @staticmethod
    def pad(tensors, padding_value=0):
        size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                                  for i in range(len(tensors[0].size()))]
        out_tensor = tensors[0].data.new(*size).fill_(padding_value)
        for i, tensor in enumerate(tensors):
            out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
        return out_tensor

    def collate_fn_crf_last(self, batch):
        tokens, bi_chars, bert_input, attention_mask, mask, tags = zip(*batch)
        tokens = self.pad(tokens, padding_value=self.tokenizer.pad_token_id)
        bi_chars = self.pad(bi_chars, padding_value=0)
        bert_input = self.pad(bert_input, padding_value=self.tokenizer.pad_token_id)
        mask = self.pad(mask, padding_value=0).bool()
        attention_mask = self.pad(attention_mask, padding_value=0).bool()
        tags = self.pad(tags, padding_value=self.pad_id)

        return ((tokens.to(self.args.device), bi_chars.to(self.args.device), bert_input.to(self.args.device),
                 attention_mask.to(self.args.device), mask.to(self.args.device)),
                tags.to(self.args.device))

    def collate_fn_bigram(self, batch):
        tokens, bi_chars, bert_input, attention_mask, mask, non_stop_tags, stop_tags = zip(*batch)
        tokens = self.pad(tokens, padding_value=self.tokenizer.pad_token_id)
        bi_chars = self.pad(bi_chars, padding_value=0)
        bert_input = self.pad(bert_input, padding_value=self.tokenizer.pad_token_id)
        mask = self.pad(mask, padding_value=0).bool()
        attention_mask = self.pad(attention_mask, padding_value=0).bool()
        
        non_stop_tags = self.pad(non_stop_tags, padding_value=self.pad_id)
        stop_tags = self.pad(stop_tags, padding_value=self.pad_id)

        return ((tokens.to(self.args.device), bi_chars.to(self.args.device), bert_input.to(self.args.device),
                 attention_mask.to(self.args.device), mask.to(self.args.device)),
                non_stop_tags.to(self.args.device), stop_tags.to(self.args.device))
