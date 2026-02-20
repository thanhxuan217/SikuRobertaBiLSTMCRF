import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ..utils.common import pad, bos, eos
from parsering.task_config import get_task_config


class ParquetStreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, split, tokenizer, labels_dic, pad_id, is_crf2=False):
        self.dataset = load_dataset('parquet', data_dir=data_dir, split=split, streaming=True)
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
            
            # For character BERT, we just use the tokenizer vocab directly 
            # (In the original script, chars2ids mapped characters to a custom vocab)
            # The custom vocab had ~20k chars. Now we use tokenizer's vocab.
            try:
                words_index = self.tokenizer.convert_tokens_to_ids([self.bos] + text[:length - 2] + [self.eos])
            except:
                words_index = []
                for c in [self.bos] + text[:length - 2] + [self.eos]:
                    words_index.append(self.tokenizer.vocab.get(c, self.tokenizer.unk_token_id))
            words_index = torch.tensor(words_index, dtype=torch.long)
            
            # Dummy bi_chars (we disable bigram embeddings since they were commented out in the original model)
            bi_words_index = torch.zeros(length, dtype=torch.long)
            
            attention_mask = torch.ones(length).gt(0)
            mask = torch.tensor([0] + [1] * (length - 2) + [0])
            
            # Tags to IDs padding properly
            # if punctuation task, there may be multiple tags for some characters if it's double tag? 
            # The user dataset provides matching shapes for text and labels arrays.
            mapped_tags = [self.labels_dic.get(e, self.pad_id) for e in labels[:length-2]]
            tags = torch.tensor(mapped_tags, dtype=torch.long)
            
            if self.is_crf2:
                yield (words_index, bi_words_index, bert_input, attention_mask, mask, tags, tags)
            else:
                yield (words_index, bi_words_index, bert_input, attention_mask, mask, tags)


class Load:
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
            "n_chars": self.tokenizer.vocab_size + 3,  # safety margin
            "n_bigrams": 3,
            "n_stop_labels": 1,
        })
        
        # Determine if we're yielding a 2-tag element (for gram models)
        self.is_crf2 = ('gram' in args.mode)
        
        self.train_dataset = ParquetStreamingDataset('data', 'train', self.tokenizer, self.labels_dic, self.pad_id, self.is_crf2)
        try:
            self.val_dataset = ParquetStreamingDataset('data', 'validation', self.tokenizer, self.labels_dic, self.pad_id, self.is_crf2)
        except:
            print("Validation dataset stream not found. using train temporarily")
            self.val_dataset = self.train_dataset
            
        self.train = self.train_dataset
        self.validation = self.val_dataset

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
