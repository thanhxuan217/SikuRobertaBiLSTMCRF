# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 18:16

import os
import torch
import torch.nn as nn
from parsering.modules import BertEmbedding, BiLSTM, MLP, CRF
from parsering.modules.dropout import IndependentDropout, SharedDropout
from parsering.checkpoint import load_checkpoint, restore_args, serialize_args

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class roberta_bilstm_crf(nn.Module):
    def __init__(self, args):
        super(roberta_bilstm_crf, self).__init__()

        self.args = args
        self.pretrained = False
        self.use_qlora = getattr(args, 'use_qlora', False)
        
        # 1. Tầng Embedding (Nhúng kí tự / Character Embedding)
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed  # Tổng số chiều input cho LSTM

        # 2. Tầng BERT Embedding (Lấy đặc trưng phong phú từ Roberta/BERT model)
        # Nếu dùng QLoRA, truyền quantization config để load BERT ở dạng 4-bit
        quantization_config = None
        if self.use_qlora:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        self.feat_embed = BertEmbedding(model=args.base_model,
                                        n_layers=args.n_bert_layers,
                                        n_out=args.n_feat_embed,
                                        quantization_config=quantization_config)
        n_lstm_input += args.n_feat_embed

        # Áp dụng LoRA adapters lên BERT nếu dùng QLoRA
        if self.use_qlora:
            self._apply_lora(args)

        # Lớp Dropout cho Embedding (để chống overfitting)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # 3. Tầng BiLSTM (Học theo chuỗi với ngữ cảnh chiều trái - phải và phải - trái)
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # 4. Neural Network Head: MLP và CRF (Chỉ có một nhánh cho single-task)
        # Sử dụng cho bài toán phân mảng câu hoặc nhận dạng thực thể
        self.mlp = MLP(n_in=args.n_lstm_hidden*2,
                       n_out=args.n_labels)

        # Lớp phân loại chuỗi CRF (Conditional Random Field)
        self.crf = CRF(n_labels=args.n_labels)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def _apply_lora(self, args):
        """Áp dụng LoRA adapters lên BERT backbone cho QLoRA fine-tuning."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Chuẩn bị model cho k-bit training (enable gradient cho input embeddings, etc.)
        self.feat_embed.bert = prepare_model_for_kbit_training(self.feat_embed.bert)
        
        lora_config = LoraConfig(
            r=getattr(args, 'lora_r', 16) or 16,
            lora_alpha=getattr(args, 'lora_alpha', 32) or 32,
            lora_dropout=getattr(args, 'lora_dropout', 0.05) or 0.05,
            bias="none",
            target_modules=["query", "value"],  # Áp dụng LoRA vào attention layers
            task_type="FEATURE_EXTRACTION",
        )
        
        self.feat_embed.bert = get_peft_model(self.feat_embed.bert, lora_config)
        self.feat_embed.bert.print_trainable_parameters()

    def forward(self, feed_dict, target=None, do_predict=False):
        """
        Quá trình lan truyền tiến (Forward Pass) với một nhánh CRF duy nhất.
        """
        chars = feed_dict["chars"]
        char_embed = self.char_embed(chars)

        batch_size, seq_len = feed_dict['bert'][0].shape

        # get outputs from embedding layers
        # Lấy mask và tính độ dài thực của từng câu để phục vụ pack_padded_sequence
        mask = feed_dict['bert'][1]
        lens = mask.sum(dim=1).cpu()

        # Lấy đặc trưng từ pretrained module (BERT/Roberta)
        feat_embed = self.feat_embed(*feed_dict['bert'])

        # Gộp embedding kí tự và embedding từ BERT
        char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
        feat_embed = torch.cat((char_embed, feat_embed), dim=-1)
        # embed: (Batch Size, Length, Dimension)

        # Cho qua lớp đóng gói padding và LSTM
        x = pack_padded_sequence(feat_embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # Đưa qua MLP (Multi-Layer Perceptron)
        x = self.mlp(x)
        
        # Xử lý mask CRF (loại bỏ vị trí đầu CLS và cuối SEP)
        mask = feed_dict['crf_mask']
        mask = mask[:, 1: -1]
        x = x[:, 1: -1]

        ret = {}
        # Pha Huấn Luyện (Training/Validation): Tính độ lỗi (loss)
        if target is not None:
            loss = self.crf(x, target, mask)
            ret['loss'] = loss
            
        # Pha Dự Đoán (Inference/Testing): Dùng thuật toán Viterbi để xuất nhãn tối ưu
        if do_predict:
            predict_labels = self.crf.viterbi(x, mask)
            ret['predict'] = predict_labels

        return ret

    @classmethod
    def load(cls, path, base_model=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = load_checkpoint(path, map_location=device)
        args = restore_args(state['args'])
        
        # Override base_model nếu được chỉ định (vì checkpoint có thể lưu đường dẫn cũ)
        if base_model is not None:
            args.base_model = base_model
        
        # Khi load, tắt QLoRA để tránh re-quantize
        # LoRA weights đã được merge hoặc load riêng
        original_qlora = getattr(args, 'use_qlora', False)
        args.use_qlora = False
        
        model = cls(args)
        
        # Load LoRA adapter weights nếu có
        base_name = os.path.splitext(os.path.basename(path))[0]
        lora_dir = os.path.join(os.path.dirname(path), f'lora_adapters_{base_name}')
        # Fallback cho checkpoint cũ
        if not os.path.exists(lora_dir):
            old_lora_dir = os.path.join(os.path.dirname(path), 'lora_adapters')
            if os.path.exists(old_lora_dir):
                lora_dir = old_lora_dir

        if original_qlora and os.path.exists(lora_dir):
            print(f"[QLoRA Inference] Tự động tải LoRA adapters từ {lora_dir}")
            from peft import PeftModel
            model.feat_embed.bert = PeftModel.from_pretrained(
                model.feat_embed.bert, lora_dir
            )
            # Merge LoRA weights để inference nhanh hơn
            print(f"[QLoRA Inference] Đang gộp (merge) trọng số LoRA vào backbone...")
            model.feat_embed.bert = model.feat_embed.bert.merge_and_unload()
            # Load phần còn lại (non-BERT weights)
            non_bert_state = {k: v for k, v in state['state_dict'].items() 
                            if 'feat_embed.bert' not in k}
            model.load_state_dict(non_bert_state, strict=False)
            print(f"[QLoRA Inference] Tải trọng số QLoRA thành công!")
        else:
            model.load_state_dict(state['state_dict'], False)
        
        model.to(device)
        return model

    def save(self, path, **kwargs):
        state_dict, pretrained = self.state_dict(), None
        
        if self.use_qlora:
            # Lưu LoRA adapters riêng theo tên file (VD: lora_adapters_model_best)
            base_name = os.path.splitext(os.path.basename(path))[0]
            lora_dir = os.path.join(os.path.dirname(path), f'lora_adapters_{base_name}')
            self.feat_embed.bert.save_pretrained(lora_dir)
            print(f"LoRA adapters saved to {lora_dir}")
            
            # Lưu các weights khác (không phải BERT) vào state_dict chính
            non_bert_state = {k: v for k, v in state_dict.items() 
                            if 'feat_embed.bert' not in k}
            state = {
                'args': serialize_args(self.args),
                'state_dict': non_bert_state,
                'pretrained': pretrained,
                **kwargs
            }
        else:
            if self.pretrained:
                pretrained = {'embed': state_dict.pop('char_pretrained.weight')}
                if hasattr(self, 'bi_pretrained'):
                    pretrained.update(
                        {'bi_embed': state_dict.pop('bi_pretrained.weight')})
                if hasattr(self, 'tri_pretrained'):
                    pretrained.update(
                        {'tri_embed': state_dict.pop('tri_pretrained.weight')})
            state = {
                'args': serialize_args(self.args),
                'state_dict': state_dict,
                'pretrained': pretrained,
                **kwargs
            }
        torch.save(state, path)


