# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
"""
File này định nghĩa lớp `Train_single` kế thừa từ `CMD` in `cmd_single.py`. 
Đảm nhiệm vòng lặp huấn luyện (train loop) qua các epochs dành cho mô hình Single-CRF, 
bao gồm việc nạp dữ liệu từ loader của một task duy nhất, 
thiết lập phân cấp learning rate (classification head học nhanh hơn) và cấu hình optimizer.
Hỗ trợ QLoRA: khi bật --use_qlora, dùng paged_adamw_8bit và chỉ optimize LoRA + head params.
"""

from datetime import datetime, timedelta
import os

from .cmd_single import CMD

from ..utils.load_streaming import Load
from ..utils.metric import Metric

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader


class Train_single(CMD):
    """
    Lớp Trainer dành cho Mô hình Single-CRF
    (Sử dụng roberta_bilstm_crf thay vì bigram_bert_model).
    Hỗ trợ QLoRA fine-tuning khi args.use_qlora = True.
    """

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--data', default='data/ptb_pos',
                               help='path to train file')
        return subparser

    def __call__(self, args):
        super(Train_single, self).__call__(args)
        print('Preprocess the data')
        loader = Load(args) # Gọi đối tượng tiền xử lý file từ "utils/load_single.py"
        
        # Collate (gom batch) version single model
        collate_fn = loader.collate_fn_crf_last
        
        print(f"{args}")
        train, dev = loader.train, loader.validation
        # print(len(train))
        self.trainset = DataLoader(train, batch_size=args.batch_size,
                                   collate_fn=collate_fn)
        self.devset = DataLoader(dev, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

        # create the model
        print("Create the model.")

        self.model = self.model_cl(args).to(args.device)
        print(f"{self.model}")

        use_qlora = getattr(args, 'use_qlora', False)
        
        # Setup Learning Rate
        lr = args.bert_lr
        decay = args.bert_decay

        if use_qlora:
            # === QLoRA Mode ===
            # BERT backbone đã frozen & quantized, chỉ optimize:
            # 1. LoRA adapter params (trong BERT)
            # 2. BiLSTM, char_embed, projection params
            # 3. MLP + CRF head params (learning rate cao hơn)
            import bitsandbytes as bnb
            
            # Thu thập LoRA params (trainable params trong BERT)
            lora_params = [p for p in self.model.feat_embed.bert.parameters() if p.requires_grad]
            
            # Thu thập non-BERT trainable params (char_embed, projection, lstm)
            non_head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'feat_embed.bert' in name:
                    continue  # Đã xử lý ở trên
                if 'crf' in name or 'mlp' in name:
                    continue  # Xử lý riêng bên dưới
                non_head_params.append(param)
            
            # Classification head params (MLP + CRF) - LR cao hơn
            times, weight_decay = 10, 0.01
            head_params = [
                {'params': self.model.crf.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
                {'params': self.model.mlp.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            ]
            
            # Dùng Paged AdamW 8-bit để tiết kiệm VRAM cho optimizer states
            self.optimizer = bnb.optim.PagedAdamW8bit(
                [
                    {'params': lora_params, 'lr': lr},
                    {'params': non_head_params, 'lr': lr},
                ] + head_params,
                lr=lr,
                betas=(args.mu, args.nu),
                eps=args.epsilon,
            )
            
            # In thông tin trainable params
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[QLoRA] Total params: {total_params:,} | Trainable: {trainable_params:,} "
                  f"({100 * trainable_params / total_params:.2f}%)")
        else:
            # === Normal Mode (không QLoRA) ===
            # Chia thành 2 nhóm thông số: 1 là của cốt lõi BERT/LSTM, 2 là mô-đun MLP/CRF phía trên cùng
            bert_lstm_params = [param for name, param in self.model.named_parameters()
                                if 'crf' not in name and 'mlp' not in name]
                                
            # Đầu não phân loại (Classification Head) được cấp quyền học nhanh gấp 10 lần (`times=10`)
            times, weight_decay = 10, 0.01
            crf_params = [
                {'params': self.model.crf.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
                {'params': self.model.mlp.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            ]

            # Optimizer dùng Adam
            self.optimizer = Adam([{'params': bert_lstm_params}] + crf_params,
                                  lr,
                                  (args.mu, args.nu),
                                  args.epsilon)

        decay_steps = args.decay_epochs * len(self.trainset)
        self.scheduler = ExponentialLR(self.optimizer,
                                       decay ** (1 / decay_steps))
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        saved_best_model = False

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            print(f"Epoch {epoch} / {args.epochs}:")

            train_stats = self.train(self.trainset)
            print(
                f"{'train:':6} Loss: {train_stats['avg_loss']:.4f} | "
                f"Time: {timedelta(seconds=int(train_stats['elapsed_seconds']))}"
            )

            loss, metric_punc = self.evaluate(self.devset)
            print(f"{'dev:':6} Loss: {loss:.4f}")
            print('punc', metric_punc)

            t = datetime.now() - start
            # save the model if it is the best so far
            if metric_punc > best_metric and epoch > args.patience // 5:
                best_e, best_metric = epoch, metric_punc
                self.model.save(args.save_model)
                saved_best_model = True
                print(f"{t} elapsed (saved)\n")
            else:
                print(f"{t} elapsed\n")
            total_time += t

            completed_ratio = epoch / args.epochs * 100
            avg_epoch_time = total_time / epoch
            remaining_epochs = args.epochs - epoch
            eta_remaining = avg_epoch_time * remaining_epochs
            print(
                f"[progress] epoch {epoch}/{args.epochs} ({completed_ratio:.1f}%) | "
                f"total_elapsed={total_time} | est_remaining={eta_remaining}\n"
            )

            if epoch - best_e >= args.patience:
                break

        if not saved_best_model:
            print(
                f"No best checkpoint was saved during training. "
                f"Saving the final model to {args.save_model}."
            )
            self.model.save(args.save_model)
            saved_best_model = True
            best_e = epoch

        if os.path.exists(args.save_model):
            self.model = self.model_cl.load(args.save_model)
        else:
            print(
                f"Warning: checkpoint '{args.save_model}' was not found after training. "
                f"Using the in-memory model for final evaluation."
            )
        loss, metric_punc = self.evaluate(self.devset)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the loss of dev at epoch {best_e} is {loss:.2f}")

        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
