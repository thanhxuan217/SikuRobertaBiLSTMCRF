# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
"""
File này định nghĩa lớp `Train_single` kế thừa từ `CMD` in `cmd_single.py`. 
Đảm nhiệm vòng lặp huấn luyện (train loop) qua các epochs dành cho mô hình Single-CRF, 
bao gồm việc nạp dữ liệu từ loader của một task duy nhất, 
thiết lập phân cấp learning rate (classification head học nhanh hơn) và cấu hình optimizer.
"""

from datetime import datetime, timedelta

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
                                   shuffle=True, collate_fn=collate_fn)
        self.devset = DataLoader(dev, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_fn)

        # create the model
        print("Create the model.")

        self.model = self.model_cl(args).to(args.device)
        print(f"{self.model}")

        # Setup Learning Rate
        lr = args.bert_lr
        decay = args.bert_decay

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

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            print(f"Epoch {epoch} / {args.epochs}:")

            self.train(self.trainset)

            loss, metric_punc = self.evaluate(self.devset)
            print(f"{'dev:':6} Loss: {loss:.4f}")
            print('punc', metric_punc)

            t = datetime.now() - start
            # save the model if it is the best so far
            if metric_punc > best_metric and epoch > args.patience // 5:
                best_e, best_metric = epoch, metric_punc
                self.model.save(args.save_model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break

        self.model = self.model_cl.load(args.save_model)
        loss, metric_punc = self.evaluate(self.devset)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the loss of dev at epoch {best_e} is {loss:.2f}")

        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
