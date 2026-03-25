# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
"""
File này định nghĩa lớp `Train` kế thừa từ `CMD` in `cmd_gram.py`. 
Đảm nhiệm vòng lặp huấn luyện (train loop) cho mô hình Hai-CRF (Bigram BERT Model) 
bao gồm: cấu hình dữ liệu DataLoader, khởi tạo optimizer/learning rate scheduler, 
và thực hiện fit qua các epochs cùng cơ chế early stopping.
"""

from datetime import datetime, timedelta


from .cmd_gram import CMD

from ..utils.load_streaming import Load
from ..utils.metric import Metric

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader


class Train(CMD):
    """
    Lớp Train kế thừa từ CMD, dùng để huấn luyện mô hình hai-CRF (phân mảng và ngắt câu).
    """

    def add_subparser(self, name, parser):
        # Thêm các tham số dòng lệnh cho quá trình huấn luyện
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--data', default='data/ptb_pos',
                               help='path to train file') # Đường dẫn tới thư mục chứa dữ liệu huấn luyện
        return subparser

    def __call__(self, args):
        # Khởi tạo các cấu hình chung từ lớp cha CMD
        super(Train, self).__call__(args)
        print('Preprocess the data')
        
        # Load object chịu trách nhiệm đọc dữ liệu (ví dụ: parquet) và xây dựng từ vựng
        loader = Load(args)

        # Hàm collate_fn_bigram được dùng để gom các mẫu dữ liệu thành một batch
        collate_fn = loader.collate_fn_bigram
        print(f"{args}")
        
        # Lấy tập dữ liệu huấn luyện (train) và tập xác thực (validation/dev)
        train, dev = loader.train, loader.validation
        # print(len(train))
        
        # Khởi tạo DataLoader để tải dữ liệu huấn luyện và xác thực theo từng batch
        self.trainset = DataLoader(train, batch_size=args.batch_size,
                                   collate_fn=collate_fn)
        self.devset = DataLoader(dev, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

        # create the model
        print("Create the model.")

        # Khởi tạo mô hình (model_cl đã được set là bigram_bert_model ở trong CMD)
        # Đưa mô hình vào thiết bị (CPU hoặc GPU) thông qua args.device
        self.model = self.model_cl(args).to(args.device)
        print(f"{self.model}")

        # Tốc độ học cho BERT và các tầng LSTM
        lr = args.bert_lr
        decay = args.bert_decay

        # Tách riêng các tham số của BERT và BiLSTM (những thứ không phải CRF hoặc MLP)
        bert_lstm_params = [param for name, param in self.model.named_parameters()
                            if 'crf' not in name and 'mlp' not in name]
        
        # Thiết lập tốc độ học (learning rate) lớn hơn cho phần CRF và MLP ở trên cùng (classification head)
        times, weight_decay = 10, 0.01
        crf_params = [
            {'params': self.model.crf.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            {'params': self.model.crf2.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            {'params': self.model.mlp.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
            {'params': self.model.mlp2.parameters(), 'lr': lr * times, 'weight_decay': weight_decay},
        ]

        # Trình tối ưu hóa Adam, gộp cả tham số của BERT/LSTM và CRF/MLP
        self.optimizer = Adam([{'params': bert_lstm_params}] + crf_params,
                              lr,
                              (args.mu, args.nu),
                              args.epsilon)

        # Trình giảm tốc độ học (learning rate scheduler)
        decay_steps = args.decay_epochs * len(self.trainset)
        self.scheduler = ExponentialLR(self.optimizer,
                                       decay ** (1 / decay_steps))
        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        # Bắt đầu vòng lặp huấn luyện qua các epoch
        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            print(f"Epoch {epoch} / {args.epochs}:")

            # Gọi hàm train từ lớp cha CMD để thực hiện lan truyền xuôi/ngược cho epoch này
            self.train(self.trainset)
            
            # Đánh giá mô hình trên tập dev/validation sau mỗi epoch
            loss, metric_non_s, metric_stop = self.evaluate(self.devset)
            print(f"{'dev:':6} Loss: {loss:.4f}")
            print('punc', metric_stop)

            t = datetime.now() - start
            # save the model if it is the best so far
            # Lưu lại model nếu kết quả đánh giá (metric_stop) trên tập dev tốt nhất từ trước tới nay
            if metric_stop > best_metric and epoch > args.patience // 5:
                best_e, best_metric = epoch, metric_stop
                self.model.save(args.save_model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            
            # Nếu mô hình không cải thiện sau 'patience' epochs thì dừng sớm (Early Stopping)
            if epoch - best_e >= args.patience:
                break

        self.model = self.model_cl.load(args.save_model)
        loss, metric_non_s, metric_stop = self.evaluate(self.devset)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the loss of dev at epoch {best_e} is {loss:.2f}")

        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
