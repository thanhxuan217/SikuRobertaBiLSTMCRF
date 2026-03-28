# -*- coding: utf-8 -*-
# @ModuleName: crf_2_model
# @Function: Mô hình mạng nơ-ron khai thác đặc trưng BERT, BiLSTM và giải mã bằng CRF đôi (Dual CRF) để thực hiện đồng thời hai tác vụ: phân mảng câu (segmentation/non-stop) và dự đoán ngắt câu (punctuation/stop).
# @Author: Wxb
# @Time: 2024/2/25 15:21
import torch
import torch.nn as nn
from parsering.modules import MLP, BertEmbedding, BiLSTM, Biaffine, CRF
from parsering.modules.dropout import IndependentDropout, SharedDropout
from parsering.checkpoint import load_checkpoint, restore_args, serialize_args
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)


class bigram_bert_model(nn.Module):
    """
    Mô hình bigram_bert_model kết hợp khả năng biểu diễn ngữ cảnh của BERT, BiLSTM và giải mã chuỗi biểu diễn điểm với hai lớp CRF.
    Đầu vào bao gồm nhúng ký tự (character embedding) và nhúng BERT (BERT embedding) được kết hợp lại.
    Sau đó, đặc trưng đi qua mạng BiLSTM và chia làm hai nhánh MLP + CRF độc lập:
      - Nhánh 1 (non-stop): Dự đoán phân mảnh / phân mảng văn bản.
      - Nhánh 2 (stop): Dự đoán ngắt câu / chèn dấu câu.
    """
    def __init__(self, args):
        """
        Khởi tạo các thành phần lớp kiến trúc của mô hình.
        
        Args:
            args: Đối tượng chứa cấu hình tham số (hyperparameters) như kích thước nhúng, số lớp, tỷ lệ dropout,...
        """
        super(bigram_bert_model, self).__init__()

        self.args = args
        # Biến boolean kiểm tra pretrained embeddings (không liên quan BERT)
        self.pretrained = False
        
        # 1. Tầng Embedding (Nhúng kí tự / Character Embedding)
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_feat_embed)
        n_lstm_input = args.n_feat_embed  # 100
        
        # 2. Tầng BERT Embedding (Lấy đặc trưng phong phú từ Roberta/BERT model)
        self.feat_embed = BertEmbedding(model=args.base_model,
                                        n_layers=args.n_bert_layers,
                                        n_out=args.n_feat_embed)
        n_lstm_input += args.n_feat_embed # Tổng số chiều input cho LSTM

        # Lớp Dropout cho Embedding (để chống overfitting)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # 3. Tầng BiLSTM (Học theo chuỗi với ngữ cảnh 2 chiều trái - phải)
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # 4. Neural Network Head thứ 1: Tầng MLP và lớp suy luận phân mảng CRF
        # Dùng cho task phân cụm/cắt mảng câu
        self.mlp = MLP(n_in=args.n_lstm_hidden * 2,
                       n_out=args.n_labels)

        self.crf = CRF(n_labels=args.n_labels)

        # 5. Neural Network Head thứ 2: MLP và CRF cho ngắt dấu câu
        # Dùng cho task dự đoán ngắt câu/chèn dấu câu cụ thể
        self.mlp2 = MLP(n_in=args.n_lstm_hidden * 2,
                        n_out=args.n_stop_labels)

        self.crf2 = CRF(n_labels=args.n_stop_labels)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def forward(self, feed_dict, target1=None, target2=None, do_predict=False):
        """
        Quá trình lan truyền tiến (Forward Pass).
        Hàm này tính toán đầu ra từ embeddings, truyền qua BiLSTM rồi chia tỷ lệ cho hai nhánh CRF.
        
        Args:
            feed_dict: Từ điển chứa dữ liệu batch đầu vào (gồm 'chars', 'bert', 'crf_mask').
            target1: Tensor chứa nhãn thực tế cho task phân mảng (nhánh non-stop), mặc định None.
            target2: Tensor chứa nhãn thực tế cho task ngắt câu (nhánh stop), mặc định None.
            do_predict: Boolean cờ chỉ định có giải mã tạo ra chuỗi dự đoán (dùng Viterbi) hay không.
            
        Returns:
            stop, non_stop: Kết quả từ hai nhánh (từ điển chứa 'loss' và/hoặc 'predict').
        """
        chars = feed_dict["chars"]
        # Lấy Output từ tầng character embedding
        char_embed = self.char_embed(chars)

        batch_size, seq_len = feed_dict['bert'][0].shape

        mask = feed_dict['bert'][1]
        lens = mask.sum(dim=1).cpu() # Tính độ dài thực của từng câu
        
        # Lấy đặc trưng từ pretrained module (BERT/Roberta)
        feat_embed = self.feat_embed(*feed_dict['bert'])

        # Dropout cho embedded features và gộp hai vector lại
        char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
        feat_embed = torch.cat((char_embed, feat_embed), dim=-1)
        # feat_embed: (Batch Size, Length, Dimension)

        # Pack padded sequence để loại bỏ padding khi cho qua LSTM, cải thiện hiệu suất tính toán
        x = pack_padded_sequence(feat_embed, lens, True, False)
        x, _ = self.lstm(x) # Truyền qua BiLSTM layer
        x, _ = pad_packed_sequence(x, True, total_length=seq_len) # Pad sequence trở lại độ dài cũ sau LSTM
        x = self.lstm_dropout(x)

        # Nhánh 1: Dự đoán phân mảng câu (Non-stop)
        x1 = self.mlp(x) # Chuyển output của LSTM thành vector chiều không gian n_labels (số nhãn phân mảng)
        mask = feed_dict['crf_mask'] # CRF Mask dùng để bỏ qua dự đoán chữ đầu [CLS] và đuôi đuôi [SEP]
        non_stop = self.forward_crf(x1, mask, target1, do_predict, ind=0)

        # Nhánh 2: Dự đoán ngắt câu (Stop)
        x2 = self.mlp2(x) # Chuyển output của LSTM thành logit nhãn ngắt câu
        stop = self.forward_crf(x2, mask, target2, do_predict, ind=1)

        return stop, non_stop

    def forward_crf(self, x, mask, target=None, do_predict=False, ind=0):
        """
        Sub-hàm hỗ trợ xử lý việc tính toán Loss hoặc giải mã dãy dự đoán qua lớp CRF tương ứng.
        
        Args:
            x: Tensor đầu vào chứa logit sinh ra từ lớp MLP.
            mask: Tensor mask để loại bỏ tác động của padding và các token đặc biệt ([CLS], [SEP]).
            target: Tensor nhãn chuẩn (ground truth) dùng để tính Loss khi huấn luyện.
            do_predict: Boolean cho biết liệu có cần sử dụng thuật toán Viterbi để suy luận ra chuỗi nhãn tốt nhất.
            ind: Chỉ số định danh nhánh CRF (0 đại diện nhánh non-stop CRF, 1 đại diện nhánh stop CRF).
            
        Returns:
            ret: Từ điển chứa giá trị 'loss' (nếu có target) và/hoặc 'predict' (các nhãn đã dự đoán).
        """
        # Bỏ đi logit vị trí thứ 0 (token [CLS]) và vị trí -1 (token [SEP]) đối với chuỗi mask
        mask = mask[:, 1: -1]
        x = x[:, 1: -1]

        ret = {}
        # Giai đoạn Training: Tính toán Loss dựa vào Negative Log-Likelihood của CRF
        if target is not None:
            if not ind:
                loss = self.crf(x, target, mask)
            else:
                loss = self.crf2(x, target, mask)
            ret['loss'] = loss
        
        # Giai đoạn Evaluation/Prediction: Áo dụng thuật toán viterbi để tìm đường đi xác suất cao nhất gán nhãn
        if do_predict:
            if not ind:
                predict_labels = self.crf.viterbi(x, mask)
            else:
                predict_labels = self.crf2.viterbi(x, mask)
            ret['predict'] = predict_labels
        return ret

    @classmethod
    def load(cls, path):
        """
        Hàm tiện ích (Class method) dùng để tải mô hình đã lưu từ đường dẫn chỉ định.
        
        Args:
            path: Đường dẫn tệp chứa state_dict và cấu hình của mô hình.
            
        Returns:
            model: Đối tượng mô hình đã được phục hồi trọng số.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = load_checkpoint(path, map_location=device)
        args = restore_args(state['args'])
        model = cls(args)
        # model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        """
        Lưu trạng thái hiện tại của mô hình (bao gồm trọng số, cấu hình tham số) ra một tệp.
        
        Args:
            path: Nơi lưu tệp mô hình trên hệ thống.
        """
        state_dict, pretrained = self.state_dict(), None
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
            'pretrained': pretrained
        }
        torch.save(state, path)
