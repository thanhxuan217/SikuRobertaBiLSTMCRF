# 🏯 SikuRobertaBiLSTMCRF

**Ancient Chinese Punctuation & Sentence Segmentation using RoBERTa + BiLSTM + CRF**

> Based on: [Two Sequence Labeling Approaches to Sentence Segmentation and Punctuation Prediction for Classic Chinese Texts](https://aclanthology.org/2024.lt4hala-1.28) (Wang & Li, LT4HALA-WS 2024)

## 📋 Table of Contents

- [Architecture](#architecture)
- [Dataset Format](#dataset-format)
- [Setup](#setup)
- [Training on Kaggle](#training-on-kaggle-)
- [Training Locally](#training-locally)
- [Resume Training](#resume-training)
- [Prediction](#prediction)
- [Citation](#citation)

---

## Architecture

Mô hình sử dụng kiến trúc **Single-Stage (Single CRF)** để xử lý tất cả các nhãn đấu câu trong một lần chạy duy nhất. Điều này giúp tối ưu tốc độ huấn luyện và suy luận trong khi vẫn đảm bảo độ chính xác cao.

```
Input Text → SikuRoBERTa → BiLSTM → MLP → CRF → Predicted Labels
                 ↓
          Character Embedding
```

### Supported Labels (Punctuation Task)

```
['O', '，', '。', '：', '、', '；', '？', '！']
  │     │     │     │     │     │     │     │
 None Comma Period Colon Dun  Semi  Ques Excl
```

---

## Dataset Format

### Parquet Format (Recommended for Kaggle)

Mỗi file `.parquet` chứa các cột sau:

| Column | Type | Description |
|--------|------|-------------|
| `text` | `string` | Chuỗi văn bản cổ (mỗi ký tự là một token) |
| `labels` | `list[string]` | Nhãn tương ứng cho **mỗi ký tự** trong `text` |
| `domain` | `string` | *(optional)* Lĩnh vực văn bản (VD: 史藏, 經藏) |
| `filename` | `string` | *(optional)* Tên file gốc |

**Ví dụ:**
```json
{
  "text": "太史公曰余登箕山其上蓋有許由冢云",
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "，", "O", "O", "O", "O", "O", "O", "O", "O", "。"],
  "domain": "史藏",
  "filename": "史记.txt"
}
```

> **⚠️ Quan trọng:** `len(text) == len(labels)` — Mỗi ký tự phải có đúng một nhãn.

### Data Directory Structure

Bạn có thể tổ chức dữ liệu theo 1 trong 3 cách:

```
# Cách 1: Tách sẵn train/val (Khuyến nghị)
data/
├── train/
│   ├── part_0.parquet
│   ├── part_1.parquet
│   └── ...
└── val/
    └── part_0.parquet

# Cách 2: Chỉ có train/ (tự động dùng file cuối làm val)
data/
└── train/
    ├── part_0.parquet
    └── part_1.parquet

# Cách 3: Đặt trực tiếp (tự động split 95/5)
data/
├── part_0.parquet
├── part_1.parquet
└── ...
```

---

## Setup

### Prerequisites

- Python >= 3.8
- CUDA GPU (khuyến nghị)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download SikuRoBERTa

Download [SikuRoBERTa](https://github.com/hsc748NLP/SikuBERT-for-digital-humanities-and-classical-Chinese-information-processing) và đặt vào thư mục `SIKU-BERT/`:

```
SIKU-BERT/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

---

## Training on Kaggle 🚀

Dự án mặc định sử dụng **QLoRA (4-bit quantization + LoRA)** để tối ưu bộ nhớ GPU, cho phép huấn luyện model RoBERTa lớn trên các GPU cấu hình thấp như T4.

### Bước 1: Chuẩn bị trên Kaggle

1. **Upload SikuRoBERTa** lên Kaggle dưới dạng **Dataset**
2. **Dataset** của bạn (các file `.parquet`) cũng cần được add vào Kaggle Notebook
3. Bật **Internet** trong Settings và chọn **GPU T4 x2** hoặc **P100**

### Bước 2: Chạy Training

```bash
# Cell 1: Clone repo
!git clone https://github.com/thanhxuan217/SikuRobertaBiLSTMCRF.git
%cd SikuRobertaBiLSTMCRF

# Cell 2: Install deps (bao gồm bitsandbytes và peft cho QLoRA)
!pip install -q transformers datasets pyarrow scikit-learn bitsandbytes peft

# Cell 3: Link model và data
!ln -s /kaggle/input/siku-roberta ./SIKU-BERT
!mkdir -p data/train
!ln -s /kaggle/input/your-dataset-name/*.parquet data/train/

# Cell 4: Train với QLoRA
!python -u run.py train \
    -p \
    --feat=SIKU-BERT \
    --data=data \
    --batch_size=32 \
    --task=punctuation \
    --use_qlora \
    -f=exp/SIKU-BERT.blstm.crf.qlora
```

---

## Training Locally

Tất cả các lượt chạy huấn luyện hiện nay đều được tối ưu hóa với **QLoRA**.

```bash
python -u run.py train \
    -p \
    --feat=SIKU-BERT \
    --data=data \
    --batch_size=50 \
    --task=punctuation \
    --use_qlora \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.local
```

### ⚙️ Các tham số quan trọng

| Param | Mô tả |
|-------|-------|
| `--use_qlora` | **Bắt buộc**. Kích hoạt 4-bit quantization và LoRA adapters. |
| `--batch_size` | Kích thước batch. Giảm xuống nếu gặp lỗi OOM. |
| `--task` | `punctuation` (dấu câu) hoặc `segmentation` (ngắt câu). |
| `--save_steps` | Tần suất lưu checkpoint (mặc định 10,000 steps). |

---

## Resume Training

Nếu quá trình huấn luyện bị gián đoạn (do hết thời gian trên Kaggle hoặc lỗi mạng), bạn có thể tiếp tục từ checkpoint gần nhất. Hệ thống sẽ tự động khôi phục:
- Model weights & LoRA adapters.
- Optimizer & Scheduler states.
- Epoch và Step hiện tại.

```bash
python -u run.py train \
    --feat=SIKU-BERT \
    --data=data \
    --task=punctuation \
    --use_qlora \
    --resume \
    -f=exp/SIKU-BERT.blstm.crf.local
```

> [!TIP]
> Bạn nên đặt `--save_steps` nhỏ hơn (ví dụ: 2000 hoặc 5000) nếu môi trường huấn luyện không ổn định để tránh mất quá nhiều tiến trình.

---

## Prediction

Sử dụng model đã huấn luyện để dự đoán nhãn cho dữ liệu mới.

```bash
python -u run.py predict \
    --feat=SIKU-BERT \
    --data=path/to/test_data \
    --use_qlora \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.local
```

---

## Project Structure

```
SikuRobertaBiLSTMCRF/
├── run.py                    # Entry point
├── config.ini                # Model hyperparameters
├── requirements.txt          # Dependencies
├── data/                     # Training data (.parquet files)
├── parsering/
│   ├── BasePlusModel.py      # roberta_bilstm_crf architecture
│   ├── cmd/
│   │   ├── train_single.py   # Trainer logic
│   │   └── predict_single.py # Predictor logic
│   └── modules/              # Neural network components (BERT, BiLSTM, CRF, MLP)
└── exp/                      # Saved models & LoRA adapters
```

---

## Citation

```bibtex
@inproceedings{wang-li-2024-two,
    title = "Two Sequence Labeling Approaches to Sentence Segmentation and Punctuation Prediction for Classic {C}hinese Texts",
    author = "Wang, Xuebin  and Li, Zhenghua",
    editor = "Sprugnoli, Rachele  and Passarotti, Marco",
    booktitle = "Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024",
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lt4hala-1.28",
    pages = "237--241",
}
```
