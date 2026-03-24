# 🏯 SikuRobertaBiLSTMCRF

**Ancient Chinese Punctuation & Sentence Segmentation using RoBERTa + BiLSTM + CRF**

> Based on: [Two Sequence Labeling Approaches to Sentence Segmentation and Punctuation Prediction for Classic Chinese Texts](https://aclanthology.org/2024.lt4hala-1.28) (Wang & Li, LT4HALA-WS 2024)

## 📋 Table of Contents

- [Architecture](#architecture)
- [Dataset Format](#dataset-format)
- [Setup](#setup)
- [Training on Kaggle](#training-on-kaggle-)
- [Training Locally](#training-locally)
- [Prediction](#prediction)
- [Citation](#citation)

---

## Architecture

The project supports **two approaches**:

| Approach | Mode | Description |
|----------|------|-------------|
| **One-Stage (Single CRF)** | `train_single` | Một CRF duy nhất xử lý tất cả nhãn dấu câu. Đơn giản, nhanh hơn. |
| **Two-Stage (Dual CRF)** | `train` | Hai CRF riêng biệt: một cho dấu ngắt câu, một cho dấu ngoặc. Phức tạp hơn. |

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

### Bước 1: Chuẩn bị trên Kaggle

1. **Upload SikuRoBERTa** lên Kaggle dưới dạng **Dataset** (ví dụ: `siku-roberta`)
2. **Dataset** của bạn (các file `.parquet`) cũng cần được add vào Kaggle Notebook
3. Tạo một **Kaggle Notebook mới** với **GPU T4 x2** hoặc **P100**
4. Bật **Internet** trong Settings

### Bước 2: Thêm Input Datasets

Trong Kaggle Notebook, add 2 datasets:
- Dataset parquet của bạn (VD: `your-username/your-dataset`)
- SikuRoBERTa model (VD: `your-username/siku-roberta`)

### Bước 3: Chạy Training

Copy nội dung file [`kaggle_notebook.py`](kaggle_notebook.py) vào notebook, **sửa 2 biến sau**:

```python
# Đổi theo đường dẫn thực tế trên Kaggle
KAGGLE_DATASET_DIR = "/kaggle/input/your-dataset-name"
SIKU_BERT_PATH = "/kaggle/input/siku-roberta"
```

**Hoặc** chạy trực tiếp qua command line trong cell:

```bash
# Cell 1: Clone repo
!git clone https://github.com/thanhxuan217/SikuRobertaBiLSTMCRF.git
%cd SikuRobertaBiLSTMCRF

# Cell 2: Install deps
!pip install -q transformers datasets pyarrow scikit-learn

# Cell 3: Link model
!ln -s /kaggle/input/siku-roberta ./SIKU-BERT

# Cell 4: Setup data  
!mkdir -p data/train data/val
!ln -s /kaggle/input/your-dataset-name/*.parquet data/train/
# Hoac copy 1 file lam validation:
# !cp /kaggle/input/your-dataset-name/last_file.parquet data/val/

# Cell 5: Train (One-Stage Punctuation)
!python -u run.py train_single \
    -p \
    --feat=SIKU-BERT \
    --data=data \
    --batch_size=32 \
    --task=punctuation \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.kaggle
```

### Bước 4: Lưu model

Sau khi train xong, model được lưu tại:
```
exp/SIKU-BERT.blstm.crf.kaggle/model.pth
```

Download file này để sử dụng cho prediction.

### ⚙️ Tham số có thể điều chỉnh

| Param | Default | Mô tả |
|-------|---------|-------|
| `--batch_size` | 32 | Giảm xuống 16 hoặc 8 nếu gặp OOM |
| `--task` | `punctuation` | Đổi sang `segmentation` cho bài toán phân đoạn từ |
| `--data` | `data` | Đường dẫn đến thư mục chứa parquet |
| `-d` | `0` | GPU device ID |

### ❗ Xử lý lỗi thường gặp trên Kaggle

| Lỗi | Giải pháp |
|-----|-----------|
| `CUDA out of memory` | Giảm `--batch_size` (32 → 16 → 8) |
| `FileNotFoundError: SIKU-BERT` | Kiểm tra symlink `ln -s` đến model |
| `No parquet files found` | Kiểm tra đường dẫn dataset, thử `!ls /kaggle/input/` |
| `ModuleNotFoundError: datasets` | Chạy `!pip install datasets` |

---

## Training Locally

### One-Stage (Single CRF) — Recommended

```bash
python -u run.py train_single \
    -p \
    --feat=SIKU-BERT \
    --data=data \
    --batch_size=50 \
    --task=punctuation \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.single
```

### Two-Stage (Dual CRF)

```bash
python -u run.py train \
    -p \
    --feat=SIKU-BERT \
    --data=data \
    --batch_size=50 \
    --task=punctuation \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.dual
```

### With SLURM

```bash
sbatch train_single.slurm    # One-Stage
sbatch train.slurm            # Two-Stage
```

---

## Prediction

```bash
# One-Stage
python -u run.py predict_single \
    --feat=SIKU-BERT \
    --data=path/to/test_data \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.single

# Two-Stage
python -u run.py predict \
    --feat=SIKU-BERT \
    --data=path/to/test_data \
    -d=0 \
    -f=exp/SIKU-BERT.blstm.crf.dual
```

---

## Project Structure

```
SikuRobertaBiLSTMCRF/
├── run.py                    # Entry point
├── config.ini                # Model hyperparameters
├── requirements.txt          # Dependencies
├── kaggle_notebook.py        # Kaggle training script
├── SIKU-BERT/                # Pre-trained SikuRoBERTa (download separately)
├── data/                     # Training data (.parquet files)
│   ├── train/
│   └── val/
├── dataset/                  # LLM-generated supplementary data
│   ├── train.json
│   └── dev.json
├── parsering/
│   ├── BasePlusModel.py      # roberta_bilstm_crf model
│   ├── config.py             # Config parser
│   ├── task_config.py        # Task-specific label definitions
│   ├── cmd/
│   │   ├── cmd_single.py     # Single CRF command base
│   │   ├── cmd_gram.py       # Dual CRF command base
│   │   ├── train_single.py   # One-stage trainer
│   │   ├── train_gram.py     # Two-stage trainer
│   │   ├── predict_single.py # One-stage predictor
│   │   └── predict_gram.py   # Two-stage predictor
│   ├── modules/              # Neural network components
│   │   ├── bert.py           # BERT embedding layer
│   │   ├── bilstm.py         # BiLSTM layer
│   │   ├── crf.py            # CRF layer
│   │   └── mlp.py            # MLP layer
│   └── utils/
│       ├── common.py         # Constants & punctuation maps
│       ├── load.py           # Original data loader (txt/corpus)
│       ├── load_single.py    # Single-task data loader (txt)
│       ├── load_streaming.py # Streaming parquet loader ⭐
│       └── metric.py         # Evaluation metrics
└── exp/                      # Saved models (created during training)
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
