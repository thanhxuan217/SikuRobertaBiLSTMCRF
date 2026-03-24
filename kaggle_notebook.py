# ============================================================
# Kaggle Notebook - SikuRobertaBiLSTMCRF (One-Stage Punctuation)
# ============================================================
# Huong dan: Copy toan bo noi dung file nay vao mot cell trong Kaggle Notebook.
# Yeu cau: GPU Runtime (T4 hoac P100), Internet bat.
#
# Dataset Kaggle cua ban can co cot: text, labels
# VD: {"text": "...", "labels": ["O", "O", "，"], "domain": "...", "filename": "..."}
# Labels = ['O', '，', '。', '：', '、', '；', '？', '！']
# ============================================================

# %% [markdown]
# # 🏯 SikuRobertaBiLSTMCRF - Classical Chinese Punctuation
# **One-Stage Approach**: RoBERTa + BiLSTM + CRF

# %% --- Cell 1: Install Dependencies ---
# !pip install -q transformers datasets pyarrow scikit-learn

# %% --- Cell 2: Clone repo & download model ---
import os
import subprocess

# === CONFIGURATION ===
# Doi KAGGLE_DATASET_DIR thanh duong dan dataset cua ban tren Kaggle
# Vi du: /kaggle/input/your-dataset-name
KAGGLE_DATASET_DIR = "/kaggle/input/your-dataset-name"

# Duong dan SikuRoBERTa model tren Kaggle (upload model nay nhu 1 Kaggle Dataset)
# Hoac dung tu HuggingFace: "SIKU-BERT/sikuroberta"
SIKU_BERT_PATH = "/kaggle/input/siku-roberta"

# Duong dan lam viec
WORK_DIR = "/kaggle/working"
REPO_DIR = os.path.join(WORK_DIR, "SikuRobertaBiLSTMCRF")

# Clone repository
if not os.path.exists(REPO_DIR):
    subprocess.run([
        "git", "clone", 
        "https://github.com/thanhxuan217/SikuRobertaBiLSTMCRF.git",
        REPO_DIR
    ], check=True)
    print("✅ Cloned repository")
else:
    print("✅ Repository already exists")

# %% --- Cell 3: Setup data directory ---
import shutil
import glob

DATA_DIR = os.path.join(REPO_DIR, "data")

# Tim tat ca file parquet trong Kaggle dataset
parquet_files = sorted(glob.glob(os.path.join(KAGGLE_DATASET_DIR, "**/*.parquet"), recursive=True))
print(f"📊 Found {len(parquet_files)} parquet files in dataset")

if not parquet_files:
    # Thu tim truc tiep trong /kaggle/input/
    parquet_files = sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True))
    print(f"📊 Found {len(parquet_files)} parquet files in /kaggle/input/")

# Tao cau truc thu muc data/train/ va data/val/
os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "val"), exist_ok=True)

# Split: 95% train, 5% val
split_idx = max(1, int(len(parquet_files) * 0.95))
train_files = parquet_files[:split_idx]
val_files = parquet_files[split_idx:] if split_idx < len(parquet_files) else parquet_files[-1:]

print(f"🔀 Split: {len(train_files)} train files, {len(val_files)} val files")

# Symlink (nhanh hon copy)
for f in train_files:
    dest = os.path.join(DATA_DIR, "train", os.path.basename(f))
    if not os.path.exists(dest):
        os.symlink(f, dest)

for f in val_files:
    dest = os.path.join(DATA_DIR, "val", os.path.basename(f))
    if not os.path.exists(dest):
        os.symlink(f, dest)

print("✅ Data directory ready")
print(f"   Train: {os.listdir(os.path.join(DATA_DIR, 'train'))[:3]}...")
print(f"   Val:   {os.listdir(os.path.join(DATA_DIR, 'val'))}")

# %% --- Cell 4: Setup SikuRoBERTa model ---
# Tao symlink den model SikuRoBERTa
siku_bert_dest = os.path.join(REPO_DIR, "SIKU-BERT")
if not os.path.exists(siku_bert_dest):
    os.symlink(SIKU_BERT_PATH, siku_bert_dest)
    print(f"✅ Linked SikuRoBERTa: {SIKU_BERT_PATH} -> {siku_bert_dest}")
else:
    print(f"✅ SikuRoBERTa already linked")

# Verify model files
required_files = ["config.json", "tokenizer.json"]
for f in required_files:
    path = os.path.join(siku_bert_dest, f)
    if os.path.exists(path):
        print(f"   ✅ {f}")
    else:
        print(f"   ⚠️ Missing: {f} - Check your SIKU_BERT_PATH!")

# %% --- Cell 5: Create required directories ---
os.makedirs(os.path.join(REPO_DIR, "exp"), exist_ok=True)
os.makedirs(os.path.join(REPO_DIR, "log", "train"), exist_ok=True)
# Tao dataset directory voi dummy json (vi code co the can doc)
os.makedirs(os.path.join(REPO_DIR, "dataset"), exist_ok=True)

# Tao dummy train.json va dev.json neu chua co (load_streaming.py khong can nhung de an toan)
import json
for fname in ["train.json", "dev.json"]:
    fpath = os.path.join(REPO_DIR, "dataset", fname)
    if not os.path.exists(fpath):
        with open(fpath, "w") as f:
            json.dump([], f)

print("✅ Directory structure ready")

# %% --- Cell 6: Train! ---
print("=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

# Cau hinh training
BATCH_SIZE = 32       # Giam neu gap OOM (Out of Memory)
DEVICE = "0"          # GPU index
FEAT = "SIKU-BERT"
TASK = "punctuation"  # Quan trong: phai la "punctuation" cho bai toan dau cau

os.chdir(REPO_DIR)

train_cmd = [
    "python", "-u", "run.py", "train_single",
    "-p",
    f"--feat={FEAT}",
    f"--data={DATA_DIR}",
    f"--batch_size={BATCH_SIZE}",
    f"--task={TASK}",
    f"-d={DEVICE}",
    f"-f=exp/{FEAT}.blstm.crf.kaggle",
]

print(f"Command: {' '.join(train_cmd)}")
print()

# Run training
process = subprocess.run(train_cmd, cwd=REPO_DIR)

if process.returncode == 0:
    print("\n✅ Training completed successfully!")
else:
    print(f"\n❌ Training failed with return code {process.returncode}")

# %% --- Cell 7: Save model to output ---
# Copy trained model to /kaggle/working/ for download
model_path = os.path.join(REPO_DIR, f"exp/{FEAT}.blstm.crf.kaggle/model.pth")
if os.path.exists(model_path):
    output_path = os.path.join(WORK_DIR, "model.pth")
    shutil.copy2(model_path, output_path)
    print(f"✅ Model saved to {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
else:
    print("⚠️ Model file not found. Training may not have completed.")
