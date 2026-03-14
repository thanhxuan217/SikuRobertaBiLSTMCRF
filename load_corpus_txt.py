# -*- coding: utf-8 -*-
"""
load_corpus_txt.py: Đọc dữ liệu từ thư mục corpus_txt/ theo cách lazy loading.

Hỗ trợ 2 cách sử dụng:
1. Generator function: corpus_generator()
2. PyTorch Dataset class: CorpusTxtDataset

Chỉ xử lý 7 dấu câu on-stage (theo source code gốc):
    ，。：、；？！
"""

from pathlib import Path
from typing import Generator, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ── 7 dấu câu on-stage (khớp với self.stop trong load.py) ──────────────────
STOP_PUNCS = {'，', '。', '：', '、', '；', '？', '！'}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Generator (yield từng file)
# ═══════════════════════════════════════════════════════════════════════════════

def corpus_generator(
    root_dir: str = "corpus_txt",
    encoding: str = "utf-8",
) -> Generator[Dict, None, None]:
    """
    Duyệt toàn bộ corpus_txt/ và yield từng file dưới dạng dict.

    Yields:
        dict với các key:
            - folder_id  : str  (tên thư mục con, vd "100003")
            - file_id    : str  (tên file không đuôi, vd "1")
            - path       : str  (đường dẫn đầy đủ)
            - text        : str  (nội dung file gốc)
            - clean_text  : str  (chỉ giữ chữ, bỏ mọi dấu câu on-stage)
            - chars       : list[str]  (danh sách ký tự sau khi bỏ dấu câu)
            - stop_tags   : list[str]  (nhãn on-stage cho từng char: O hoặc dấu câu)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

    # Sắp xếp thư mục con theo số
    for folder in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not folder.is_dir():
            continue

        folder_id = folder.name

        # Sắp xếp file theo số
        for txt_file in sorted(folder.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            file_id = txt_file.stem

            text = txt_file.read_text(encoding=encoding, errors="replace")

            # ── Xử lý on-stage (chỉ gắn nhãn 7 dấu câu) ──────────────
            chars, stop_tags = _on_stage_tag(text)

            yield {
                "folder_id":  folder_id,
                "file_id":    file_id,
                "path":       str(txt_file),
                "text":       text,
                "clean_text": "".join(chars),
                "chars":      chars,
                "stop_tags":  stop_tags,
            }


def _on_stage_tag(text: str):
    """
    Gán nhãn on-stage cho chuỗi văn bản,
    chỉ quan tâm 7 dấu câu trong STOP_PUNCS.

    Logic giống double_tag trong load.py (phần stop_tags):
        - Gặp dấu câu on-stage → gán cho ký tự liền trước (nếu có).
        - Các ký tự bình thường     → nhãn "O".
        - Các dấu câu khác (ngoặc kép, ngoặc nhọn…) → bỏ qua.

    Returns:
        chars      : list[str]  – danh sách ký tự (không bao gồm dấu câu)
        stop_tags  : list[str]  – nhãn tương ứng, mỗi phần tử là "O" hoặc dấu câu
    """
    chars = []
    stop_tags = []

    for ch in text:
        if ch in STOP_PUNCS:
            # Gán dấu câu cho ký tự liền trước
            if stop_tags:
                # Chỉ gán nếu chưa có dấu câu (giữ dấu đầu tiên)
                if stop_tags[-1] == "O":
                    stop_tags[-1] = ch
        elif ch.strip():
            # Ký tự thường (bỏ whitespace)
            chars.append(ch)
            stop_tags.append("O")

    return chars, stop_tags


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PyTorch Dataset (lazy loading)
# ═══════════════════════════════════════════════════════════════════════════════

class CorpusTxtDataset(Dataset):
    """
    Custom PyTorch Dataset đọc dữ liệu lazy từ corpus_txt/.
    Chỉ scan đường dẫn file khi khởi tạo, đọc nội dung khi __getitem__.
    """

    def __init__(self, root_dir: str = "corpus_txt", encoding: str = "utf-8"):
        super().__init__()
        self.encoding = encoding
        self.file_list = []  # list of (folder_id, file_id, path)

        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

        for folder in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not folder.is_dir():
                continue
            folder_id = folder.name
            for txt_file in sorted(folder.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
                self.file_list.append((folder_id, txt_file.stem, txt_file))

        print(f"[CorpusTxtDataset] Tìm thấy {len(self.file_list):,} file.")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict:
        folder_id, file_id, path = self.file_list[idx]
        text = path.read_text(encoding=self.encoding, errors="replace")

        chars, stop_tags = _on_stage_tag(text)

        return {
            "folder_id":  folder_id,
            "file_id":    file_id,
            "path":       str(path),
            "text":       text,
            "clean_text": "".join(chars),
            "chars":      chars,
            "stop_tags":  stop_tags,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Demo / Test nhanh
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "corpus_txt"

    print("=" * 60)
    print("  Demo: Generator")
    print("=" * 60)
    count = 0
    for sample in corpus_generator(root):
        print(f"  📁 folder={sample['folder_id']}  📄 file={sample['file_id']}")
        print(f"     chars[:20]   = {sample['chars'][:20]}")
        print(f"     stop_tags[:20] = {sample['stop_tags'][:20]}")
        print()
        count += 1
        if count >= 3:  # Chỉ in 3 file đầu để demo
            break

    print(f"  ... (tổng cộng sẽ duyệt toàn bộ nếu không break)\n")

    print("=" * 60)
    print("  Demo: PyTorch Dataset")
    print("=" * 60)
    ds = CorpusTxtDataset(root)
    print(f"  Dataset size = {len(ds):,}")
    if len(ds) > 0:
        s = ds[0]
        print(f"  Sample[0]: folder={s['folder_id']}, file={s['file_id']}")
        print(f"     len(chars)={len(s['chars'])}, len(stop_tags)={len(s['stop_tags'])}")
        print(f"     chars[:30]     = {s['chars'][:30]}")
        print(f"     stop_tags[:30] = {s['stop_tags'][:30]}")
