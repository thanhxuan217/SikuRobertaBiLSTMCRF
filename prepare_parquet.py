# -*- coding: utf-8 -*-
"""
Helper script to convert corpus_txt directory into a Parquet file 
and pre-extract vocabulary (chars and bigrams) for lazy loading.
"""

from pathlib import Path
import json
from collections import Counter
import pandas as pd
from tqdm import tqdm

from parsering.utils.common import punctuation, tag_before

def check_non_stop(line):
    # Dummy implementation from load.py
    return True

def build_parquet_and_vocab(corpus_dir="corpus_txt", output_parquet="dataset.parquet", output_vocab="vocab.json"):
    root = Path(corpus_dir)
    if not root.exists():
        print(f"Directory {corpus_dir} not found. Skipping dataset prep.")
        return

    print("Scanning corpus text to build Parquet file & Vocab...")
    
    data_lines = []
    chars_set = set()
    bichars_set = set()
    file_count = 0

    for folder in tqdm(sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0)):
        if not folder.is_dir():
            continue

        for txt_file in sorted(folder.glob('*.txt'), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            try:
                text = txt_file.read_text(encoding='utf-8', errors='replace')
            except Exception as e:
                continue

            file_count += 1
            raw_lines = [line.strip() for line in text.splitlines() if line.strip()]

            new_lines = []
            for line in raw_lines:
                while line and line[0] in punctuation and line[0] not in tag_before:
                    if new_lines:
                        new_lines[-1] += line[0]
                    line = line[1:]

                if not line:
                    continue

                has_punc = any(p in line for p in punctuation)
                if has_punc:
                    if check_non_stop(line):
                        line = line.replace("''", "'").replace('""', '"').replace('《《', '《').replace('》》', '》').replace('：：', '：')
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            # Extract characters and bi-characters (ignoring punctuation for vocab logic roughly)
            for line in new_lines:
                data_lines.append(line)
                
                # Rough vocab extraction (actual load.py extracts after double_tag)
                # But for standard characters it's fine
                for ch in line:
                    if ch.strip():
                        chars_set.add(ch)
                        
                for i in range(len(line) - 1):
                    bichars_set.add(line[i:i+2])

    print(f"Read {file_count:,} files, extracted {len(data_lines):,} lines.")
    
    # Save to Parquet
    df = pd.DataFrame({"text": data_lines})
    df.to_parquet(output_parquet, engine="pyarrow")
    print(f"Saved to {output_parquet}")

    # Save Vocab
    vocab = {
        "chars": list(chars_set),
        "bichars": list(bichars_set)
    }
    with open(output_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab to {output_vocab}")

if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus_txt"
    out_pq = sys.argv[2] if len(sys.argv) > 2 else "dataset.parquet"
    out_vocab = sys.argv[3] if len(sys.argv) > 3 else "vocab.json"
    build_parquet_and_vocab(corpus, out_pq, out_vocab)
