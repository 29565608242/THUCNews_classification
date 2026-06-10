#!/usr/bin/env python3
"""将 data/raw_clean/ 的数据转换成训练脚本需要的 CSV 格式

raw_clean 数据包含:
  - train/dev/test.jsonl —— 原始文本 (raw text) + 中文标签名
  - train/dev/test.txt   —— jieba 分词后文本 (segmented text) + 标签 ID
  - label_map.txt        —— 标签名 -> ID 映射

输出: data/train.csv, data/valid.csv, data/test.csv（与 data.py 输出格式一致）
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CLEAN_DIR = ROOT / "data" / "raw_clean"
OUT_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def load_label_map(path: Path) -> dict:
    """加载 label_map.txt -> {label_name: label_id}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, lid = line.split("\t")
            mapping[name] = int(lid)
    return mapping


def prepare_split(jsonl_path: Path, txt_path: Path, label_map: dict) -> pd.DataFrame:
    """读取一个分片的 jsonl 和 txt，合并为 DataFrame"""
    # 1. 读取 jsonl（原始文本 + 标签名）
    raw_texts, label_names = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_texts.append(obj["text"])
            label_names.append(obj["label"])

    # 2. 读取 txt（分词后文本 + 标签名）
    seg_texts, txt_label_names = [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: label_name\tsegmented_text
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            lbl, text = parts
            txt_label_names.append(lbl)
            seg_texts.append(text)

    # 验证长度一致
    assert len(raw_texts) == len(seg_texts) == len(label_names) == len(txt_label_names), \
        f"行数不匹配: jsonl={len(raw_texts)}, txt={len(seg_texts)}"

    # 验证标签名一致
    for i in range(len(label_names)):
        assert label_names[i] == txt_label_names[i], \
            f"第 {i} 行标签不匹配: jsonl={label_names[i]}, txt={txt_label_names[i]}"

    # 3. 获取 label ID
    label_ids = [label_map[name] for name in label_names]

    # 4. 构建 DataFrame
    df = pd.DataFrame({
        "text": seg_texts,
        "raw_text": raw_texts,
        "label_name": label_names,
        "label": label_ids,
    })
    return df


def run():
    print("=" * 50)
    print("使用 raw_clean 数据生成训练 CSV")
    print("=" * 50)

    # 1. 加载标签映射
    label_map_fpath = RAW_CLEAN_DIR / "label_map.txt"
    label_map = load_label_map(label_map_fpath)
    print(f"标签映射 ({len(label_map)} 类): {label_map}")

    # 2. 处理各分片
    splits = {
        "train.csv": ("train.jsonl", "train.txt"),
        "valid.csv": ("dev.jsonl", "dev.txt"),
        "test.csv": ("test.jsonl", "test.txt"),
    }

    for out_name, (jsonl_name, txt_name) in splits.items():
        jsonl_path = RAW_CLEAN_DIR / jsonl_name
        txt_path = RAW_CLEAN_DIR / txt_name

        print(f"\n处理 {jsonl_name} + {txt_name} ...")
        df = prepare_split(jsonl_path, txt_path, label_map)

        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  -> {out_path} ({len(df)} 条)")

    # 3. 保存 label_mapping.json（供 evaluate.py 使用）
    label_mapping_json = {name: lid for name, lid in label_map.items()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mapping_out = MODEL_DIR / "label_mapping.json"
    with open(mapping_out, "w", encoding="utf-8") as f:
        json.dump(label_mapping_json, f, ensure_ascii=False, indent=2)
    print(f"\n标签映射 -> {mapping_out}")

    # 4. 打印样本分布
    print(f"\n数据集概览:")
    for out_name in splits:
        df = pd.read_csv(OUT_DIR / out_name)
        print(f"  {out_name}: {len(df)} 条")
    print(f"  类别数: {len(label_map)}")

    train_df = pd.read_csv(OUT_DIR / "train.csv")
    print(f"\n训练集类别分布:")
    dist = train_df["label_name"].value_counts()
    for cat, cnt in dist.items():
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    run()
