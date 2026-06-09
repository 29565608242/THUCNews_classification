#!/usr/bin/env python3
"""数据预处理——读取 THUCNews → 清洗 → 分词 → 去停用词 → 划分 → 保存 CSV"""

import csv
import os
import re
import glob
import random
from pathlib import Path
from typing import List, Optional, Tuple

import jieba
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_DIR, TRAIN_CSV, VALID_CSV, TEST_CSV, LABEL_MAPPING_PATH,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, MAX_SAMPLES,
    ensure_dirs,
)
from utils import seed_everything, save_json


# ── 默认停用词（常见高频无意义词） ──
DEFAULT_STOPWORDS = set(
    "的了在是我有和就这不人都一个上也很到说要去你会着没看好自己"
    "这那她它为所么还都可对能下过子时们间头用做面什出里只来进"
    "生学年中大多如想看得见两地与但而或因被等"
)


def load_stopwords(filepath: Optional[str] = None) -> set:
    """加载停用词表，若文件不存在则返回默认列表"""
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return DEFAULT_STOPWORDS


# ── 文本清洗 ──

def clean_text(text: str) -> str:
    """清洗文本：去 HTML 标签、URL、特殊符号、多余空格"""
    text = re.sub(r"<[^>]+>", "", text)                     # HTML 标签
    text = re.sub(r"http\S+|www\.\S+", "", text)             # URL
    text = re.sub(r"[^一-龥a-zA-Z0-9]+", " ", text)  # 只保留中文、英文字母、数字
    text = re.sub(r"\s+", " ", text).strip()                  # 合并空格
    return text


def tokenize_and_filter(text: str, stopwords: set) -> str:
    """jieba 分词 + 去停用词 + 去单字词"""
    words = jieba.lcut(text)
    words = [w for w in words if len(w) > 1 and w not in stopwords]
    return " ".join(words)


# ── 读取原始 THUCNews ──

def read_thucnews_raw(data_dir: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    读取 THUCNews 原始数据。
    目录结构: data_dir/<category>/<file> (每个文件一条新闻)
    """
    data_dir = Path(data_dir)
    texts, labels = [], []

    # 按类别文件夹遍历
    category_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not category_dirs:
        raise FileNotFoundError(
            f"未在 {data_dir} 中找到类别子目录。\n"
            f"THUCNews 的目录结构应为: raw_dir/<类别名>/<文件名>"
        )

    print(f"发现 {len(category_dirs)} 个类别: {[d.name for d in category_dirs]}")

    for cat_dir in category_dirs:
        category = cat_dir.name
        files = sorted(glob.glob(str(cat_dir / "*")))
        # 每个类别内随机采样至均衡
        random.shuffle(files)
        cat_limit = None if max_samples is None else max_samples // len(category_dirs)
        files = files[:cat_limit]

        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if len(text) < 10:  # 跳过过短文本
                    continue
                texts.append(text)
                labels.append(category)
            except Exception:
                continue

    df = pd.DataFrame({"text": texts, "label_name": labels})
    print(f"读取完成: {len(df)} 条新闻, {df['label_name'].nunique()} 个类别")
    return df


# ── 主流程 ──

def run():
    """完整的数据预处理流水线"""
    seed_everything(RANDOM_SEED)
    ensure_dirs()

    # 1. 读取原始数据
    print("=" * 50)
    print("步骤 1/6: 读取原始 THUCNews 数据...")
    df = read_thucnews_raw(RAW_DATA_DIR, max_samples=MAX_SAMPLES)
    print(f"  原始样本数: {len(df)}")

    # 2. 清洗
    print("\n步骤 2/6: 清洗文本...")
    df["text_clean"] = df["text"].apply(clean_text)
    # 去掉清洗后为空的文本
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    print(f"  清洗后样本数: {len(df)}")

    # 3. 保存清洗后的原始文本（BERT 使用原始文本，不传分词结果）
    df["raw_text"] = df["text_clean"]

    # 4. 分词 + 去停用词（用于 TF-IDF + SVM 和 BiLSTM）
    print("\n步骤 4/6: jieba 分词 + 去停用词...")
    stopwords = load_stopwords()
    df["text"] = df["text_clean"].apply(lambda x: tokenize_and_filter(x, stopwords))
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"  分词后样本数: {len(df)}")

    # 5. 标签编码
    print("\n步骤 5/6: 标签编码...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label_name"])
    label_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    save_json(label_mapping, str(LABEL_MAPPING_PATH))
    print(f"  类别映射: {label_mapping}")

    # 6. 划分 train/valid/test (8:1:1)
    print("\n步骤 6/6: 划分数据集 (8:1:1)...")

    # 先分出 test (10%)
    train_val, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=df["label"],
    )
    # 从剩余 90% 中分出 valid (10%/90% = 11.11%)
    val_ratio_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, valid_df = train_test_split(
        train_val, test_size=val_ratio_adj, random_state=RANDOM_SEED, stratify=train_val["label"],
    )

    # 恢复 label_name
    label_inverse = {v: k for k, v in label_mapping.items()}
    for split_df in [train_df, valid_df, test_df]:
        split_df["label_name"] = split_df["label"].map(label_inverse)

    # 保存
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    valid_df.to_csv(VALID_CSV, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"数据预处理完成!")
    print(f"  训练集: {len(train_df)} 条 -> {TRAIN_CSV}")
    print(f"  验证集: {len(valid_df)} 条 -> {VALID_CSV}")
    print(f"  测试集: {len(test_df)} 条 -> {TEST_CSV}")
    print(f"  类别数: {len(label_mapping)}")
    print(f"  标签映射 -> {LABEL_MAPPING_PATH}")

    # 打印类别分布
    print(f"\n类别分布（训练集）:")
    dist = train_df["label_name"].value_counts()
    for cat, cnt in dist.items():
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    run()
