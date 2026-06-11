#!/usr/bin/env python3
"""从 data/raw/ 重建训练集——包含数据清洗、规范化、去重、分词、去停用词

流程:
  1. 扫描 data/raw/*category*/ 原始文本
  2. 数据质量过滤：空白、超短、低质量（纯标点/数字）文本
  3. 文本规范化：全角→半角、HTML/URL 去除、重复字符压缩
  4. 文本去重：去除完全重复样本（防止训练/测试数据泄露）
  5. 分层抽样 8:1:1 分割 train/val/test
  6. jieba 分词 + 去停用词 → text 列（供 BiLSTM / SVM 使用）
  7. 保留原始原文 → raw_text 列（供 BERT 使用）
  8. 保存 CSV + label_mapping.json
"""

import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1（剩余）

# ── 数据质量过滤阈值 ──
MIN_TEXT_LENGTH = 30            # 最短文本长度（字符数）
MIN_CHINESE_RATIO = 0.3         # 最低中文字符占比（低于此值视为低质量）
MAX_TEXT_LENGTH = 50000         # 最长文本长度（字符数），超出截断

# 标签映射（与原始一致）
LABEL_MAP = {
    "体育": 0, "娱乐": 1, "家居": 2, "彩票": 3,
    "房产": 4, "教育": 5, "时尚": 6, "时政": 7,
    "星座": 8, "游戏": 9, "社会": 10, "科技": 11,
    "股票": 12, "财经": 13,
}
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

# ── 常用中文停用词表（内置，无需外部文件）──
_BUILTIN_STOPWORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你
会 着 没有 看 好 自己 这 他 她 它 们 那 什么 怎么 这个 那个 因为
所以 但是 可以 还 而 又 或者 如果 虽然 然后 这样 那样 之 与 及
被 把 对 从 为 以 于 比 向 让 给 用 做 将 能 已经 正在 这些 那些
每 各 某 哪 几 多少 如何 为何 才 刚 已经 曾经 正在 将 会 能 可以
应该 必须 一定 可能 也许 大概 非常 很 太 极 最 更 比较 相当 稍微
更加 越发 尤其 甚至 几乎 大约 左右 上下 多少 有点 有些 稍微 差点
不过 只是 但是 可是 然而 却 则 反而 固然 虽然 尽管 即使 哪怕 就算
如果 假如 倘若 要是 只要 除非 因为 由于 所以 因此 因而 于是
不仅 不但 而且 并且 况且 何况 乃至 甚至 以及 连同 或者 还是 或是
要么 与其 宁可 宁愿 与其 就是 就是说 也就是说 意思是 的话 来说
来讲 而言 来说 看来 想来 总之 总而言之 综上 综上所述 也就是说
例如 比如 比方 如同 犹如 仿佛 类似 诸如 好比 譬如 像 似的 一样
一般 同样 另外 此外 同时 还有 再者 更何况 不单 不只 不光 不论
不管 无论 任凭 凡是 所有 一切 任何 每个 各个 别的 其他 另外
一样 这么 那么 怎么 怎样 怎么样 为什么 怎么办 如何 如此 这样
那样 这些 那些 这里 那里 这边 那边 这个 那个 本 该 此 某
各 每 哪 什么 怎么 怎样 怎么样 为什么 哪些 哪些 几 多少 多久 多
大 小 高 低 长 短 宽 窄 厚 薄 深 浅 好 坏 美 丑 新 旧 真 假
正 副 主 次 单 双 左 右 前 后 上 下 内 外 中 间 旁 东 西 南 北
啊 阿 哎 哎呀 哎哟 唉 哦 噢 嗯 唔 哈 哈哈 呵 呵呵 哼 呸 咳
喂 嗨 哟 哦 咦 耶 哇 呀 啦 嘛 吗 吧 呢 么 的 了 过 着 地 得
啊 呀 哇 哪 啦 嘛 吗 吧 呢 么 哈 嘿 哟 哦 嗯 唔
""".strip().split())

# 中文字符正则（用于计算中文占比）
_RE_CHINESE = re.compile(r"[一-鿿]")

# ── 文本规范化正则 ──
_RE_HTML = re.compile(r"<[^>]+>", re.IGNORECASE)                    # HTML 标签
_RE_URL = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s,，。；；！？、\"'】】）\)]*",
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_REPEAT_CHAR = re.compile(r"(.)\1{3,}")                           # 连续重复 ≥4 次
_RE_REPEAT_PUNCT = re.compile(r"([，。！？；：、~～…—\-\.\!\?])\1{2,}") # 标点重复 ≥3 次
_RE_WHITESPACE = re.compile(r"\s+")


def _load_stopwords() -> set:
    """加载停用词表：优先从配置文件指定的路径加载，否则使用内置列表"""
    # 如果 config 中有指定 stopwords 文件路径，尝试加载
    try:
        from config import STOPWORDS_FILE
        if STOPWORDS_FILE:
            fpath = Path(STOPWORDS_FILE)
            if fpath.exists():
                words = set()
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w and not w.startswith("#"):
                            words.add(w)
                print(f"  加载停用词表: {fpath} ({len(words)} 词)")
                return words
    except (ImportError, AttributeError):
        pass
    print(f"  使用内置停用词表 ({len(_BUILTIN_STOPWORDS)} 词)")
    return _BUILTIN_STOPWORDS


# ── 文本规范化（全角→半角、HTML/URL 清理、重复字符压缩）──

def _fullwidth_to_halfwidth(text: str) -> str:
    """全角字母、数字、符号 → 半角，保留全角中文和标点"""
    result = []
    for ch in text:
        code = ord(ch)
        # 全角字母数字（FF01-FF5E）→ 半角对应（21-7E）
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        # 全角空格（3000）→ 半角空格
        elif code == 0x3000:
            result.append(chr(0x0020))
        else:
            result.append(ch)
    return "".join(result)


def _compress_repeated(text: str) -> str:
    """压缩连续重复字符：中文单字重复≥4次压缩为2次，标点重复≥3次压缩为1次"""
    # 先压缩标点
    text = _RE_REPEAT_PUNCT.sub(r"\1", text)
    # 再压缩文字（保留 2 次表示叠词/强调，如"哈哈哈"→保留"哈哈哈"不过3次以上才触发）
    text = _RE_REPEAT_CHAR.sub(r"\1\1", text)
    return text


def normalize_text(text: str, do_html: bool = True, do_url: bool = True,
                    do_fullwidth: bool = True, do_repeat: bool = True) -> str:
    """文本规范化流水线

    Args:
        text: 原始文本
        do_html: 是否移除 HTML 标签
        do_url: 是否移除 URL 和邮箱
        do_fullwidth: 是否将全角字母数字转半角
        do_repeat: 是否压缩连续重复字符
    Returns:
        规范化后的文本
    """
    if not text:
        return text

    # 1. Unicode NFKC 归一化
    text = unicodedata.normalize("NFKC", text)

    # 2. HTML 标签去除
    if do_html:
        text = _RE_HTML.sub("", text)

    # 3. URL 和邮箱去除（替换为空格，避免拼词）
    if do_url:
        text = _RE_URL.sub(" ", text)
        text = _RE_EMAIL.sub(" ", text)

    # 4. 全角→半角
    if do_fullwidth:
        text = _fullwidth_to_halfwidth(text)

    # 5. 压缩连续重复字符
    if do_repeat:
        text = _compress_repeated(text)

    # 6. 合并多余空白（包括步骤 3 引入的空格）
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


# ── 文本去重 ──

def dedup_records(records, by_field: int = 0, verbose: bool = True):
    """基于指定字段（默认 0 = text）去除完全重复的记录

    Args:
        records: (text, label_name, label_id) 列表
        by_field: 去重依据的字段索引（0=text, 1=label_name, 2=label_id）
        verbose: 是否打印去重统计
    Returns:
        去重后的记录列表
    """
    n_before = len(records)
    seen = set()
    deduped = []
    for r in records:
        key = r[by_field]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    n_after = len(deduped)
    if verbose:
        removed = n_before - n_after
        pct = removed / n_before * 100 if n_before else 0
        print(f"  去重: {removed} 条重复 ({pct:.2f}%) | 保留 {n_after}/{n_before}")
    return deduped


def is_valid_text(text: str) -> bool:
    """判断文本是否通过质量过滤"""
    # 1. 空白检查
    if not text or not text.strip():
        return False

    # 2. 最短长度检查
    if len(text) < MIN_TEXT_LENGTH:
        return False

    # 3. 最长长度检查（避免异常大文件）
    if len(text) > MAX_TEXT_LENGTH:
        return False

    # 4. 中文字符占比检查（过滤纯标点/数字/英文的噪音数据）
    chinese_chars = _RE_CHINESE.findall(text)
    chinese_ratio = len(chinese_chars) / len(text)
    if chinese_ratio < MIN_CHINESE_RATIO:
        return False

    return True


def scan_raw_data():
    """扫描 data/raw/ 下所有文件，经质量过滤后返回 (text, label_name, label_id) 列表"""
    total, filtered = 0, 0
    records = []
    for cat_name in sorted(os.listdir(RAW_DIR)):
        cat_dir = RAW_DIR / cat_name
        if not cat_dir.is_dir() or cat_name not in LABEL_MAP:
            continue
        label_id = LABEL_MAP[cat_name]
        files = sorted(os.listdir(cat_dir))
        for fname in files:
            fpath = cat_dir / fname
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace").strip()
                total += 1
                if not is_valid_text(text):
                    filtered += 1
                    continue
                records.append((text, cat_name, label_id))
            except Exception as e:
                print(f"  跳过 {fpath}: {e}")
                filtered += 1
    print(f"  共 {total} 个文件，过滤掉 {filtered} 个劣质样本，保留 {len(records)} 条")
    return records


def split_data(records):
    """按比例分层分割为 train/val/test"""
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    print(f"\n分割: 训练={len(train)}, 验证={len(val)}, 测试={len(test)}")
    return train, val, test


def build_df(records, stopwords: set):
    """从记录列表构建 DataFrame，对文本进行 jieba 分词 + 去停用词"""
    texts, labels, label_names = zip(*records)
    print(f"\njieba 分词 + 去停用词 ({len(texts)} 条)...")

    seg_texts = []
    for i, t in enumerate(texts):
        if (i + 1) % 100000 == 0:
            print(f"  进度: {i+1}/{len(texts)}")

        # jieba 分词
        words = jieba.lcut(t, cut_all=False)

        # 去停用词 + 去单字词（除了有意义的单字如"是""的"等已被停用词表覆盖）
        filtered_words = [w for w in words if w not in stopwords and len(w.strip()) > 0]

        # 如果过滤后为空，保留至少一个词以避免空行
        if not filtered_words:
            filtered_words = words[:1]  # 保留第一个词作为占位

        seg_texts.append(" ".join(filtered_words))

    df = pd.DataFrame({
        "raw_text": texts,               # 原始完整原文（BERT 使用）
        "text": seg_texts,               # jieba 分词 + 去停用词后（BiLSTM / SVM 使用）
        "label_name": label_names,
        "label": labels,
    })
    return df


def print_stats(df, name, stopwords: set):
    """打印数据集统计"""
    print(f"\n{'='*50}")
    print(f"{name} 统计 ({len(df)} 条)")
    print(f"{'='*50}")

    # 停用词统计（训练集）
    if name == "训练集":
        all_words = []
        for t in df["text"]:
            all_words.extend(t.split())
        total_tokens = len(all_words)
        stopword_hits = sum(1 for w in all_words if w in stopwords)
        print(f"  分词后总 token 数: {total_tokens}")
        if stopword_hits:
            print(f"  已去除停用词: {stopword_hits} 次 ({stopword_hits/total_tokens*100:.1f}%)")
        print(f"  token 中位数: {pd.Series([len(t.split()) for t in df['text']]).median():.0f}")

    print(f"  原始原文 (BERT 使用):")
    raw_lens = df["raw_text"].str.len()
    print(f"    字符中位数: {raw_lens.median():.0f}, 均值: {raw_lens.mean():.0f}")
    print(f"    P99: {raw_lens.quantile(0.99):.0f}, 最大: {raw_lens.max()}")
    print(f"  jieba 分词后 (BiLSTM / SVM 使用):")
    tok_lens = df["text"].str.split().str.len()
    print(f"    token 中位数: {tok_lens.median():.0f}, 均值: {tok_lens.mean():.0f}")
    print(f"    P99: {tok_lens.quantile(0.99):.0f}, 最大: {tok_lens.max()}")

    print(f"\n  类别分布:")
    dist = df["label_name"].value_counts()
    for cat, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"    {cat}: {cnt:>6} ({pct:.1f}%)")


def run():
    print("=" * 50)
    print("从 data/raw/ 重建训练集（含数据清洗）")
    print("=" * 50)

    # 0. 加载停用词表
    print("\n加载停用词表...")
    stopwords = _load_stopwords()

    # 1. 扫描原始数据 + 质量过滤
    print("\n扫描原始数据（含质量过滤）...")
    print(f"  过滤规则: 最少字符={MIN_TEXT_LENGTH}, 最低中文占比={MIN_CHINESE_RATIO}")
    records = scan_raw_data()

    # 打印各类别数量
    cat_counts = Counter(r[1] for r in records)
    for cat in sorted(cat_counts):
        print(f"    {cat}: {cat_counts[cat]}")

    # 1.5 文本规范化（全角→半角、HTML/URL 去除、重复字符压缩）
    print("\n文本规范化...")
    # 尝试从 config.py 读取规范化开关；如果 import 失败则使用默认值 (True)
    _norm_html = True
    _norm_url = True
    _norm_fullwidth = True
    _norm_repeat = True
    _dedup_enabled = True
    try:
        from config import (
            NORMALIZE_HTML, NORMALIZE_URL, NORMALIZE_FULLWIDTH,
            NORMALIZE_REPEAT, DEDUP_ENABLED,
        )
        _norm_html = NORMALIZE_HTML
        _norm_url = NORMALIZE_URL
        _norm_fullwidth = NORMALIZE_FULLWIDTH
        _norm_repeat = NORMALIZE_REPEAT
        _dedup_enabled = DEDUP_ENABLED
    except (ImportError, AttributeError):
        pass
    n_raw = len(records)
    raw_texts_before = [r[0] for r in records]  # 保存原始文本用于统计
    for i, (text, cat_name, label_id) in enumerate(records):
        normalized = normalize_text(
            text, do_html=_norm_html, do_url=_norm_url,
            do_fullwidth=_norm_fullwidth, do_repeat=_norm_repeat,
        )
        records[i] = (normalized, cat_name, label_id)
    n_changed = sum(1 for i in range(n_raw) if records[i][0] != raw_texts_before[i])
    print(f"  已处理 {n_raw} 条，其中 {n_changed} 条发生变化 ({n_changed/n_raw*100:.1f}%)")

    # 1.6 文本去重（基于文本内容，去除训练/测试集中的重复）
    if _dedup_enabled:
        print("\n文本去重...")
        records = dedup_records(records, by_field=0)
    else:
        print("\n文本去重: 已跳过 (DEDUP_ENABLED=False)")

    # 2. 分割
    print("\n分割数据集...")
    train_records, val_records, test_records = split_data(records)

    # 3. 构建 DataFrame（分词 + 去停用词）
    train_df = build_df(train_records, stopwords)
    val_df = build_df(val_records, stopwords)
    test_df = build_df(test_records, stopwords)

    # 4. 统计信息
    print_stats(train_df, "训练集", stopwords)
    print_stats(val_df, "验证集", stopwords)
    print_stats(test_df, "测试集", stopwords)

    # 5. 保存 CSV
    print(f"\n保存 CSV...")
    out_train = OUT_DIR / "train.csv"
    out_val = OUT_DIR / "valid.csv"
    out_test = OUT_DIR / "test.csv"

    train_df.to_csv(out_train, index=False, encoding="utf-8")
    val_df.to_csv(out_val, index=False, encoding="utf-8")
    test_df.to_csv(out_test, index=False, encoding="utf-8")
    print(f"  {out_train} ({len(train_df)} 条)")
    print(f"  {out_val} ({len(val_df)} 条)")
    print(f"  {out_test} ({len(test_df)} 条)")

    # 6. 保存 label_mapping.json
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = MODEL_DIR / "label_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)
    print(f"  标签映射 -> {mapping_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run()
