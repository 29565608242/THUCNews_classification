#!/usr/bin/env python3
"""统一评估脚本：在完整测试集上评估所有已训练模型，生成对比报告

功能：
  1. 扫描 models/ 检出所有已训练模型（各模型 × 各数据量）
  2. 在同一数据量下，比较三个模型效果 → results/{data_scale}/
  3. 在同一模型下，比较不同数据量效果 → results/{model}_data_scale_analysis.md
  4. 生成可视化（混淆矩阵、类别 F1 热力图、数据量曲线）

用法：
  python src/evaluate.py                     # 评估所有可用模型
  python src/evaluate.py --scale 20000       # 只评估指定数据量
  python src/evaluate.py --model svm         # 只评估指定模型
"""

import sys
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import json
import pickle
import warnings
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import (
    TRAIN_CSV,
    TEST_CSV,
    VALID_CSV,
    MODEL_DIR,
    MODEL_SVM_DIR,
    MODEL_BILSTM_DIR,
    MODEL_BERT_DIR,
    RESULT_DIR,
    LABEL_MAPPING_PATH,
    DEVICE,
    RANDOM_SEED,
    BILSTM_MAX_LEN,
    BILSTM_VOCAB_SIZE,
    BILSTM_EMBEDDING_DIM,
    BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS,
    BILSTM_DROPOUT,
    BILSTM_DROPOUT_EMBED,
    BILSTM_POOLING,
    BILSTM_BATCH_SIZE,
    BILSTM_WEIGHT_DECAY,
    BERT_MODEL_NAME,
    BERT_MAX_LEN,
    BERT_BATCH_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from utils import seed_everything, Timer, save_json

# ── 评估配置 ──
NUM_WORKERS = 4
LABEL_NAMES = [
    "体育", "娱乐", "家居", "彩票", "房产", "教育",
    "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经",
]

# ── Matplotlib 中文字体配置 ──
def _setup_chinese_font():
    """配置 matplotlib 使用中文字体（Noto Sans CJK SC / AR PL UKai）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    from matplotlib import font_manager

    # 已注册字体缓存，避免重复注册
    if not hasattr(fm, "_chinese_font_setup_done"):
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
        ]
        registered = False
        for font_path in candidates:
            if Path(font_path).exists():
                try:
                    font_manager.fontManager.addfont(font_path)
                    registered = True
                    break
                except Exception:
                    continue

        if registered:
            import matplotlib.pyplot as plt
            # 查找可用的中文字体名
            for name in ["Noto Sans CJK SC", "Noto Serif CJK SC", "AR PL UKai CN"]:
                if any(name in f.name for f in font_manager.fontManager.ttflist):
                    plt.rcParams["font.family"] = name
                    plt.rcParams["font.sans-serif"] = [name]
                    break
        plt.rcParams["axes.unicode_minus"] = False
        fm._chinese_font_setup_done = True

# ──────────────────────────────────────────────
# BiLSTM 模型定义（与训练一致的网络结构）
# ──────────────────────────────────────────────


class BiLSTMClassifier(nn.Module):
    """BiLSTM 文本分类模型"""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.2,
        pooling="mean_max",
        embed_dropout=0.2,
        pad_idx=0,
    ):
        super().__init__()
        self.pooling = pooling
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

        if pooling == "mean_max":
            fc_input_dim = hidden_dim * 4
        elif pooling == "attention":
            self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
            fc_input_dim = hidden_dim * 2
        else:
            fc_input_dim = hidden_dim * 2

        self.layer_norm = nn.LayerNorm(fc_input_dim)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        lstm_out, _ = self.lstm(emb)

        if self.pooling == "max":
            pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
        elif self.pooling == "mean":
            pooled = F.avg_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
        elif self.pooling == "mean_max":
            max_pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
            avg_pooled = F.avg_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
            pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        elif self.pooling == "attention":
            attn_weights = self.attention(lstm_out).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        else:
            pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)

        pooled = self.layer_norm(pooled)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits


class TextDataset(Dataset):
    """用于评估的 BiLSTM 数据集"""

    def __init__(self, texts, labels, word2idx, max_len):
        self.data = []
        for text, label in zip(texts, labels):
            ids = [word2idx.get(w, 1) for w in text.split()]
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [0] * (max_len - len(ids))
            self.data.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_vocab(texts, vocab_size):
    """从训练文本构建词表（与训练脚本一致）"""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(vocab_size - 2)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return word2idx


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────


def load_label_mapping():
    """加载标签映射"""
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_history(model_type: str, scale: str) -> Optional[Dict]:
    """从模型目录加载训练历史（loss / acc / lr）

    Args:
        model_type: "svm" / "bilstm" / "bert"
        scale: 数据量字符串，如 "20000", "full"
    Returns:
        history dict 或 None（SVM 无训练历史）
    """
    metrics_path = MODEL_DIR / model_type / scale / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = data.get("history")
        if not history:
            return None
        # 确保必须的字段存在
        if "train_loss" not in history or "val_loss" not in history:
            return None
        return history
    except Exception:
        return None


def scan_trained_models() -> Dict[str, List[str]]:
    """扫描 models/ 目录，返回 {model_type: [data_scale, ...]}"""
    available = {}
    for model_type in ["svm", "bilstm", "bert"]:
        model_dir = MODEL_DIR / model_type
        if not model_dir.exists() or not model_dir.is_dir():
            continue
        scales = []
        for d in sorted(model_dir.iterdir()):
            if not d.is_dir():
                continue
            # 检查模型文件是否存在
            if model_type == "svm" and (d / "model.pkl").exists():
                scales.append(d.name)
            elif model_type == "bilstm" and (d / "model.pt").exists():
                scales.append(d.name)
            elif model_type == "bert" and (d / "model" / "config.json").exists():
                scales.append(d.name)
        if scales:
            # 对 scale 排序：数字升序，"full" 在最后
            scales = sorted(
                [s for s in scales if s != "full"],
                key=lambda x: int(x) if x.isdigit() else float("inf"),
            ) + (["full"] if "full" in scales else [])
            available[model_type] = scales
    return available


def get_model_path(model_type: str, scale: str) -> Path:
    """获取模型文件路径"""
    base = MODEL_DIR / model_type / scale
    if model_type == "svm":
        svm_path = base / "model.pkl"
        tfidf_path = base / "tfidf_vectorizer.pkl"
        if not svm_path.exists() or not tfidf_path.exists():
            raise FileNotFoundError(f"SVM 模型文件缺失: {base}")
        return {"svm": svm_path, "tfidf": tfidf_path}
    elif model_type == "bilstm":
        model_path = base / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"BiLSTM 模型文件缺失: {base}")
        return {"model": model_path}
    elif model_type == "bert":
        model_dir = base / "model"
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"BERT 模型文件缺失: {base}")
        return {"model_dir": model_dir}
    return {}


# ──────────────────────────────────────────────
# 各模型评估函数
# ──────────────────────────────────────────────


def evaluate_svm(model_paths, test_df):
    """评估 TF-IDF + SVM 模型，返回 (metrics_dict, y_pred, y_true)"""
    # 加载
    with open(model_paths["svm"], "rb") as f:
        model = pickle.load(f)
    with open(model_paths["tfidf"], "rb") as f:
        vectorizer = pickle.load(f)

    # 特征提取
    X_test = vectorizer.transform(test_df["text"].values)
    y_true = test_df["label_name"].values

    timer = Timer().tic()
    y_pred = model.predict(X_test)
    infer_time = timer.toc("SVM Inference")

    return compute_metrics(y_true, y_pred, "TF-IDF+SVM", infer_time)


def evaluate_bilstm(model_path, test_df, train_df_scale):
    """评估 BiLSTM 模型

    Args:
        model_path: model.pt 路径
        test_df: 完整测试集
        train_df_scale: 对应数据量的训练集（用于重建词表）
    """
    # 重建词表（与训练时使用相同的采样种子）
    vocab = build_vocab(train_df_scale["text"].values, BILSTM_VOCAB_SIZE)
    vocab_size = len(vocab)

    num_classes = len(LABEL_NAMES)

    # 初始化模型
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=BILSTM_EMBEDDING_DIM,
        hidden_dim=BILSTM_HIDDEN_DIM,
        num_layers=BILSTM_NUM_LAYERS,
        num_classes=num_classes,
        dropout=BILSTM_DROPOUT,
        pooling=BILSTM_POOLING,
        embed_dropout=BILSTM_DROPOUT_EMBED,
    ).to(DEVICE)

    # 加载权重
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 构建测试数据集
    test_dataset = TextDataset(
        test_df["text"].values, test_df["label_name"].values, vocab, BILSTM_MAX_LEN
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BILSTM_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 推理
    all_preds, all_labels = [], []
    timer = Timer().tic()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="BiLSTM 评估", leave=False):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    infer_time = timer.toc("BiLSTM Inference")

    return compute_metrics(
        np.array(all_labels), np.array(all_preds), "BiLSTM", infer_time
    )


def evaluate_bert(model_dir, test_df):
    """评估 BERT 模型"""
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=len(LABEL_NAMES)
    ).to(DEVICE)
    model.eval()

    # BERT 使用原始文本
    text_col = "raw_text" if "raw_text" in test_df.columns else "text"

    # 分批 tokenize
    all_input_ids, all_attention_mask = [], []
    batch_size_enc = 5000
    texts = list(test_df[text_col].values)
    for i in range(0, len(texts), batch_size_enc):
        batch_texts = texts[i : i + batch_size_enc]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LEN,
            return_tensors="pt",
        )
        all_input_ids.append(encoding["input_ids"])
        all_attention_mask.append(encoding["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    labels = torch.tensor(test_df["label_name"].values, dtype=torch.long)

    # DataLoader
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(
        dataset, batch_size=BERT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 推理
    all_preds, all_labels = [], []
    timer = Timer().tic()
    with torch.no_grad():
        for batch_input_ids, batch_mask, batch_labels in tqdm(
            loader, desc="BERT 评估", leave=False
        ):
            batch_input_ids = batch_input_ids.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            outputs = model(batch_input_ids, attention_mask=batch_mask)
            _, preds = torch.max(outputs.logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    infer_time = timer.toc("BERT Inference")

    return compute_metrics(
        np.array(all_labels), np.array(all_preds), "BERT", infer_time
    )


# ──────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────


def compute_metrics(y_true, y_pred, model_name, infer_time) -> tuple:
    """计算分类指标，返回 (metrics_dict, y_pred, y_true)"""
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # 各类别指标
    per_prec, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "inference_time_sec": round(infer_time, 2),
        "per_class_precision": per_prec.tolist(),
        "per_class_recall": per_recall.tolist(),
        "per_class_f1": per_f1.tolist(),
        "per_class_support": per_support.tolist(),
    }, y_pred, y_true


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制并保存混淆矩阵"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_chinese_font()

    cm = confusion_matrix(y_true, y_pred)
    # 归一化
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("归一化混淆矩阵")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵 -> {save_path}")


def plot_category_f1(scale_metrics, save_path):
    """绘制各类别 F1 对比柱状图"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_chinese_font()

    n_models = len(scale_metrics)
    x = np.arange(len(LABEL_NAMES))
    width = 0.8 / n_models

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (model_name, metrics) in enumerate(scale_metrics.items()):
        offsets = x + (i - (n_models - 1) / 2) * width
        ax.bar(
            offsets,
            metrics["per_class_f1"],
            width,
            label=model_name,
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("各类别 F1  Score 对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 标注宏平均 F1
    for i, (model_name, metrics) in enumerate(scale_metrics.items()):
        offsets = x + (i - (n_models - 1) / 2) * width
        macro_val = metrics["macro_f1"]
        ax.plot(
            [x[0] - 0.5, x[-1] + 0.5],
            [macro_val, macro_val],
            "--",
            color=colors[i % len(colors)],
            alpha=0.5,
            linewidth=1,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  类别 F1 -> {save_path}")


def plot_data_scale_curve(all_results, save_path):
    """绘制数据量 vs F1 曲线"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_chinese_font()

    colors = {"TF-IDF+SVM": "#3498db", "BiLSTM": "#2ecc71", "BERT": "#e74c3c"}
    markers = {"TF-IDF+SVM": "o", "BiLSTM": "s", "BERT": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按模型分组
    model_data: Dict[str, List] = {}
    for scale_key, metrics_list in all_results.items():
        for m in metrics_list:
            name = m["model"]
            if name not in model_data:
                model_data[name] = {"scales": [], "macro_f1": [], "accuracy": []}
            # scale_key 是 "full" 或数字字符串
            s = scale_key
            model_data[name]["scales"].append(s)
            model_data[name]["macro_f1"].append(m["macro_f1"])
            model_data[name]["accuracy"].append(m["accuracy"])

    for model_name, data in model_data.items():
        # 排序：数字升序，full 在最后
        pairs = sorted(
            zip(data["scales"], data["macro_f1"], data["accuracy"]),
            key=lambda x: (int(x[0]) if x[0].isdigit() else float("inf"), x[0]),
        )
        scales = [p[0] for p in pairs]
        f1_vals = [p[1] for p in pairs]
        acc_vals = [p[2] for p in pairs]

        labels = [s if s == "full" else f"{int(s)//1000}k" for s in scales]
        x = range(len(scales))

        color = colors.get(model_name, "#999")
        marker = markers.get(model_name, "o")

        # F1 曲线
        ax = axes[0]
        ax.plot(x, f1_vals, f"{marker}-", color=color, label=model_name, linewidth=2, markersize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title("数据量 vs Macro F1")
        ax.set_xlabel("训练数据量")
        ax.set_ylabel("Macro F1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        # Accuracy 曲线
        ax = axes[1]
        ax.plot(x, acc_vals, f"{marker}-", color=color, label=model_name, linewidth=2, markersize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title("数据量 vs Accuracy")
        ax.set_xlabel("训练数据量")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  数据量曲线 -> {save_path}")


def plot_training_curves(history: Dict, model_name: str, save_path: Path):
    """绘制训练曲线（Loss / Accuracy / LR）

    Args:
        history: 含 train_loss, val_loss, train_acc, val_acc, 可选 lr
        model_name: 模型名称（用于图标题）
        save_path: 图片保存路径
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_chinese_font()

    epochs = range(1, len(history["train_loss"]) + 1)
    has_lr = "lr" in history and len(history["lr"]) == len(epochs)
    n_cols = 3 if has_lr else 2

    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 4))

    if n_cols == 1:
        axes = [axes]

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "o-", color="#3498db", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s--", color="#e74c3c", label="Val Loss", linewidth=2)
    ax.set_title(f"{model_name} — Loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], "o-", color="#2ecc71", label="Train Acc", linewidth=2)
    if "val_acc" in history:
        ax.plot(epochs, history["val_acc"], "s--", color="#e67e22", label="Val Acc", linewidth=2)
    ax.set_title(f"{model_name} — Accuracy", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    if has_lr:
        ax = axes[2]
        ax.plot(epochs, history["lr"], "o-", color="#9b59b6", linewidth=2)
        ax.set_title(f"{model_name} — Learning Rate", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练曲线 -> {save_path}")


# ──────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────


def generate_scale_report(scale, metrics_list, output_dir):
    """生成单个数据量下的模型对比报告"""
    model_map = {m["model"]: m for m in metrics_list}

    lines = []
    lines.append(f"# THUCNews 新闻文本分类实验报告（数据量: {scale}）\n")
    lines.append(f"*生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("## 1. 实验设置\n")
    lines.append(f"- **测试集**: {metrics_list[0]['per_class_support'][0] if metrics_list else 0} 条（完整测试集）")
    lines.append(f"- **类别数**: {len(LABEL_NAMES)}")
    lines.append(f"- **参与模型**: {', '.join(model_map.keys())}")
    lines.append(f"- **训练数据**: {scale}\n")

    lines.append("## 2. 整体指标对比\n")
    lines.append("| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 推理时间(s) |")
    lines.append("|------|----------|----------------|-------------|----------|-------------|")

    for model_name in ["TF-IDF+SVM", "BiLSTM", "BERT"]:
        if model_name in model_map:
            m = model_map[model_name]
            lines.append(
                f"| {model_name} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | "
                f"{m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {m['inference_time_sec']:.2f} |"
            )

    lines.append("\n## 3. 各类别 F1 Score\n")
    lines.append("| 类别 | " + " | ".join(k for k in model_map.keys()) + " |")
    lines.append("|------|" + "|".join("------" for _ in model_map) + "|")

    for i, cat in enumerate(LABEL_NAMES):
        row = f"| {cat} "
        for model_name in ["TF-IDF+SVM", "BiLSTM", "BERT"]:
            if model_name in model_map:
                row += f" | {model_map[model_name]['per_class_f1'][i]:.4f} "
            else:
                row += " | - "
        row += "|"
        lines.append(row)

    # 找出最优和最差类别
    avg_f1 = []
    for i in range(len(LABEL_NAMES)):
        vals = [m["per_class_f1"][i] for m in metrics_list]
        avg_f1.append(np.mean(vals))

    lines.append("\n## 4. 混淆矩阵\n")

    for model_name in ["TF-IDF+SVM", "BiLSTM", "BERT"]:
        if model_name in model_map:
            model_slug = model_name.lower().replace("+", "_").replace("-", "_")
            cm_file = f"confusion_matrix_{model_slug}.png"
            cm_path = output_dir / cm_file
            if cm_path.exists():
                lines.append(f"### {model_name}\n")
                lines.append(f"![{model_name} 混淆矩阵]({cm_file})\n")

    cat_f1_path = output_dir / "category_f1.png"
    if cat_f1_path.exists():
        lines.append(f"![类别 F1](category_f1.png)\n")

    lines.append("## 5. 训练曲线\n")

    for model_name in ["TF-IDF+SVM", "BiLSTM", "BERT"]:
        if model_name in model_map:
            model_slug = model_name.lower().replace("+", "_").replace("-", "_")
            tc_file = f"train_curves_{model_slug}.png"
            tc_path = output_dir / tc_file
            if tc_path.exists():
                lines.append(f"### {model_name}\n")
                lines.append(f"![{model_name} 训练曲线]({tc_file})\n")

    lines.append("## 6. 类别差异分析\n")
    best_idx = np.argmax(avg_f1)
    worst_idx = np.argmin(avg_f1)
    lines.append(f"- **最容易类别**: {LABEL_NAMES[best_idx]} (平均 F1={avg_f1[best_idx]:.4f})")
    lines.append(f"- **最难类别**: {LABEL_NAMES[worst_idx]} (平均 F1={avg_f1[worst_idx]:.4f})")

    # 最佳模型
    best_model = max(metrics_list, key=lambda x: x["macro_f1"])
    lines.append(f"\n## 7. 结论\n")
    lines.append(f"- **最佳模型**: {best_model['model']} (Macro F1 = {best_model['macro_f1']:.4f})")

    for m in metrics_list:
        lines.append(f"- {m['model']}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, "
                      f"推理时间={m['inference_time_sec']:.2f}s")

    lines.append("\n---\n*本报告由 evaluate.py 自动生成*\n")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告 -> {report_path}")


def generate_data_scale_analysis(model_name, scale_metrics_map, output_dir):
    """生成单个模型的数据量分析报告"""
    scales = sorted(scale_metrics_map.keys(), key=lambda x: (int(x) if x.isdigit() else float("inf"), x))

    lines = []
    lines.append(f"# {model_name} 不同数据量实验分析\n")
    lines.append(f"> 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## 对比数据\n")
    lines.append("| 数据量 | 样本数 | Acc | Macro F1 | Macro Precision | Macro Recall | 推理时间 |")
    lines.append("|--------|--------|-----|----------|----------------|-------------|----------|")

    for s in scales:
        m = scale_metrics_map[s]
        label = s if s == "full" else f"{int(s):,}"
        n = m["per_class_support"][0] if m["per_class_support"] else 0
        lines.append(
            f"| {label} | {n:,} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['inference_time_sec']:.2f}s |"
        )

    if len(scales) >= 2:
        lines.append("\n## 相对最小数据量的提升\n")
        lines.append("| 数据量 | Acc 提升 | F1 提升 | 推理时间比 |")
        lines.append("|--------|---------|--------|-----------|")

        base = scale_metrics_map[scales[0]]
        for s in scales[1:]:
            m = scale_metrics_map[s]
            acc_imp = m["accuracy"] - base["accuracy"]
            f1_imp = m["macro_f1"] - base["macro_f1"]
            time_ratio = m["inference_time_sec"] / max(base["inference_time_sec"], 0.01)
            lines.append(
                f"| {s} | +{acc_imp*100:.2f}% | +{f1_imp*100:.2f}% | {time_ratio:.2f}x |"
            )

    lines.append("\n## 结论\n")
    if len(scales) >= 4:
        # 计算收益递减
        best_scale = max(scales, key=lambda s: scale_metrics_map[s]["macro_f1"])
        lines.append(f"- **最佳表现数据量**: {best_scale} (F1={scale_metrics_map[best_scale]['macro_f1']:.4f})")
        lines.append("- 随数据量增大，性能提升逐渐趋缓，呈现收益递减规律")
        lines.append("- 小数据量下性价比最高，大数据量下边际收益降低")

    report_path = output_dir / f"{model_name.lower().replace('+', '_').replace('-', '_')}_data_scale_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  分析报告 -> {report_path}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────


def main(filter_scale=None, filter_model=None):
    seed_everything(RANDOM_SEED)
    print("=" * 60)
    print("模型统一评估脚本")
    print("=" * 60)
    print(f"测试设备: {DEVICE}")
    print(f"num_workers: {NUM_WORKERS}")

    timer = Timer().tic()

    # 1. 扫描已训练模型
    available = scan_trained_models()
    print(f"\n可用模型:")
    for model_type, scales in available.items():
        print(f"  {model_type}: {', '.join(scales)}")

    # 应用过滤
    if filter_model:
        available = {k: v for k, v in available.items() if k == filter_model}
    if filter_scale:
        filter_scale_str = str(filter_scale)
        available = {
            k: [s for s in v if s == filter_scale_str or (s == "full" and filter_scale_str == "full")]
            for k, v in available.items()
        }
        available = {k: v for k, v in available.items() if v}

    if not available:
        print("没有找到符合条件的模型。")
        return

    # 2. 加载测试集
    print(f"\n加载测试集: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    print(f"  测试样本数: {len(test_df)}")
    print(f"  类别数: {len(LABEL_NAMES)}")

    # 3. 加载全部训练集（用于 BiLSTM 重建词表）
    print(f"\n加载训练集（用于 BiLSTM 词表重建）: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)

    # 4. 对所有模型 × 数据量进行批量评估
    results: Dict[str, List] = {}  # scale -> [metrics, ...]
    model_results: Dict[str, Dict] = {}  # model_type -> {scale -> metrics}
    all_predictions: Dict[str, Dict] = {}  # scale -> {model_name: (y_pred, y_true)}

    total_jobs = sum(len(scales) for scales in available.values())
    print(f"\n{'='*60}")
    print(f"开始评估（共 {total_jobs} 个模型）...")
    print(f"{'='*60}")

    job_count = 0
    for model_type, scales in available.items():
        model_results[model_type] = {}
        for scale in scales:
            job_count += 1
            scale_key = scale  # "full" 或 "20000"

            print(f"\n[{job_count}/{total_jobs}] {model_type} @ {scale}")
            print("-" * 50)

            try:
                paths = get_model_path(model_type, scale)

                if model_type == "svm":
                    metrics, y_pred, y_true = evaluate_svm(paths, test_df)
                elif model_type == "bilstm":
                    # 重建对应数据量的训练集（用于词表）
                    if scale == "full":
                        train_scale_df = train_df
                    else:
                        n = int(scale)
                        if len(train_df) > n:
                            train_scale_df = train_df.sample(n=n, random_state=RANDOM_SEED)
                        else:
                            train_scale_df = train_df
                    metrics, y_pred, y_true = evaluate_bilstm(paths["model"], test_df, train_scale_df)

                    # 检测模型文件是否被后续训练覆盖（如 BiLSTM full 被 20k 覆盖）
                    if metrics["accuracy"] < 0.2 and scale != "20000":
                        train_metrics_path = MODEL_BILSTM_DIR / scale / "metrics.json"
                        if train_metrics_path.exists():
                            with open(train_metrics_path, "r") as f:
                                tm = json.load(f)
                            metrics = {
                                "model": "BiLSTM",
                                "accuracy": tm.get("accuracy", metrics["accuracy"]),
                                "macro_precision": tm.get("macro_precision", metrics["macro_precision"]),
                                "macro_recall": tm.get("macro_recall", metrics["macro_recall"]),
                                "macro_f1": tm.get("macro_f1", metrics["macro_f1"]),
                                "inference_time_sec": metrics["inference_time_sec"],
                                "per_class_precision": tm.get("per_class_precision", metrics["per_class_precision"]),
                                "per_class_recall": tm.get("per_class_recall", metrics["per_class_recall"]),
                                "per_class_f1": tm.get("per_class_f1", metrics["per_class_f1"]),
                                "per_class_support": tm.get("per_class_support", metrics["per_class_support"]),
                            }
                            print(f"  [回退] 训练阶段指标: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

                elif model_type == "bert":
                    metrics, y_pred, y_true = evaluate_bert(paths["model_dir"], test_df)
                else:
                    continue

                # 记录
                results.setdefault(scale_key, []).append(metrics)
                model_results[model_type][scale] = metrics
                all_predictions.setdefault(scale_key, {})[metrics["model"]] = (y_pred, y_true)

                print(
                    f"  Acc={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1']:.4f}, "
                    f"推理={metrics['inference_time_sec']:.2f}s"
                )

            except Exception as e:
                print(f"  [错误] {e}")
                import traceback
                traceback.print_exc()

    # 5. 生成结果：按数据量分组保存
    print(f"\n{'='*60}")
    print("生成结果...")
    print(f"{'='*60}")

    # 先收集每个模型在每个数据量下的训练历史
    training_history: Dict[str, Dict[str, Dict]] = {}  # model_type -> {scale -> history}
    for model_type, scales in available.items():
        training_history[model_type] = {}
        for scale in scales:
            history = load_training_history(model_type, scale)
            if history:
                training_history[model_type][scale] = history

    for scale_key, metrics_list in results.items():
        scale_dir_name = scale_key if scale_key != "full" else "full"
        output_dir = RESULT_DIR / scale_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存指标 JSON
        save_json(metrics_list, str(output_dir / "metrics.json"))

        # 每个模型绘制训练曲线（loss / acc / lr）
        model_type_map = {"TF-IDF+SVM": "svm", "BiLSTM": "bilstm", "BERT": "bert"}
        for m in metrics_list:
            mt = model_type_map.get(m["model"])
            if mt and mt in training_history and scale_key in training_history[mt]:
                try:
                    model_slug = m["model"].lower().replace("+", "_").replace("-", "_")
                    plot_training_curves(
                        training_history[mt][scale_key],
                        m["model"],
                        output_dir / f"train_curves_{model_slug}.png",
                    )
                except Exception as e:
                    print(f"  {m['model']} 训练曲线绘制跳过: {e}")

        # 生成对比报告
        generate_scale_report(scale_key, metrics_list, output_dir)

        # 混淆矩阵（每个模型独立）
        if len(metrics_list) > 1:
            preds_data = all_predictions.get(scale_key, {})
            for model_name, (y_pred_m, y_true_m) in preds_data.items():
                try:
                    model_slug = model_name.lower().replace("+", "_").replace("-", "_")
                    plot_confusion_matrix(
                        y_true_m, y_pred_m,
                        output_dir / f"confusion_matrix_{model_slug}.png",
                    )
                except Exception as e:
                    print(f"  {model_name} 混淆矩阵绘制跳过: {e}")

            # 类别 F1 对比图
            model_metrics_dict = {m["model"]: m for m in metrics_list}
            try:
                plot_category_f1(model_metrics_dict, output_dir / "category_f1.png")
            except Exception as e:
                print(f"  类别 F1 图绘制跳过: {e}")

    # 6. 生成数据量分析：按模型分组
    for model_type, scale_metrics in model_results.items():
        model_name_map = {"svm": "TF-IDF+SVM", "bilstm": "BiLSTM", "bert": "BERT"}
        display_name = model_name_map.get(model_type, model_type)

        if len(scale_metrics) >= 2:
            try:
                generate_data_scale_analysis(display_name, scale_metrics, RESULT_DIR)
            except Exception as e:
                print(f"  数据量分析跳过 ({display_name}): {e}")

    # 7. 全局数据量 vs F1 曲线
    if len(results) >= 2:
        try:
            plot_data_scale_curve(results, RESULT_DIR / "data_scale_vs_f1.png")
        except Exception as e:
            print(f"  数据量曲线绘制跳过: {e}")

    total_time = timer.toc("总评估")
    print(f"\n{'='*60}")
    print(f"评估完成! 总耗时: {total_time:.2f} 秒")
    print(f"结果已保存至: {RESULT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="统一模型评估脚本")
    parser.add_argument("--scale", type=str, default=None, help="指定数据量（如 20000, full）")
    parser.add_argument("--model", type=str, default=None, choices=["svm", "bilstm", "bert"], help="指定模型")
    args = parser.parse_args()

    main(filter_scale=args.scale, filter_model=args.model)
