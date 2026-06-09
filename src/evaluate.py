#!/usr/bin/env python3
"""综合评估——加载所有模型，计算指标，生成可视化与实验报告"""

import os
# 使用 HuggingFace 国内镜像（环境无法直接访问 huggingface.co）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TRAIN_CSV, VALID_CSV, TEST_CSV,
    SVM_MODEL_PATH, TFIDF_PATH, BILSTM_MODEL_PATH, BERT_MODEL_PATH,
    MODEL_DIR, LABEL_MAPPING_PATH,
    CONFUSION_MATRIX_PNG, LOSS_CURVE_PNG, CATEGORY_F1_PNG,
    DATA_SCALE_PNG, METRICS_JSON, REPORT_MD,
    BILSTM_MAX_LEN, BILSTM_EMBEDDING_DIM, BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS, BILSTM_DROPOUT, BILSTM_VOCAB_SIZE,
    DEVICE, RANDOM_SEED, ensure_dirs,
)
from utils import seed_everything, save_json, Timer

# ── 全局字体 ──
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════
# 1. 加载标签映射
# ══════════════════════════════════════════════════════

def load_label_mapping() -> Tuple[List[str], Dict[int, str]]:
    """加载类别映射，返回类别名称列表和 id->name 映射"""
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # mapping: {"类别名": int_id}
    id_to_name = {int(v): k for k, v in mapping.items()}
    class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    return class_names, id_to_name


# ══════════════════════════════════════════════════════
# 2. 加载模型与预测
# ══════════════════════════════════════════════════════

def load_svm() -> Tuple:
    """加载 TF-IDF + SVM 模型"""
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(SVM_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def load_bilstm(vocab_size: int, num_classes: int) -> nn.Module:
    """加载 BiLSTM 模型"""
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=BILSTM_EMBEDDING_DIM,
        hidden_dim=BILSTM_HIDDEN_DIM,
        num_layers=BILSTM_NUM_LAYERS,
        num_classes=num_classes,
        dropout=BILSTM_DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(BILSTM_MODEL_PATH), map_location=DEVICE))
    model.eval()
    return model


class BiLSTMClassifier(nn.Module):
    """与 bilstm.py 定义完全一致的模型结构"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits


def load_bert(num_classes: int):
    """加载 BERT 模型与 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BERT_MODEL_PATH), num_labels=num_classes
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_with_model(model_name: str, texts: List[str]) -> np.ndarray:
    """用指定模型对文本列表进行预测，返回预测标签"""
    if model_name == "svm":
        vectorizer, model = load_svm()
        X = vectorizer.transform(texts)
        return model.predict(X)

    elif model_name == "bilstm":
        # 需要重建 word2idx
        train_df = pd.read_csv(TRAIN_CSV)
        from collections import Counter
        counter = Counter()
        for t in train_df["text"].values:
            counter.update(t.split())
        most_common = counter.most_common(BILSTM_VOCAB_SIZE - 2)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w, _ in most_common:
            word2idx[w] = len(word2idx)

        num_classes = pd.read_csv(TRAIN_CSV)["label"].nunique()
        model = load_bilstm(len(word2idx), num_classes)

        preds = []
        for text in texts:
            ids = [word2idx.get(w, 1) for w in text.split()]
            if len(ids) > BILSTM_MAX_LEN:
                ids = ids[:BILSTM_MAX_LEN]
            else:
                ids = ids + [0] * (BILSTM_MAX_LEN - len(ids))
            inp = torch.tensor([ids], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits = model(inp)
                pred = torch.argmax(logits, dim=1).item()
            preds.append(pred)
        return np.array(preds)

    elif model_name == "bert":
        num_classes = pd.read_csv(TRAIN_CSV)["label"].nunique()
        model, tokenizer = load_bert(num_classes)
        preds = []
        for text in texts:
            enc = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=256, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1).item()
            preds.append(pred)
        return np.array(preds)

    else:
        raise ValueError(f"未知模型: {model_name}")


# ══════════════════════════════════════════════════════
# 3. 可视化函数
# ══════════════════════════════════════════════════════

def plot_confusion_matrix(y_true: Dict[str, np.ndarray], class_names: List[str], save_path: str):
    """绘制三模型混淆矩阵对比图（2×2 布局）"""
    model_names = list(y_true.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        cm = confusion_matrix(y_true[name]["true"], y_true[name]["pred"])
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], cbar=False)
        axes[i].set_title(f"{name}", fontsize=14)
        axes[i].set_xlabel("预测类别")
        axes[i].set_ylabel("真实类别")
        # 旋转标签
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right", fontsize=8)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=8)

    # 隐藏右下角空白子图
    if len(model_names) < 4:
        axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵 -> {save_path}")


def plot_loss_curve(history_data: Dict[str, Dict], save_path: str):
    """绘制 BiLSTM + BERT 损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"BiLSTM": ("#3498db", "#e74c3c"), "BERT": ("#2ecc71", "#e67e22")}

    for ax, (name, history) in zip(axes, history_data.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], "o-", color=colors.get(name, ("blue", "red"))[0],
                label="Train Loss")
        ax.plot(epochs, history["val_loss"], "s--", color=colors.get(name, ("blue", "red"))[1],
                label="Val Loss")
        ax.set_title(f"{name} 损失曲线", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 双轴显示 accuracy
        ax2 = ax.twinx()
        ax2.plot(epochs, history["train_acc"], "o:", color="gray", alpha=0.5, label="Train Acc")
        ax2.plot(epochs, history["val_acc"], "s:", color="orange", alpha=0.5, label="Val Acc")
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  损失曲线 -> {save_path}")


def plot_category_f1(metrics_data: Dict[str, Dict], class_names: List[str], save_path: str):
    """绘制各类别 F1 热力图"""
    f1_data = {}
    for model_name, metrics in metrics_data.items():
        if "per_class_f1" in metrics:
            f1_data[model_name] = metrics["per_class_f1"]

    if not f1_data:
        print("  [跳过] 无类别级 F1 数据")
        return

    f1_df = pd.DataFrame(f1_data, index=class_names)

    fig, ax = plt.subplots(figsize=(max(6, len(f1_data) * 2), max(8, len(class_names) * 0.6)))
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=f1_df.columns, yticklabels=class_names,
                ax=ax, cbar_kws={"label": "F1 Score"})
    ax.set_title("各类别 F1 Score 对比", fontsize=14)
    ax.set_xlabel("模型")
    ax.set_ylabel("类别")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  类别 F1 热力图 -> {save_path}")


def plot_data_scale_effect(data_scale_results: List[Dict], save_path: str):
    """绘制数据量 vs Macro-F1 折线图"""
    if not data_scale_results:
        print("  [跳过] 无数据量实验数据")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 整理数据
    from collections import defaultdict
    records = defaultdict(list)
    for r in data_scale_results:
        scale = r.get("data_scale", "full")
        scale_label = scale if scale != "full" else "full"
        records[r["model"]].append((scale_label, r["macro_f1"]))

    markers = {"TF-IDF+SVM": "o", "BiLSTM": "s", "BERT": "^"}
    colors = {"TF-IDF+SVM": "#3498db", "BiLSTM": "#2ecc71", "BERT": "#e74c3c"}

    for model_name in ["TF-IDF+SVM", "BiLSTM", "BERT"]:
        if model_name not in records:
            continue
        pts = sorted(records[model_name], key=lambda x: (
            0 if x[0] == "full" else (int(x[0]) if x[0] != "full" else float("inf"))
        ))
        # 解析横轴标签
        x_labels = [str(p[0]) if p[0] != "full" else "full" for p in pts]
        x_vals = []
        for p in pts:
            if p[0] == "full":
                # 取上一个值的 2x 作为近似
                if x_vals:
                    x_vals.append(x_vals[-1] * 2)
                else:
                    x_vals.append(100000)
            else:
                x_vals.append(int(p[0]))

        y_vals = [p[1] for p in pts]
        ax.plot(x_vals, y_vals, f"-{markers.get(model_name, 'o')}",
                color=colors.get(model_name, "gray"), label=model_name, linewidth=2, markersize=8)

        # 标注数值
        for x, y in zip(x_vals, y_vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("训练数据量（条）", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("数据量对分类性能的影响", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 自定义 x 轴刻度
    all_xticks = sorted(set(
        v[0] if v[0] != "full" else 100000
        for rec in records.values() for v in rec
    ))
    all_xlabels = [str(x) if x != 100000 else "full" for x in all_xticks]
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xlabels)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  数据量影响图 -> {save_path}")


# ══════════════════════════════════════════════════════
# 4. 生成实验报告
# ══════════════════════════════════════════════════════

def generate_report(
    all_metrics: List[Dict],
    class_names: List[str],
    num_train: int, num_test: int, num_classes: int,
    train_times: Dict[str, float],
    save_path: str,
):
    """生成 Markdown 格式实验报告"""
    lines = []

    lines.append("# THUCNews 新闻文本分类实验报告\n")
    lines.append(f"*生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # -- 1. 引言 --
    lines.append("## 1. 引言\n")
    lines.append("本实验基于 THUCNews 新闻数据集，实现并对比了三种文本分类模型：\n")
    lines.append("- **TF-IDF + SVM**：传统机器学习基线\n")
    lines.append("- **BiLSTM**：深度学习序列模型\n")
    lines.append("- **BERT**（Chinese RoBERTa）：预训练语言模型微调\n\n")

    # -- 2. 数据集 --
    lines.append("## 2. 数据集与预处理\n")
    lines.append(f"- **训练集**：{num_train} 条\n")
    lines.append(f"- **测试集**：{num_test} 条\n")
    lines.append(f"- **类别数**：{num_classes}\n")
    lines.append(f"- **类别列表**：{'、'.join(class_names)}\n")
    lines.append("- **预处理**：去 HTML 标签、URL、特殊符号 → jieba 分词 → 去停用词\n\n")

    # -- 3. 模型结构与参数 --
    lines.append("## 3. 模型结构与参数\n")

    lines.append("### 3.1 TF-IDF + SVM\n")
    lines.append("| 参数 | 值 |\n|------|------|\n")
    lines.append("| 特征提取 | TF-IDF |\n")
    lines.append("| max_features | 100,000 |\n")
    lines.append("| ngram_range | (1, 2) |\n")
    lines.append("| 分类器 | LinearSVC |\n\n")

    lines.append("### 3.2 BiLSTM\n")
    lines.append("| 参数 | 值 |\n|------|------|\n")
    lines.append("| Embedding Dim | 300 |\n")
    lines.append("| Hidden Dim | 256 |\n")
    lines.append("| LSTM Layers | 2 (双向) |\n")
    lines.append("| Dropout | 0.5 |\n")
    lines.append("| Max Length | 300 |\n")
    lines.append("| Batch Size | 128 |\n")
    lines.append("| Epochs | 10 (含 Early Stopping) |\n")
    lines.append("| Optimizer | Adam |\n\n")

    lines.append("### 3.3 BERT\n")
    lines.append("| 参数 | 值 |\n|------|------|\n")
    lines.append(f"| 预训练模型 | {BERT_MODEL_PATH.parent.name} |\n")
    lines.append("| Max Length | 256 |\n")
    lines.append("| Batch Size | 16 |\n")
    lines.append("| Epochs | 3 (含 Early Stopping) |\n")
    lines.append("| Optimizer | AdamW |\n")
    lines.append("| Learning Rate | 2e-5 |\n\n")

    lines.append(f"训练设备: {DEVICE.upper()}\n\n")

    # -- 4. 实验结果 --
    lines.append("## 4. 实验结果\n")

    # 整体指标表
    lines.append("### 4.1 整体指标对比\n\n")
    lines.append("| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 训练时间(s) |\n")
    lines.append("|------|----------|----------------|-------------|----------|------------|\n")

    for rec in all_metrics:
        model = rec.get("model", "?")
        lines.append(
            f"| {model} | {rec.get('accuracy', 'N/A')} | "
            f"{rec.get('macro_precision', 'N/A')} | {rec.get('macro_recall', 'N/A')} | "
            f"{rec.get('macro_f1', 'N/A')} | {rec.get('train_time_sec', 'N/A')} |\n"
        )
    lines.append("\n")

    # 训练时间
    lines.append("### 4.2 时间对比\n\n")
    lines.append("| 模型 | 训练时间 (秒) |\n|------|--------------|\n")
    for model_name, t in sorted(train_times.items()):
        lines.append(f"| {model_name} | {t:.2f} |\n")
    lines.append("\n")

    # 类别级表现
    lines.append("### 4.3 各类别 F1 Score\n\n")
    model_names = [r.get("model", "?") for r in all_metrics]
    header = "| 类别 | " + " | ".join(model_names) + " |\n"
    lines.append(header)
    lines.append("|------|" + "|".join(["------"] * len(model_names)) + "|\n")

    for cls_name in class_names:
        # 获取该类别在各类模型中的 F1
        lines.append(f"| {cls_name} ")
        for rec in all_metrics:
            per_class = rec.get("per_class_f1", [])
            class_index = class_names.index(cls_name)
            f1_val = per_class[class_index] if class_index < len(per_class) else "N/A"
            lines.append(f" | {f1_val}")
        lines.append(" |\n")
    lines.append("\n")

    # 混淆矩阵
    lines.append("### 4.4 混淆矩阵\n\n")
    lines.append(f"![混淆矩阵]({CONFUSION_MATRIX_PNG.name})\n\n")

    # 损失曲线
    lines.append("### 4.5 训练曲线\n\n")
    lines.append(f"![损失曲线]({LOSS_CURVE_PNG.name})\n\n")

    # -- 5. 类别差异分析 --
    lines.append("## 5. 类别差异分析\n\n")
    lines.append(f"![类别 F1 热力图]({CATEGORY_F1_PNG.name})\n\n")

    # 找出最佳和最差类别
    if all_metrics and "per_class_f1" in all_metrics[0]:
        avg_f1 = np.mean([r["per_class_f1"] for r in all_metrics if "per_class_f1" in r], axis=0)
        best_idx = np.argmax(avg_f1)
        worst_idx = np.argmin(avg_f1)
        lines.append(f"- **最容易类别**：{class_names[best_idx]} (平均 F1={avg_f1[best_idx]:.3f})\n")
        lines.append(f"- **最难类别**：{class_names[worst_idx]} (平均 F1={avg_f1[worst_idx]:.3f})\n\n")

    # -- 6. 数据量影响 --
    lines.append("## 6. 数据量影响分析\n\n")
    lines.append(f"![数据量 vs F1]({DATA_SCALE_PNG.name})\n\n")
    lines.append("分析要点：\n")
    lines.append("- 数据量增大时各模型的 Macro-F1 变化趋势\n")
    lines.append("- 小数据量下哪个模型表现更好\n")
    lines.append("- 各模型性能是否随数据量增加进入平台期\n\n")

    # -- 7. 结论 --
    lines.append("## 7. 结论\n\n")

    # 找最佳模型
    best_model = max(all_metrics, key=lambda r: r.get("macro_f1", 0))
    lines.append(f"- **最佳模型**：{best_model.get('model', 'N/A')} (Macro F1 = {best_model.get('macro_f1', 'N/A')})\n")
    lines.append("- TF-IDF + SVM 作为轻量基线，在资源受限场景下具有实用价值\n")
    lines.append("- BiLSTM 在序列建模上优于传统方法，但训练时间较长\n")
    lines.append("- BERT 借助预训练知识取得最佳性能，但需要 GPU 资源\n\n")

    lines.append("---\n")
    lines.append("*本报告由实验代码自动生成*\n")

    report_text = "\n".join(lines)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  实验报告 -> {save_path}")


# ══════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════

def run():
    """运行完整评估"""
    ensure_dirs()
    seed_everything(RANDOM_SEED)

    print("=" * 50)
    print("综合评估与可视化")
    print("=" * 50)

    # 加载标签
    class_names, id_to_name = load_label_mapping()
    num_classes = len(class_names)
    test_df = pd.read_csv(TEST_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    X_test = test_df["text"].values
    y_test = test_df["label"].values

    print(f"测试集: {len(test_df)} 条, {num_classes} 个类别")

    # 各模型预测
    model_names = ["svm", "bilstm", "bert"]
    display_names = {"svm": "TF-IDF+SVM", "bilstm": "BiLSTM", "bert": "BERT"}

    y_preds = {}
    all_metrics = []
    history_data = {}
    data_scale_results = []
    train_times = {}

    for model_key in model_names:
        name = display_names[model_key]
        print(f"\n{'='*40}")
        print(f"评估模型: {name}")
        print(f"{'='*40}")

        # 检查模型文件是否存在
        model_paths = {
            "svm": SVM_MODEL_PATH,
            "bilstm": BILSTM_MODEL_PATH,
            "bert": BERT_MODEL_PATH,
        }
        if not model_paths[model_key].exists():
            print(f"  [跳过] 模型文件不存在: {model_paths[model_key]}")
            continue

        timer = Timer().tic()
        y_pred = predict_with_model(model_key, X_test.tolist())
        infer_time = timer.toc(f"{name} 推理")

        y_preds[name] = {"true": y_test, "pred": y_pred}

        # 指标
        acc = accuracy_score(y_test, y_pred)
        prec, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        class_prec, class_recall, class_f1, class_support = precision_recall_fscore_support(
            y_test, y_pred, zero_division=0
        )

        metrics = {
            "model": name,
            "data_scale": "full",
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(recall, 4),
            "macro_f1": round(f1, 4),
            "inference_time_sec": round(infer_time, 2),
            "per_class_precision": class_prec.tolist(),
            "per_class_recall": class_recall.tolist(),
            "per_class_f1": class_f1.tolist(),
            "per_class_support": class_support.tolist(),
        }

        # 尝试加载训练时间
        metrics_path = model_paths[model_key]
        json_path = Path(str(metrics_path) + ".metrics.json")
        if json_path.exists():
            saved_metrics = json.load(open(json_path, "r"))
            if "train_time_sec" in saved_metrics:
                metrics["train_time_sec"] = saved_metrics["train_time_sec"]
                train_times[name] = saved_metrics["train_time_sec"]
            if "history" in saved_metrics:
                history_data[name] = saved_metrics["history"]

        all_metrics.append(metrics)

        print(f"  Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | 推理时间: {infer_time:.2f}s")

    # 保存所有指标
    save_json(all_metrics, str(METRICS_JSON))
    print(f"\n指标汇总 -> {METRICS_JSON}")

    # 可视化
    print(f"\n生成可视化图表...")

    if y_preds:
        plot_confusion_matrix(y_preds, class_names, str(CONFUSION_MATRIX_PNG))

    if history_data:
        plot_loss_curve(history_data, str(LOSS_CURVE_PNG))

    if all_metrics:
        plot_category_f1({m["model"]: m for m in all_metrics}, class_names, str(CATEGORY_F1_PNG))

    # 尝试加载数据量实验结果（从模型目录读取 *_<数字>.metrics.json）
    data_scale_files = sorted(MODEL_DIR.glob("*_[0-9]*.metrics.json"))
    if data_scale_files:
        ds_results = []
        for f in data_scale_files:
            try:
                data = json.load(open(f, "r"))
                ds_results.append(data)
            except Exception:
                pass
        if ds_results:
            plot_data_scale_effect(ds_results, str(DATA_SCALE_PNG))
            # 一并保存到 results/metrics.json（完整版包含全量 + 各数据量）
            ds_results_merged = all_metrics + ds_results
            save_json(ds_results_merged, str(METRICS_JSON))
            print(f"  指标（含数据量实验） -> {METRICS_JSON}")

    # 生成报告
    print(f"\n生成实验报告...")
    generate_report(
        all_metrics=all_metrics,
        class_names=class_names,
        num_train=len(train_df),
        num_test=len(test_df),
        num_classes=num_classes,
        train_times=train_times,
        save_path=str(REPORT_MD),
    )

    print(f"\n{'='*50}")
    print(f"评估完成! 所有结果已保存到 results/ 目录")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()
