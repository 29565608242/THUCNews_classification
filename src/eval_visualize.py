"""可视化图表——混淆矩阵、损失曲线、类别 F1 热力图、数据量影响折线图"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import MODEL_DIR, DATA_SCALE_PNG, METRICS_JSON
from utils import save_json

# ── 全局字体 ──
plt.rcParams["font.sans-serif"] = ["AR PL UKai CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(y_true: Dict[str, np.ndarray],
                          class_names: List[str], save_path: str):
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


# ══════════════════════════════════════════════════════
# 数据量影响分析
# ══════════════════════════════════════════════════════

def run_data_scale_analysis(full_metrics=None, models_filter=None):
    """加载并绘制数据量实验对比图

    Args:
        full_metrics: 已有的全量指标列表（用于合并保存）
        models_filter: 模型名称过滤列表，如 ["TF-IDF+SVM", "BiLSTM"]，None 表示全部
    """
    data_scale_files = sorted(MODEL_DIR.glob("*/[0-9]*/metrics.json"))
    if not data_scale_files:
        print("  [跳过] 无数据量实验数据")
        return

    ds_results = []
    for f in data_scale_files:
        try:
            data = json.load(open(f, "r"))
            if models_filter and data.get("model") not in models_filter:
                continue
            ds_results.append(data)
        except Exception:
            pass

    if not ds_results:
        print("  [跳过] 无匹配的数据量实验数据")
        return

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"数据量实验汇总 ({len(ds_results)} 组)")
    print(f"{'='*60}")
    summary = {}
    for r in ds_results:
        key = f"{r['model']}@{r.get('data_scale','full')}"
        summary[key] = r.get("macro_f1", "N/A")
    for k, v in sorted(summary.items()):
        print(f"  {k:<30s}  Macro F1={v}")

    plot_data_scale_effect(ds_results, str(DATA_SCALE_PNG))

    # 合并保存
    if full_metrics:
        merged = full_metrics + ds_results
        save_json(merged, str(METRICS_JSON))
        print(f"  指标（含数据量实验） -> {METRICS_JSON}")

    print(f"  数据量影响图 -> {DATA_SCALE_PNG}")


def plot_data_scale_effect(data_scale_results, save_path):
    """绘制数据量 vs Macro-F1 折线图"""
    if not data_scale_results:
        print("  [跳过] 无数据量实验数据")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    records = defaultdict(list)
    for r in data_scale_results:
        scale = r.get("data_scale", "full")
        records[r["model"]].append((scale, r["macro_f1"]))

    markers = {"TF-IDF+SVM": "o", "BiLSTM": "s", "BERT": "^"}
    colors = {"TF-IDF+SVM": "#3498db", "BiLSTM": "#2ecc71", "BERT": "#e74c3c"}

    for model_name in ["TF-IDF+SVM", "BiLSTM", "BERT"]:
        if model_name not in records:
            continue
        pts = sorted(records[model_name], key=lambda x: (
            0 if x[0] == "full" else (int(x[0]) if x[0] != "full" else float("inf"))
        ))
        x_labels = [str(p[0]) if p[0] != "full" else "full" for p in pts]
        x_vals = []
        for p in pts:
            if p[0] == "full":
                if x_vals:
                    x_vals.append(x_vals[-1] * 2)
                else:
                    x_vals.append(100000)
            else:
                x_vals.append(int(p[0]))

        y_vals = [p[1] for p in pts]
        ax.plot(x_vals, y_vals, f"-{markers.get(model_name, 'o')}",
                color=colors.get(model_name, "gray"), label=model_name,
                linewidth=2, markersize=8)

        for x, y in zip(x_vals, y_vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("训练数据量（条）", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("数据量对分类性能的影响", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

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
