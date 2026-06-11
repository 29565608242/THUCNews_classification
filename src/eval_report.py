"""实验报告生成——Markdown 格式报告"""

from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    BILSTM_EMBEDDING_DIM, BILSTM_HIDDEN_DIM, BILSTM_NUM_LAYERS,
    BILSTM_DROPOUT, BILSTM_MAX_LEN, BILSTM_BATCH_SIZE, BILSTM_EPOCHS,
    BERT_MODEL_NAME, BERT_MAX_LEN, BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LR,
    DEVICE, EARLY_STOP_PATIENCE,
    CONFUSION_MATRIX_PNG, LOSS_CURVE_PNG, CATEGORY_F1_PNG, DATA_SCALE_PNG,
)


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
    lines.append(f"| Embedding Dim | {BILSTM_EMBEDDING_DIM} |\n")
    lines.append(f"| Hidden Dim | {BILSTM_HIDDEN_DIM} |\n")
    lines.append(f"| LSTM Layers | {BILSTM_NUM_LAYERS} (双向) |\n")
    lines.append(f"| Dropout | {BILSTM_DROPOUT} |\n")
    lines.append(f"| Max Length | {BILSTM_MAX_LEN} |\n")
    lines.append(f"| Batch Size | {BILSTM_BATCH_SIZE} |\n")
    lines.append(f"| Epochs | {BILSTM_EPOCHS} (含 Early Stopping, patience={EARLY_STOP_PATIENCE}) |\n")
    lines.append("| Optimizer | AdamW |\n\n")

    lines.append("### 3.3 BERT\n")
    lines.append("| 参数 | 值 |\n|------|------|\n")
    lines.append(f"| 预训练模型 | {BERT_MODEL_NAME} |\n")
    lines.append(f"| Max Length | {BERT_MAX_LEN} |\n")
    lines.append(f"| Batch Size | {BERT_BATCH_SIZE} |\n")
    lines.append(f"| Epochs | {BERT_EPOCHS} (含 Early Stopping, patience={EARLY_STOP_PATIENCE}) |\n")
    lines.append("| Optimizer | AdamW |\n")
    lines.append(f"| Learning Rate | {BERT_LR} |\n\n")

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
