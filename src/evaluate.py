#!/usr/bin/env python3
"""综合评估入口——加载模型、计算指标、生成可视化与实验报告

用法:
    python src/evaluate.py                              # 全量评估三个模型
    python src/evaluate.py --data_scale 20000           # 评估指定数据量
    python src/evaluate.py --models svm,bilstm          # 只评估指定模型
    python src/evaluate.py --data-scale-only            # 仅分析数据量实验
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
)

from config import (
    TRAIN_CSV, TEST_CSV,
    SVM_MODEL_PATH, BILSTM_MODEL_PATH, BERT_MODEL_PATH,
    SVM_METRICS_PATH, BILSTM_METRICS_PATH, BERT_METRICS_PATH,
    RESULT_DIR,
    CONFUSION_MATRIX_PNG, LOSS_CURVE_PNG, CATEGORY_F1_PNG,
    METRICS_JSON, REPORT_MD,
    RANDOM_SEED, ensure_dirs,
)
from utils import seed_everything, save_json, Timer

from eval_models import load_label_mapping, predict_with_model
from eval_visualize import (
    plot_confusion_matrix, plot_loss_curve, plot_category_f1,
    run_data_scale_analysis,
)
from eval_report import generate_report


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════

def run(models=None, data_scale_only=False, data_scale=None):
    """运行完整评估

    Args:
        models: 模型名称列表，如 ["svm", "bilstm"]，None 表示全部
        data_scale_only: 仅分析数据量实验，跳过完整评估
        data_scale: 指定数据量（条数），评估该数据量下的所有模型
    """
    ensure_dirs()
    seed_everything(RANDOM_SEED)

    # ── 路径解析（支持 data_scale 子目录）──
    if data_scale:
        scale_str = str(data_scale)
        _result_dir = RESULT_DIR / scale_str
        os.makedirs(_result_dir, exist_ok=True)
        _metrics_json = _result_dir / "metrics.json"
        _confusion_matrix_png = _result_dir / "confusion_matrix.png"
        _loss_curve_png = _result_dir / "loss_curve.png"
        _category_f1_png = _result_dir / "category_f1.png"
        _report_md = _result_dir / "report.md"
    else:
        _result_dir = RESULT_DIR
        _metrics_json = METRICS_JSON
        _confusion_matrix_png = CONFUSION_MATRIX_PNG
        _loss_curve_png = LOSS_CURVE_PNG
        _category_f1_png = CATEGORY_F1_PNG
        _report_md = REPORT_MD

    print("=" * 50)
    if data_scale:
        print(f"评估: data_scale={data_scale}")
    else:
        print("综合评估与可视化")
    print("=" * 50)

    # 加载标签
    class_names, id_to_name, name_to_id = load_label_mapping()
    num_classes = len(class_names)
    test_df = pd.read_csv(TEST_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    X_test = test_df["text"].values
    y_test = test_df["label_name"].values

    print(f"测试集: {len(test_df)} 条, {num_classes} 个类别")

    # 各模型预测
    all_model_keys = ["svm", "bilstm", "bert"]
    display_names = {"svm": "TF-IDF+SVM", "bilstm": "BiLSTM", "bert": "BERT"}

    # 如果指定了 models，只评估选中的模型
    model_names = all_model_keys if models is None else [m for m in all_model_keys if m in models]
    if not model_names:
        print(f"  [错误] 无效的模型名称: {models}，可选: {all_model_keys}")
        return

    if data_scale_only:
        print("  仅分析数据量实验，跳过完整评估")
        run_data_scale_analysis(models_filter=[display_names.get(m, m) for m in model_names])
        return

    y_preds = {}
    all_metrics = []
    history_data = {}
    train_times = {}

    for model_key in model_names:
        name = display_names[model_key]
        print(f"\n{'='*40}")
        print(f"评估模型: {name}")
        print(f"{'='*40}")

        # SVM/BiLSTM 用分词后 text，BERT 用原始 raw_text（与训练一致）
        if model_key == "bert":
            model_texts = test_df.get("raw_text", test_df["text"]).values
        else:
            model_texts = test_df["text"].values

        # 检查模型文件是否存在
        _model_paths = {
            "svm": (SVM_MODEL_PATH.parent / str(data_scale) / "model.pkl") if data_scale else SVM_MODEL_PATH,
            "bilstm": (BILSTM_MODEL_PATH.parent / str(data_scale) / "model.pt") if data_scale else BILSTM_MODEL_PATH,
            "bert": (BERT_MODEL_PATH.parent / str(data_scale) / "model") if data_scale else BERT_MODEL_PATH,
        }
        mp = _model_paths[model_key]
        if not mp.exists():
            print(f"  [跳过] 模型文件不存在: {mp}")
            continue

        timer = Timer().tic()
        y_pred = predict_with_model(model_key, model_texts.tolist(), data_scale=data_scale)
        infer_time = timer.toc(f"{name} 推理")

        # 统一预测标签为整数
        if y_pred.dtype.kind in ('U', 'S', 'O'):
            y_pred = np.array([name_to_id[p] for p in y_pred])

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
            "data_scale": data_scale if data_scale else "full",
            "num_train_samples": data_scale if data_scale else len(train_df),
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

        # 尝试加载训练指标
        metrics_files = {
            "svm": (SVM_METRICS_PATH.parent / str(data_scale) / "metrics.json") if data_scale else SVM_METRICS_PATH,
            "bilstm": (BILSTM_METRICS_PATH.parent / str(data_scale) / "metrics.json") if data_scale else BILSTM_METRICS_PATH,
            "bert": (BERT_METRICS_PATH.parent / str(data_scale) / "metrics.json") if data_scale else BERT_METRICS_PATH,
        }
        json_path = metrics_files[model_key]
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
    save_json(all_metrics, str(_metrics_json))
    print(f"\n指标汇总 -> {_metrics_json}")

    # 可视化
    print(f"\n生成可视化图表...")

    if y_preds:
        plot_confusion_matrix(y_preds, class_names, str(_confusion_matrix_png))

    if history_data:
        plot_loss_curve(history_data, str(_loss_curve_png))

    if all_metrics:
        plot_category_f1({m["model"]: m for m in all_metrics}, class_names, str(_category_f1_png))

    # 数据量分析（仅全量评估时执行）
    if not data_scale:
        run_data_scale_analysis(full_metrics=all_metrics,
                                models_filter=[display_names.get(m, m) for m in model_names])

    # 生成报告
    print(f"\n生成实验报告...")
    generate_report(
        all_metrics=all_metrics,
        class_names=class_names,
        num_train=data_scale if data_scale else len(train_df),
        num_test=len(test_df),
        num_classes=num_classes,
        train_times=train_times,
        save_path=str(_report_md),
    )

    print(f"\n{'='*50}")
    report_dir = _result_dir if data_scale else RESULT_DIR
    print(f"评估完成! 所有结果已保存到 {report_dir} 目录")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="综合评估与数据量实验分析")
    parser.add_argument("--models", type=str, default=None,
                        help="要评估的模型，逗号分隔，如: svm,bilstm (默认全部)")
    parser.add_argument("--data-scale-only", action="store_true",
                        help="仅分析数据量实验，跳过完整评估")
    parser.add_argument("--data_scale", type=int, default=None,
                        help="评估指定数据量下的模型（如 20000），结果保存到 results/<N>/")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else None
    run(models=models, data_scale_only=args.data_scale_only, data_scale=args.data_scale)
