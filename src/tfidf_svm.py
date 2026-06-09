#!/usr/bin/env python3
"""TF-IDF + SVM 模型——训练与评估"""

import argparse
import pickle
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import (
    TRAIN_CSV, TEST_CSV, SVM_MODEL_PATH, TFIDF_PATH,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RANDOM_SEED,
    ensure_dirs,
)
from utils import seed_everything, Timer, save_json

warnings.filterwarnings("ignore")


def train(data_scale=None):
    """训练 TF-IDF + SVM 模型"""
    ensure_dirs()
    seed_everything(RANDOM_SEED)

    # 1. 加载数据
    print("=" * 50)
    print("TF-IDF + SVM 训练")
    print("=" * 50)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if data_scale and len(train_df) > data_scale:
        train_df = train_df.sample(n=data_scale, random_state=RANDOM_SEED)
        print(f"采样训练数据: {data_scale} 条")
    print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")

    X_train = train_df["text"].values
    y_train = train_df["label"].values
    X_test = test_df["text"].values
    y_test = test_df["label"].values

    # 2. 特征提取
    print(f"\n特征提取: TF-IDF (max_features={TFIDF_MAX_FEATURES}, ngram={TFIDF_NGRAM_RANGE})...")
    timer = Timer().tic()
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    timer.toc("TF-IDF 特征提取")
    print(f"  特征维度: {X_train_tfidf.shape[1]}")

    # 3. 训练
    print("\n训练 LinearSVC...")
    timer.tic()
    model = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED)
    model.fit(X_train_tfidf, y_train)
    train_time = timer.toc("SVM 训练")

    # 4. 评估
    print("\n评估模型...")
    timer.tic()
    y_pred = model.predict(X_test_tfidf)
    infer_time = timer.toc("SVM 推理")

    acc = accuracy_score(y_test, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n{'='*50}")
    print(f"TF-IDF + SVM 测试集指标:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro Precision:{prec:.4f}")
    print(f"  Macro Recall:   {recall:.4f}")
    print(f"  Macro F1:       {f1:.4f}")
    print(f"  训练时间:        {train_time:.2f} 秒")
    print(f"  推理时间:        {infer_time:.2f} 秒")

    # 类别级指标
    class_prec, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )
    class_metrics = {
        "per_class_precision": class_prec.tolist(),
        "per_class_recall": class_recall.tolist(),
        "per_class_f1": class_f1.tolist(),
        "per_class_support": class_support.tolist(),
    }

    # 5. 保存
    print(f"\n保存模型...")
    scale_suffix = f"_{data_scale}" if data_scale else ""
    svm_path = str(SVM_MODEL_PATH).replace(".pkl", f"{scale_suffix}.pkl")
    tfidf_path = str(TFIDF_PATH).replace(".pkl", f"{scale_suffix}.pkl")

    metrics = {
        "model": "TF-IDF+SVM",
        "data_scale": data_scale if data_scale else "full",
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(infer_time, 2),
        **class_metrics,
    }

    with open(svm_path, "wb") as f:
        pickle.dump(model, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  SVM 模型     -> {svm_path}")
    print(f"  TF-IDF 向量器 -> {tfidf_path}")

    save_json(metrics, tfidf_path + ".metrics.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_scale", type=int, default=None,
                        help="训练数据量（条数），不传则使用全量")
    args = parser.parse_args()
    train(data_scale=args.data_scale)
