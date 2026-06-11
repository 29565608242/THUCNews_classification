#!/usr/bin/env python3
"""BERT 模型——微调训练与评估"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 使用 HuggingFace 国内镜像（环境无法直接访问 huggingface.co）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import (
    TRAIN_CSV, VALID_CSV, TEST_CSV, BERT_MODEL_PATH, BERT_METRICS_PATH,
    BERT_LOSS_CURVE,
    BERT_MODEL_NAME, BERT_MAX_LEN, BERT_BATCH_SIZE,
    BERT_EPOCHS, BERT_LR, RANDOM_SEED, EARLY_STOP_PATIENCE,
    DEVICE, ensure_dirs,
)
from utils import seed_everything, Timer, save_json


# ── 数据集 ──

class BertDataset(Dataset):
    """BERT 数据集——预分词，避免每轮重复调用 tokenizer"""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def tokenize_dataset(texts, labels, tokenizer, max_len, batch_size=5000):
    """分批 tokenize 整个数据集，避免一次加载全部文本到内存"""
    print(f"  预分词 {len(texts)} 条文本（分批 {batch_size}）...")
    all_input_ids, all_attention_mask = [], []
    labels = list(labels)
    texts = list(texts)

    for i in tqdm(range(0, len(texts), batch_size), desc="预分词", unit="批"):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        all_input_ids.append(encoding["input_ids"])
        all_attention_mask.append(encoding["attention_mask"])

    return BertDataset(
        torch.cat(all_input_ids, dim=0),
        torch.cat(all_attention_mask, dim=0),
        torch.tensor(labels, dtype=torch.long),
    )


# ── 训练 ──

def train_epoch(model, loader, optimizer, scheduler, criterion, scaler=None):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="训练", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        _, preds = torch.max(logits, 1)
        total_correct += (preds == labels).sum().item()
        total += input_ids.size(0)

    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * input_ids.size(0)
            _, preds = torch.max(logits, 1)
            total_correct += (preds == labels).sum().item()
            total += input_ids.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, total_correct / total, np.array(all_preds), np.array(all_labels)


def train(data_scale=None):
    """微调 BERT 模型"""
    ensure_dirs()
    seed_everything(RANDOM_SEED)

    print("=" * 50)
    print("BERT 微调训练")
    print("=" * 50)
    print(f"模型: {BERT_MODEL_NAME}")
    print(f"设备: {DEVICE}")

    # 1. 加载数据
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if data_scale and len(train_df) > data_scale:
        train_df = train_df.sample(n=data_scale, random_state=RANDOM_SEED)
        print(f"采样训练数据: {data_scale} 条")

    num_classes = max(train_df["label_name"].max(), valid_df["label_name"].max(), test_df["label_name"].max()) + 1
    print(f"类别数: {num_classes}")
    print(f"训练集: {len(train_df)}, 验证集: {len(valid_df)}, 测试集: {len(test_df)}")

    # 2. Tokenizer
    print(f"\n加载 tokenizer: {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 预分词（全部数据 tokenize 一次，后续不再走 tokenizer）
    batch_size = BERT_BATCH_SIZE
    # 显存不足时自动降级
    if torch.cuda.is_available():
        gb_free = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gb_free < 10:
            batch_size = 8
            print(f"  显存较小 ({gb_free:.1f}GB)，batch_size 降至 {batch_size}")

    max_len = BERT_MAX_LEN

    # BERT 使用原始文本（raw_text），不分词；回退到 text 列作兼容
    text_col = "raw_text" if "raw_text" in train_df.columns else "text"
    if text_col == "raw_text":
        print("使用原始文本（raw_text）输入 BERT tokenizer")
    else:
        print("未找到 raw_text 列，回退到 text（分词后文本）")

    train_dataset = tokenize_dataset(train_df[text_col], train_df["label_name"].values, tokenizer, max_len)
    valid_dataset = tokenize_dataset(valid_df[text_col], valid_df["label_name"].values, tokenizer, max_len)
    test_dataset = tokenize_dataset(test_df[text_col], test_df["label_name"].values, tokenizer, max_len)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)

    # 4. 计算类别权重（解决不平衡）
    print("\n计算类别权重...")
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_df["label_name"].values),
        y=train_df["label_name"].values,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    for i, w in enumerate(class_weights):
        print(f"  类别 {i}: weight={w:.4f}")

    # 5. 加载预训练模型
    print(f"\n加载预训练模型: {BERT_MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_classes,
    ).to(DEVICE)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=BERT_LR)
    total_steps = len(train_loader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # 7. 训练循环（使用混合精度加速）
    print(f"\n开始训练 ({BERT_EPOCHS} epochs)...")
    timer = Timer().tic()
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    if scaler:
        print("使用混合精度训练 (fp16)")

    # data_scale 时保存到独立子目录，避免覆盖全量模型
    if data_scale:
        scale_dir = BERT_MODEL_PATH.parent / str(data_scale)
        os.makedirs(scale_dir, exist_ok=True)
        bert_save_path = str(scale_dir / "model")
    else:
        bert_save_path = str(BERT_MODEL_PATH)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, BERT_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{BERT_EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler=scaler)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(bert_save_path)
            tokenizer.save_pretrained(bert_save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    train_time = timer.toc("BERT 训练")

    # 8. 评估
    print(f"\n加载最佳模型 (epoch {best_epoch})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_save_path, num_labels=num_classes
    ).to(DEVICE)

    print("测试集评估...")
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"\n{'='*50}")
    print(f"BERT 测试集指标:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro Precision:{prec:.4f}")
    print(f"  Macro Recall:   {recall:.4f}")
    print(f"  Macro F1:       {f1:.4f}")
    print(f"  训练时间:        {train_time:.2f} 秒")
    print(f"  模型 -> {bert_save_path}")

    # 类别级指标
    class_prec, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    if data_scale:
        scale_dir = BERT_METRICS_PATH.parent / str(data_scale)
        os.makedirs(scale_dir, exist_ok=True)
        metrics_path = str(scale_dir / "metrics.json")
    else:
        metrics_path = str(BERT_METRICS_PATH)
    metrics = {
        "model": "BERT",
        "data_scale": data_scale if data_scale else "full",
        "num_classes": num_classes,
        "num_train_samples": len(train_df),
        "num_val_samples": len(valid_df),
        "num_test_samples": len(test_df),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "train_time_sec": round(train_time, 2),
        "test_loss": round(test_loss, 4),
        "best_epoch": best_epoch,
        "history": history,
        "hyperparams": {
            "model_name": BERT_MODEL_NAME,
            "max_len": max_len,
            "batch_size": batch_size,
            "epochs": BERT_EPOCHS,
            "lr": BERT_LR,
        },
        "per_class_precision": class_prec.tolist(),
        "per_class_recall": class_recall.tolist(),
        "per_class_f1": class_f1.tolist(),
        "per_class_support": class_support.tolist(),
    }

    save_json(metrics, metrics_path)
    print(f"  指标 -> {metrics_path}")

    # 绘制训练曲线
    if not data_scale:
        from utils import plot_training_history
        plot_training_history(history, str(BERT_LOSS_CURVE), "BERT")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_scale", type=int, default=None,
                        help="训练数据量（条数），不传则使用全量")
    args = parser.parse_args()
    train(data_scale=args.data_scale)
