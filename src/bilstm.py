#!/usr/bin/env python3
"""BiLSTM 模型——定义、训练与评估"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import (
    TRAIN_CSV, VALID_CSV, TEST_CSV, BILSTM_MODEL_PATH, BILSTM_METRICS_PATH,
    BILSTM_LOSS_CURVE, MODEL_BILSTM_DIR,
    BILSTM_MAX_LEN, BILSTM_EMBEDDING_DIM, BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS, BILSTM_DROPOUT, BILSTM_DROPOUT_EMBED,
    BILSTM_BATCH_SIZE, BILSTM_EPOCHS, BILSTM_LR, BILSTM_LR_MIN,
    BILSTM_WEIGHT_DECAY, BILSTM_USE_SCHEDULER, BILSTM_VOCAB_SIZE,
    BILSTM_POOLING, RANDOM_SEED, EARLY_STOP_PATIENCE, DEVICE, ensure_dirs,
)
from utils import seed_everything, Timer, save_json


# ── 数据集 ──

class TextDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, texts, labels, word2idx, max_len):
        self.data = []
        for text, label in zip(texts, labels):
            ids = [word2idx.get(w, 1) for w in text.split()]  # 1 = UNK
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [0] * (max_len - len(ids))  # 0 = PAD
            self.data.append((torch.tensor(ids, dtype=torch.long),
                              torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_vocab(texts, vocab_size):
    """从训练文本构建词表"""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    # 保留最常见的词
    most_common = counter.most_common(vocab_size - 2)  # 留出 PAD(0) 和 UNK(1)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return word2idx


# ── 模型 ──

class BiLSTMClassifier(nn.Module):
    """BiLSTM 文本分类模型（支持多种 pooling 策略）"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes, dropout, pooling="mean_max",
                 embed_dropout=0.2, pad_idx=0):
        super().__init__()
        self.pooling = pooling
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

        # pooling 方式决定分类器输入维度
        if pooling == "mean_max":
            fc_input_dim = hidden_dim * 4  # 双向 * (mean + max)
        elif pooling == "attention":
            self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
            fc_input_dim = hidden_dim * 2
        else:  # max 或 mean
            fc_input_dim = hidden_dim * 2

        self.layer_norm = nn.LayerNorm(fc_input_dim)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)            # (batch, seq_len, emb_dim)
        emb = self.embed_dropout(emb)
        lstm_out, _ = self.lstm(emb)       # (batch, seq_len, hidden*2)

        if self.pooling == "max":
            pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
        elif self.pooling == "mean":
            pooled = F.avg_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
        elif self.pooling == "mean_max":
            max_pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
            avg_pooled = F.avg_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
            pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        elif self.pooling == "attention":
            # 自注意力和池化结合
            attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)
        else:
            pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)

        pooled = self.layer_norm(pooled)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits


# ── 训练 ──

def train_epoch(model, loader, optimizer, criterion, max_grad_norm=5.0, epoch=None, total_epochs=None):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    desc = f"Epoch {epoch}/{total_epochs}" if epoch else "训练"
    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion, desc="评估"):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, total_correct / total, np.array(all_preds), np.array(all_labels)


def train(data_scale=None, params_override=None):
    """训练 BiLSTM 模型

    Args:
        data_scale: 训练数据量
        params_override: 参数字典，覆盖 config 中的默认值
    """
    ensure_dirs()
    seed_everything(RANDOM_SEED)

    # 合并覆盖参数
    p = dict(
        max_len=BILSTM_MAX_LEN,
        embedding_dim=BILSTM_EMBEDDING_DIM,
        hidden_dim=BILSTM_HIDDEN_DIM,
        num_layers=BILSTM_NUM_LAYERS,
        dropout=BILSTM_DROPOUT,
        embed_dropout=BILSTM_DROPOUT_EMBED,
        batch_size=BILSTM_BATCH_SIZE,
        epochs=BILSTM_EPOCHS,
        lr=BILSTM_LR,
        weight_decay=BILSTM_WEIGHT_DECAY,
        use_scheduler=BILSTM_USE_SCHEDULER,
        vocab_size=BILSTM_VOCAB_SIZE,
        pooling=BILSTM_POOLING,
        lr_min=BILSTM_LR_MIN,
    )
    if params_override:
        p.update(params_override)

    print("=" * 50)
    print("BiLSTM 训练")
    print("=" * 50)
    print(f"设备: {DEVICE}")
    print(f"超参数: { {k: v for k, v in p.items() if k != 'use_scheduler'} }")

    # 1. 加载数据
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if data_scale and len(train_df) > data_scale:
        scale_ratio = data_scale / len(train_df)
        train_df = train_df.sample(n=data_scale, random_state=RANDOM_SEED)
        valid_df = valid_df.sample(n=max(1, int(len(valid_df) * scale_ratio)), random_state=RANDOM_SEED)
        test_df = test_df.sample(n=max(1, int(len(test_df) * scale_ratio)), random_state=RANDOM_SEED)
        print(f"按比例采样: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    num_classes = max(train_df["label_name"].max(), valid_df["label_name"].max(), test_df["label_name"].max()) + 1
    print(f"类别数: {num_classes}")
    print(f"训练集: {len(train_df)}, 验证集: {len(valid_df)}, 测试集: {len(test_df)}")

    # 2. 构建词表
    print("\n构建词表...")
    vocab = build_vocab(train_df["text"].values, p["vocab_size"])
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    # 3. 数据加载器
    batch_size = p["batch_size"]
    train_dataset = TextDataset(train_df["text"].values, train_df["label_name"].values, vocab, p["max_len"])
    valid_dataset = TextDataset(valid_df["text"].values, valid_df["label_name"].values, vocab, p["max_len"])
    test_dataset = TextDataset(test_df["text"].values, test_df["label_name"].values, vocab, p["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # 4. 计算类别权重（解决不平衡）
    print("\n计算类别权重...")
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_df["label_name"].values),
        y=train_df["label_name"].values,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    for i, w in enumerate(class_weights):
        print(f"  类别 {i}: weight={w:.4f}")

    # 6. 初始化模型
    print("\n初始化 BiLSTM 模型...")
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=p["embedding_dim"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        num_classes=num_classes,
        dropout=p["dropout"],
        pooling=p["pooling"],
        embed_dropout=p["embed_dropout"],
    ).to(DEVICE)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"],
    )
    max_grad_norm = 5.0

    # 学习率调度器
    scheduler = None
    if p["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
            min_lr=p["lr_min"],
        )

    # 7. 训练循环
    print(f"\n开始训练 ({p['epochs']} epochs)...")
    timer = Timer().tic()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, p["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, max_grad_norm, epoch=epoch, total_epochs=p["epochs"])
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion, desc="验证")

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(current_lr)

        print(f"  Epoch {epoch:2d}/{p['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        # 学习率调度
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(BILSTM_MODEL_PATH))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    train_time = timer.toc("BiLSTM 训练")

    # 8. 加载最佳模型并评估
    print(f"\n加载最佳模型 (epoch {best_epoch})...")
    model.load_state_dict(torch.load(str(BILSTM_MODEL_PATH), map_location=DEVICE))

    print("测试集评估...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, desc="测试")

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(test_labels, test_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="macro", zero_division=0
    )

    print(f"\n{'='*50}")
    print(f"BiLSTM 测试集指标:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro Precision:{prec:.4f}")
    print(f"  Macro Recall:   {recall:.4f}")
    print(f"  Macro F1:       {f1:.4f}")
    print(f"  训练时间:        {train_time:.2f} 秒")
    print(f"  模型 -> {BILSTM_MODEL_PATH}")

    # 类别级指标
    class_prec, class_recall, class_f1, class_support = precision_recall_fscore_support(
        test_labels, test_preds, zero_division=0
    )

    if data_scale:
        scale_dir = MODEL_BILSTM_DIR / str(data_scale)
        os.makedirs(scale_dir, exist_ok=True)
        model_path = str(scale_dir / "model.pt")
        metrics_path = str(scale_dir / "metrics.json")
        torch.save(model.state_dict(), model_path)
        print(f"  模型 -> {model_path}")
    else:
        model_path = str(BILSTM_MODEL_PATH)
        metrics_path = str(BILSTM_METRICS_PATH)

    metrics = {
        "model": "BiLSTM",
        "data_scale": data_scale if data_scale else "full",
        "num_classes": num_classes,
        "num_train_samples": len(train_df),
        "num_val_samples": len(valid_df),
        "num_test_samples": len(test_df),
        "vocab_size": len(vocab),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "train_time_sec": round(train_time, 2),
        "test_loss": round(test_loss, 4),
        "best_epoch": best_epoch,
        "history": history,
        "hyperparams": {k: v for k, v in p.items() if k != "use_scheduler"},
        "per_class_precision": class_prec.tolist(),
        "per_class_recall": class_recall.tolist(),
        "per_class_f1": class_f1.tolist(),
        "per_class_support": class_support.tolist(),
    }

    save_json(metrics, metrics_path)
    print(f"  指标 -> {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_scale", type=int, default=None,
                        help="训练数据量（条数），不传则使用全量")
    args = parser.parse_args()
    train(data_scale=args.data_scale)
