#!/usr/bin/env python3
"""BiLSTM 模型——定义、训练与评估"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    TRAIN_CSV, VALID_CSV, TEST_CSV, BILSTM_MODEL_PATH,
    BILSTM_MAX_LEN, BILSTM_EMBEDDING_DIM, BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS, BILSTM_DROPOUT, BILSTM_BATCH_SIZE,
    BILSTM_EPOCHS, BILSTM_LR, BILSTM_VOCAB_SIZE, RANDOM_SEED,
    EARLY_STOP_PATIENCE, DEVICE, ensure_dirs,
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
    """BiLSTM 文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为双向

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden*2)
        # 全局最大池化
        pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.size(1)).squeeze(-1)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits


# ── 训练 ──

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
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


def train(data_scale=None):
    """训练 BiLSTM 模型"""
    ensure_dirs()
    seed_everything(RANDOM_SEED)

    print("=" * 50)
    print("BiLSTM 训练")
    print("=" * 50)
    print(f"设备: {DEVICE}")

    # 1. 加载数据
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if data_scale and len(train_df) > data_scale:
        train_df = train_df.sample(n=data_scale, random_state=RANDOM_SEED)
        print(f"采样训练数据: {data_scale} 条")

    num_classes = max(train_df["label"].max(), valid_df["label"].max(), test_df["label"].max()) + 1
    print(f"类别数: {num_classes}")
    print(f"训练集: {len(train_df)}, 验证集: {len(valid_df)}, 测试集: {len(test_df)}")

    # 2. 构建词表
    print("\n构建词表...")
    vocab = build_vocab(train_df["text"].values, BILSTM_VOCAB_SIZE)
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    # 3. 数据加载器
    batch_size = BILSTM_BATCH_SIZE
    train_dataset = TextDataset(train_df["text"].values, train_df["label"].values, vocab, BILSTM_MAX_LEN)
    valid_dataset = TextDataset(valid_df["text"].values, valid_df["label"].values, vocab, BILSTM_MAX_LEN)
    test_dataset = TextDataset(test_df["text"].values, test_df["label"].values, vocab, BILSTM_MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 4. 初始化模型
    print("\n初始化 BiLSTM 模型...")
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=BILSTM_EMBEDDING_DIM,
        hidden_dim=BILSTM_HIDDEN_DIM,
        num_layers=BILSTM_NUM_LAYERS,
        num_classes=num_classes,
        dropout=BILSTM_DROPOUT,
    ).to(DEVICE)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BILSTM_LR)

    # 5. 训练循环
    print(f"\n开始训练 ({BILSTM_EPOCHS} epochs)...")
    timer = Timer().tic()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, BILSTM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"  Epoch {epoch:2d}/{BILSTM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

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

    # 6. 加载最佳模型并评估
    print(f"\n加载最佳模型 (epoch {best_epoch})...")
    model.load_state_dict(torch.load(str(BILSTM_MODEL_PATH), map_location=DEVICE))

    print("测试集评估...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

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

    scale_suffix = f"_{data_scale}" if data_scale else ""
    model_path = str(BILSTM_MODEL_PATH).replace(".pt", f"{scale_suffix}.pt")
    metrics_path = model_path + ".metrics.json"

    # data_scale 时保存模型到独立文件，避免覆盖全量模型
    if data_scale:
        torch.save(model.state_dict(), model_path)
        print(f"  模型 -> {model_path}")

    metrics = {
        "model": "BiLSTM",
        "data_scale": data_scale if data_scale else "full",
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "train_time_sec": round(train_time, 2),
        "history": history,
        "per_class_precision": class_prec.tolist(),
        "per_class_recall": class_recall.tolist(),
        "per_class_f1": class_f1.tolist(),
        "per_class_support": class_support.tolist(),
        "best_epoch": best_epoch,
        "test_loss": round(test_loss, 4),
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
