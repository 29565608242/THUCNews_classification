"""模型加载与推理——SVM / BiLSTM / BERT 模型加载和预测"""

import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    TRAIN_CSV,
    SVM_MODEL_PATH, TFIDF_PATH, BILSTM_MODEL_PATH, BERT_MODEL_PATH,
    BILSTM_MAX_LEN, BILSTM_EMBEDDING_DIM, BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS, BILSTM_DROPOUT, BILSTM_DROPOUT_EMBED,
    BILSTM_VOCAB_SIZE, BILSTM_POOLING, BILSTM_BATCH_SIZE,
    BERT_MAX_LEN, BERT_BATCH_SIZE,
    DEVICE, LABEL_MAPPING_PATH,
)


# ══════════════════════════════════════════════════════
# 标签映射
# ══════════════════════════════════════════════════════

def load_label_mapping() -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """加载类别映射，返回类别名称列表、id->name 映射和 name->id 映射"""
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id_to_name = {int(v): k for k, v in mapping.items()}
    class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    return class_names, id_to_name, mapping


# ══════════════════════════════════════════════════════
# BiLSTM 模型定义（与 bilstm.py 一致，用于加载 state_dict）
# ══════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """BiLSTM 文本分类模型（支持多种 pooling 策略）——与训练脚本定义一致"""

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


# ══════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════

def load_svm(data_scale: Optional[int] = None) -> Tuple:
    """加载 TF-IDF + SVM 模型"""
    if data_scale:
        scale_dir = SVM_MODEL_PATH.parent / str(data_scale)
        tfidf_path = scale_dir / "tfidf_vectorizer.pkl"
        model_path = scale_dir / "model.pkl"
    else:
        tfidf_path = TFIDF_PATH
        model_path = SVM_MODEL_PATH
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def load_bilstm(vocab_size: int, num_classes: int, data_scale: Optional[int] = None) -> nn.Module:
    """加载 BiLSTM 模型"""
    if data_scale:
        model_path = BILSTM_MODEL_PATH.parent / str(data_scale) / "model.pt"
    else:
        model_path = BILSTM_MODEL_PATH
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
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.eval()
    return model


def load_bert(num_classes: int, data_scale: Optional[int] = None):
    """加载 BERT 模型与 tokenizer"""
    if data_scale:
        model_dir = BERT_MODEL_PATH.parent / str(data_scale) / "model"
    else:
        model_dir = BERT_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=num_classes
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


# ══════════════════════════════════════════════════════
# 推理
# ══════════════════════════════════════════════════════

def predict_with_model(model_name: str, texts: List[str],
                       data_scale: Optional[int] = None) -> np.ndarray:
    """用指定模型对文本列表进行预测，返回预测标签"""
    if model_name == "svm":
        vectorizer, model = load_svm(data_scale=data_scale)
        X = vectorizer.transform(texts)
        return model.predict(X)

    elif model_name == "bilstm":
        # 重建 word2idx
        train_df = pd.read_csv(TRAIN_CSV)
        from collections import Counter
        counter = Counter()
        for t in train_df["text"].values:
            counter.update(t.split())
        most_common = counter.most_common(BILSTM_VOCAB_SIZE - 2)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w, _ in most_common:
            word2idx[w] = len(word2idx)

        num_classes = pd.read_csv(TRAIN_CSV)["label_name"].nunique()
        model = load_bilstm(len(word2idx), num_classes, data_scale=data_scale)

        class _InferenceDataset(torch.utils.data.Dataset):
            def __init__(self, texts, word2idx, max_len):
                self.data = []
                for text in texts:
                    ids = [word2idx.get(w, 1) for w in text.split()]
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                    else:
                        ids = ids + [0] * (max_len - len(ids))
                    self.data.append(torch.tensor(ids, dtype=torch.long))
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        dataset = _InferenceDataset(texts, word2idx, BILSTM_MAX_LEN)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BILSTM_BATCH_SIZE, num_workers=4,
        )

        preds = []
        with torch.no_grad():
            for inputs in tqdm(loader, desc="BiLSTM 推理", leave=False):
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        return np.array(preds)

    elif model_name == "bert":
        num_classes = pd.read_csv(TRAIN_CSV)["label_name"].nunique()
        model, tokenizer = load_bert(num_classes, data_scale=data_scale)

        preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), BERT_BATCH_SIZE), desc="BERT 推理", leave=False):
                batch_texts = texts[i:i + BERT_BATCH_SIZE]
                enc = tokenizer(
                    batch_texts, truncation=True, padding="max_length",
                    max_length=BERT_MAX_LEN, return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist())
        return np.array(preds)

    else:
        raise ValueError(f"未知模型: {model_name}")
