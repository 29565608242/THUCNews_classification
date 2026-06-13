# THUCNews 中文新闻文本分类

基于清华大学 THUCNews 数据集的新闻文本分类对比实验，实现并对比三种方法：**TF-IDF+SVM**、**BiLSTM**、**BERT（RoBERTa-wwm-ext）**，在多个数据规模下进行系统评估。

## 数据集

THUCNews 中文新闻分类数据集，来自清华大学 NLP 实验室，包含 14 个新闻类别。

| 项目 | 数值 |
|------|------|
| 类别数 | 14 |
| 总样本数 | **831,101** |
| 训练集 | 664,880（80%） |
| 验证集 | 83,110（10%） |
| 测试集 | 83,111（10%） |
| 划分方式 | 分层随机抽样（8:1:1，seed=42） |

**14 个类别**：体育、娱乐、家居、彩票、房产、教育、时尚、时政、星座、游戏、社会、科技、股票、财经

> ⚠️ 数据集存在严重类别不平衡：样本最多的类别与最少的类别相差约 **45.7 倍**。

### 数据预处理流水线

```
原始文本 → 质量过滤 → 文本规范化 → 文本去重 → jieba分词 → 去停用词 → 双列存储
```

- **质量过滤**：最短 30 字符 / 中文占比 ≥ 30% / 最长 50k 字符
- **文本规范化**：Unicode NFKC 归一化、全角→半角、HTML/URL 去除、连续重复字符压缩
- **文本去重**：基于文本内容的完全去重，防止数据泄露
- **分词**：jieba 精确模式 + 内置停用词表（325 词）
- **双列文本**：
  - `raw_text` — 原始文本（BERT 使用，由其自带 tokenizer 分词）
  - `text` — jieba 分词 + 去停用词后的文本（TF-IDF+SVM、BiLSTM 使用）

## 三种方法

### 1. TF-IDF + SVM

| 参数 | 值 |
|------|-----|
| 特征 | TF-IDF（100k 特征，unigram+bigram，sublinear_tf） |
| 分类器 | LinearSVC（C=1.0，class_weight="balanced"） |

### 2. BiLSTM

```
Embedding(300d, 80k词表) → 2层Bidirectional LSTM(512d)
  → Mean+Max Pooling → LayerNorm → Dropout → FC(14类)
```

| 训练策略 | 配置 |
|----------|------|
| 损失函数 | CrossEntropyLoss + Label Smoothing(0.1) + 类别权重 |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 调度器 | ReduceLROnPlateau (factor=0.5, patience=2) |
| 正则化 | 梯度裁剪(max=5.0) + Early Stopping(patience=6) |
| 序列长度 | max_len=48（覆盖 P99.9） |

### 3. BERT

微调 **hfl/chinese-roberta-wwm-ext**（哈工大讯飞中文 RoBERTa，全词掩码），约 1.1 亿参数。

| 微调策略 | 配置 |
|----------|------|
| 学习率 | 2e-5 + linear warmup(10%) |
| 优化 | fp16 混合精度训练 |
| Epoch | 3（Early Stopping patience=6） |
| 序列长度 | 256 |
| 输入文本 | `raw_text`（BERT 自带 WordPiece tokenizer） |

## 项目结构

```
nlp/
├── data/
│   ├── raw/                  # 14 个类别的原始 .txt 文件
│   ├── train.csv             # 训练集
│   ├── valid.csv             # 验证集
│   └── test.csv              # 测试集
├── src/
│   ├── config.py             # 全局配置（路径、超参数、数据量选项）
│   ├── data.py               # 数据预处理（质量过滤→规范化→去重→分词）
│   ├── utils.py              # 工具函数（随机种子、计时器、JSON、绘图）
│   ├── tfidf_svm.py          # TF-IDF + SVM 训练
│   ├── bilstm.py             # BiLSTM 模型定义与训练
│   ├── bert_model.py         # BERT 微调训练
│   └── evaluate.py           # 统一评估脚本（混淆矩阵、对比曲线、报告生成）
├── models/                   # 训练好的模型文件（按 <方法>/<数据量>/ 组织）
├── results/                  # 评估结果与可视化图表
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据预处理（生成 train.csv / valid.csv / test.csv）
python src/data.py

# 3. 训练模型（--data_scale 控制训练样本数，不传则使用全部 66.5 万训练集）
python src/tfidf_svm.py --data_scale 20000
python src/bilstm.py --data_scale 20000
python src/bert_model.py --data_scale 20000

# 4. 评估所有模型并生成报告与图表
python src/evaluate.py                  # 评估所有可用模型
python src/evaluate.py --scale 20000    # 仅评估指定数据量
python src/evaluate.py --model bilstm   # 仅评估指定模型
```

> **关于 `--data_scale`**：指从完整训练集（664,880 条）中采样的训练样本数量。`--data_scale 20000` 表示随机采样 2 万条训练数据（同时按比例缩减验证集和测试集），用于研究数据量对模型性能的影响。

## 实验结果摘要

> 除标注 `*` 外，所有指标均在完整测试集（83,111 条）上测得。标注 `*` 的数据来自训练时按比例缩减的小测试集。

### 完整数据量表

| 模型 | 数据量 | Accuracy | Macro F1 | Macro Prec | Macro Rec | 训练时间 | 推理时间 |
|------|:------:|:--------:|:--------:|:----------:|:---------:|:--------:|:--------:|
| TF-IDF+SVM | 2万 | 0.9332* | 0.9259* | 0.9289* | 0.9236* | 2s | 0.01s* |
| TF-IDF+SVM | 5万 | 0.9443* | 0.9347* | 0.9338* | 0.9358* | 6s | 0.02s* |
| TF-IDF+SVM | 10万 | 0.9480 | 0.9393 | 0.9405 | 0.9382 | 40s | 0.18s |
| TF-IDF+SVM | 20万 | 0.9512 | 0.9434 | 0.9436 | 0.9434 | 82s | 0.18s |
| TF-IDF+SVM | 40万 | 0.9546 | 0.9484 | 0.9468 | 0.9502 | 174s | 0.18s |
| TF-IDF+SVM | 全量 | 0.9557 | 0.9503 | 0.9479 | 0.9528 | 293s | 0.18s |
| BiLSTM | 2万 | 0.8837 | 0.8273 | 0.8271 | 0.8770 | 71s | 2.45s |
| BiLSTM | 5万 | 0.9143 | 0.8691 | 0.8556 | 0.9054 | 109s | 2.23s |
| BiLSTM | 10万 | 0.9194 | 0.8918 | 0.8658 | 0.9236 | 105s | 2.0s |
| BiLSTM | 20万 | 0.9356 | 0.9144 | 0.8939 | 0.9386 | 212s | 2.0s |
| BiLSTM | 40万 | 0.9487 | 0.9348 | 0.9235 | 0.9474 | 451s | 2.0s |
| BiLSTM | 全量 | 0.9507 | 0.9381 | 0.9250 | 0.9526 | 714s | 2.0s |
| BERT | 2万 | 0.9504* | 0.9416* | 0.9370* | 0.9472* | 149s | ~127s |
| BERT | 全量 | 0.9761 | 0.9738 | 0.9754 | 0.9722 | 4631s | 127s |

### 全量数据三模型对比

| 模型 | Accuracy | Macro F1 | Macro Prec | Macro Rec | 训练时间 | 推理时间 |
|------|:--------:|:--------:|:----------:|:---------:|:--------:|:--------:|
| TF-IDF+SVM | 0.9557 | 0.9503 | 0.9479 | 0.9528 | 4.9 min | **0.18s** |
| BiLSTM | 0.9507 | 0.9381 | 0.9250 | 0.9526 | 11.9 min | 2.0s |
| BERT | **0.9761** | **0.9738** | **0.9754** | **0.9722** | 77 min | 127s |

## 关键结论

- 🏆 **BERT 性能最强**：全量数据 F1=0.9738，2万条仍达 0.9416
- ⚡ **SVM 性价比最高**：95% 的 F1，训练快 16 倍，推理快 700 倍
- 📈 **BiLSTM 依赖数据量**：2万→全量 F1 从 0.8273 提升至 0.9381（+11.1pp），但无预训练词向量，全量下仍低于 SVM
- 📉 **收益递减**：所有模型随数据量增加性能提升趋缓

## 环境要求

- Python 3.8+
- PyTorch 2.x +
- scikit-learn 1.x
- jieba 0.42
- GPU：RTX 4090D × 2（BiLSTM 用 GPU 0，BERT 用 GPU 1）
- BERT 模型下载使用 HF 镜像站（`HF_ENDPOINT=https://hf-mirror.com`）
