# THUCNews 新闻文本分类

基于 **THUCNews** 数据集的新闻文本分类实验，实现并对比三种文本分类方法在不同数据量下的表现。

## 项目结构

```text
nlp/
├── data/                    # 数据目录
│   ├── raw/                 # 原始文本（按类别分目录）
│   │   ├── 体育/
│   │   ├── 娱乐/
│   │   ├── 家居/
│   │   ├── 彩票/
│   │   ├── 房产/
│   │   ├── 教育/
│   │   ├── 时尚/
│   │   ├── 时政/
│   │   ├── 星座/
│   │   ├── 游戏/
│   │   ├── 社会/
│   │   ├── 科技/
│   │   ├── 股票/
│   │   └── 财经/
│   ├── train.csv            # 训练集（8:1:1 分层抽样）
│   ├── valid.csv            # 验证集
│   └── test.csv             # 测试集
├── src/                     # 源代码
│   ├── data.py              # 数据清洗、预处理、分词、分割
│   ├── config.py            # 全局配置与超参数
│   ├── utils.py             # 工具函数（种子、计时、绘图）
│   ├── tfidf_svm.py         # TF-IDF + SVM 基线模型
│   ├── bilstm.py            # BiLSTM 深度学习模型
│   ├── bert_model.py        # BERT 微调模型
│   └── evaluate.py          # 统一评估与对比报告生成
├── models/                  # 训练好的模型与指标
│   ├── label_mapping.json   # 类别名称 ↔ ID 映射
│   ├── svm/                 # SVM 模型（按数据量子目录）
│   ├── bilstm/              # BiLSTM 模型
│   └── bert/                # BERT 模型
├── results/                 # 评估结果与可视化
│   ├── full/                # 全量数据结果
│   ├── 20000/               # 各数据量实验结果
│   ├── 50000/
│   ├── 100000/
│   ├── 200000/
│   └── 400000/
├── requirements.txt         # Python 依赖
└── readme.md                # 本文件
```

## 数据集

**THUCNews** 是清华大学发布的新闻文本分类数据集，包含 14 个类别：

| 类别  | 说明          | 类别  | 说明            |
|-------|---------------|-------|-----------------|
| 体育  | Sports        | 娱乐  | Entertainment   |
| 家居  | Home          | 彩票  | Lottery         |
| 房产  | Real Estate   | 教育  | Education       |
| 时尚  | Fashion       | 时政  | Politics        |
| 星座  | Horoscope     | 游戏  | Gaming          |
| 社会  | Society       | 科技  | Technology      |
| 股票  | Stock         | 财经  | Finance         |

**数据规模**: 约 66.5 万条新闻文本，按 **8:1:1** 分层抽样划分训练/验证/测试集。

## 数据预处理流程

`data.py` 实现了完整的文本清洗与预处理流水线：

1. **文本规范化**
   - Unicode NFKC 归一化
   - HTML 标签去除
   - URL 和邮箱地址去除
   - 全角字母数字 → 半角
   - 连续重复字符压缩
2. **数据质量过滤**
   - 空白/超短文本过滤（最短 10~30 字符）
   - 中文字符占比 < 30% 的低质量文本过滤
   - 超长文本截断（50000 字符上限）
3. **文本去重** — 基于文本内容去除完全重复样本
4. **分词** — jieba 分词 + 去停用词（内置 100+ 中文停用词表）
5. **双列存储**
   - `raw_text` — 原始文本（供 BERT 使用）
   - `text` — 分词后文本（供 BiLSTM / SVM 使用）

## 模型

### 1. TF-IDF + SVM（基线模型）

- **特征**: TF-IDF (max_features=100000, unigram + bigram, sublinear_tf)
- **分类器**: LinearSVC (C=1.0, class_weight="balanced", max_iter=2000)
- **优点**: 训练快、可解释性强、小数据量下表现稳定

### 2. BiLSTM（深度学习模型）

- **架构**: Embedding → 双向 LSTM (2层, 512维) → Pooling → LayerNorm → FC
- **Pooling 策略**: mean_max（同时使用最大池化和平均池化拼接）
- **优化**: AdamW, ReduceLROnPlateau, Early Stopping, Gradient Clipping
- **正则化**: Embedding Dropout (0.2), LSTM Dropout (0.2), Label Smoothing (0.1), L2 (1e-4)
- **输入**: jieba 分词后的文本序列（max_len=48, vocab_size=80000）
- **适用于**: 需要捕捉语义但资源有限的场景

### 3. BERT（预训练微调）

- **模型**: `hfl/chinese-roberta-wwm-ext`（哈工大讯飞联合发布的中文 RoBERTa）
- **训练**: 全参数微调，3 epochs，线性学习率调度 + warmup
- **精度**: 混合精度训练 (fp16)
- **优点**: 效果最佳，上下文理解最强
- **代价**: 训练/推理慢，需要 GPU

## 实验结果

评估使用**完整测试集（83,111 条）** 统一评测，`num_workers=4`。详见 `results/` 下各数据量独立目录中的 `report.md`、混淆矩阵及训练曲线。

### 全量数据 (约 66.5 万条)

| 模型         | Accuracy | Macro F1 | Macro Precision | Macro Recall | 推理时间 |
|--------------|----------|----------|----------------|-------------|----------|
| **BERT**     | **0.9761** | **0.9738** | 0.9754 | 0.9722 | 127.65s |
| TF-IDF+SVM   | 0.9557   | 0.9503   | 0.9479 | 0.9528 | 0.19s |
| BiLSTM       | 0.9507   | 0.9381   | 0.9250 | 0.9526 | 2.05s |

### 不同数据量下的 F1 表现

| 数据量       | TF-IDF+SVM | BiLSTM | BERT   |
|--------------|------------|--------|--------|
| 20k          | 0.9228     | 0.8273 | 0.9438 |
| 50k          | 0.9333     | —      | —      |
| 100k         | 0.9393     | —      | —      |
| 200k         | 0.9434     | —      | —      |
| 400k         | 0.9484     | —      | —      |
| 全量(~665k)  | 0.9503     | 0.9381 | 0.9738 |

> 各模型数据量影响分析报告：`results/tf_idf_svm_data_scale_analysis.md`、`results/bilstm_data_scale_analysis.md`、`results/bert_data_scale_analysis.md`

### 关键结论

- **BERT 全量数据效果最佳**（F1 = 0.9738），显著优于传统方法和 BiLSTM
- **TF-IDF+SVM 性价比极高**：全量数据 F1=0.9503，训练仅数秒；20k 数据推理仅 0.17s 即可达全量 BERT 94.8% 的 F1
- **数据量增益递减**：SVM 从 20k→100k 提升 +1.6%，200k→全量提升不到 +0.7%
- **BiLSTM** 在小数据量（20k）下表现较弱（F1=0.8273），全量数据提升到 0.9381
- **BERT 小数据量优势突出**：20k 时 F1=0.9438，预训练知识在小样本场景价值显著

## 使用方法

### 环境要求

```bash
pip install -r requirements.txt
```

依赖：scikit-learn, torch, transformers, jieba, pandas, numpy, matplotlib, seaborn, tqdm

### 数据预处理

```bash
python src/data.py
```

从 `data/raw/` 读取原始文本，经过清洗、分词后生成 `train.csv` / `valid.csv` / `test.csv`。

### 训练模型

```bash
# TF-IDF + SVM（全量数据）
python src/tfidf_svm.py

# BiLSTM（全量数据）
python src/bilstm.py

# BERT（全量数据）
python src/bert_model.py

# 指定数据量（例如 50000 条）
python src/tfidf_svm.py --data_scale 50000
python src/bilstm.py --data_scale 20000
python src/bert_model.py --data_scale 50000
```

### 评估与报告

```bash
# 评估所有可用模型
python src/evaluate.py

# 仅评估指定模型或数据量
python src/evaluate.py --model bilstm
python src/evaluate.py --scale 20000
```

评估脚本自动生成：

- 各数据量下的模型对比报告（markdown）
- 混淆矩阵、类别 F1 柱状图
- 数据量 vs 性能曲线
- 训练 Loss / Accuracy 曲线

## 配置

所有超参数集中在 `src/config.py`，包括：

- **数据预处理**: 文本长度阈值、规范化开关、去重开关
- **TF-IDF+SVM**: 最大特征数、n-gram 范围
- **BiLSTM**: 词表大小、Embedding/LSTM 维度、Dropout、Batch Size、Pooling 方式
- **BERT**: 模型名称、序列长度、Batch Size、学习率
- **通用**: 设备选择、Early Stopping 策略
- **数据量实验**: `DATA_SCALE_OPTIONS` 列表控制实验规模

## License

仅供学习和研究使用。
