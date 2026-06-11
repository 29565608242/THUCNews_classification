# THUCNews 新闻文本分类

基于 THUCNews 数据集，对比三种文本分类模型：**TF-IDF + SVM**、**BiLSTM**、**BERT**（Chinese RoBERTa）。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据准备（使用已预处理好的 raw_clean 数据）
python src/prepare_raw_clean.py

# 3-5. 训练三个模型（全量数据）
python src/tfidf_svm.py            # 约 5-15 分钟
python src/bilstm.py                # 约 30-60 分钟（GPU）
python src/bert_model.py            # 约 2-4 小时（GPU，batch=64, fp16）

# 6. 综合评估 + 可视化图表 + 实验报告
python src/evaluate.py
```

---

## 目录结构

```
nlp/
├── data/
│   ├── raw/                        # 原始 THUCNews 数据（14 个类别子目录）
│   ├── raw_clean/                  # 预处理后的数据（直接用于训练）
│   │   ├── train.jsonl / train.txt # 训练集 660,276 条
│   │   ├── dev.jsonl / dev.txt     # 验证集 82,535 条
│   │   ├── test.jsonl / test.txt   # 测试集 82,535 条
│   │   └── label_map.txt           # 14 类标签映射
│   ├── train.csv                   # prepare_raw_clean.py 生成的训练 CSV
│   ├── valid.csv                   # 验证 CSV
│   └── test.csv                    # 测试 CSV
│
├── models/
│   ├── label_mapping.json          # 共享标签映射
│   ├── svm/                        # TF-IDF + SVM
│   │   ├── model.pkl               #   训练好的 SVM 模型
│   │   ├── tfidf_vectorizer.pkl    #   TF-IDF 向量器
│   │   ├── metrics.json            #   指标（性能 + 超参数 + 数据集规模）
│   │   └── loss_curve.png          #   损失曲线（仅 BiLSTM/BERT 有）
│   ├── bilstm/
│   │   ├── model.pt
│   │   ├── metrics.json
│   │   └── loss_curve.png
│   └── bert/
│       ├── model/                  # BERT 模型目录（config.json, safetensors...）
│       ├── metrics.json
│       └── loss_curve.png
│
├── run_data_scale.sh               # 数据量实验一键脚本
│
├── results/                        # 综合评估输出
│   ├── metrics.json                # 三个模型的汇总指标
│   ├── report.md                   # 自动生成的实验报告
│   ├── confusion_matrix.png        # 混淆矩阵对比图
│   ├── loss_curve.png              # 训练损失曲线（BiLSTM + BERT 对比）
│   ├── category_f1.png             # 各类别 F1 热力图
│   └── data_scale_vs_f1.png        # 数据量影响分析图
│
└── src/                            # 源代码
    ├── config.py                   # 全局配置（路径、超参数）
    ├── utils.py                    # 工具函数（计时、种子、绘图等）
    │
    ├── prepare_raw_clean.py        # raw_clean → CSV 转换
    ├── data.py                     # 原始数据预处理流水线（可选）
    │
    ├── tfidf_svm.py                # TF-IDF + SVM 训练
    ├── bilstm.py                   # BiLSTM 训练
    ├── bert_model.py               # BERT 微调
    │
    ├── eval_models.py              # 评估：模型加载与推理（SVM / BiLSTM / BERT）
    ├── eval_visualize.py           # 评估：图表绘制（混淆矩阵、损失曲线、F1 热力图等）
    ├── eval_report.py              # 评估：Markdown 实验报告生成
    └── evaluate.py                 # 评估入口：编排上述模块，执行完整评估流程
```

---

## 数据

### 数据集规模

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| 训练集 | 660,276 | 清洗后全量 |
| 验证集 | 82,535 | 8:1:1 划分 |
| 测试集 | 82,535 | |
| **合计** | **825,346** | |

### 类别分布

> 注：以下为原始数据（原始 836,075 条经质量过滤后约 825k+ 条，数字以实际运行为准）

| 类别 | 训练集样本数 |
|------|-------------|
| 科技 | 129,481 |
| 股票 | 120,486 |
| 体育 | 105,147 |
| 娱乐 | 73,986 |
| 时政 | 50,375 |
| 社会 | 40,652 |
| 教育 | 32,703 |
| 财经 | 29,183 |
| 家居 | 24,050 |
| 游戏 | 19,374 |
| 房产 | 15,875 |
| 时尚 | 10,468 |
| 彩票 | 5,668 |
| 星座 | 2,828 |

> 最大类（科技）与最小类（星座）相差约 **46 倍**，训练时使用类别权重平衡。

### 数据预处理

原始数据经过以下清洗步骤：

1. **质量过滤**：去除空白文件、超短文本（< 10 字符）、低质量文本（中文字符占比 < 30%）
2. **文本规范化**：
   - Unicode NFKC 归一化
   - 去除 HTML 标签（标记 ∥ 段落残留）
   - 去除 URL 和邮箱地址
   - 全角字母数字 → 半角（`ＡＢＣ１２３` → `ABC123`）
   - 压缩连续重复字符（`太长了。。。。。` → `太长了。`，`哈哈哈` 保留两字）
3. **文本去重**：基于完整文本去除完全重复样本，防止训练/测试集数据泄露
4. **jieba 中文分词**
5. **去停用词**：内置 200+ 常用中文停用词表（也支持外部文件加载），同时去除单字噪音
6. **按 8:1:1 分层抽样**划分训练/验证/测试集
7. **双通道输出**：

   - `raw_text` — 原始文本（用于 BERT）
   - `text` — 分词+去停用词后文本（用于 TF-IDF+SVM 和 BiLSTM）

预处理结果已保存在 `data/raw_clean/`，包含两种格式：
- `.txt` — 分词后文本 + 标签ID（用于 TF-IDF+SVM 和 BiLSTM）
- `.jsonl` — 原始文本 + 中文标签名（用于 BERT 的 tokenizer）

可通过 `src/config.py` 中的以下开关控制预处理行为：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `NORMALIZE_HTML` | `True` | 去除 HTML 标签 |
| `NORMALIZE_URL` | `True` | 去除 URL 和邮箱地址 |
| `NORMALIZE_FULLWIDTH` | `True` | 全角字母数字→半角 |
| `NORMALIZE_REPEAT` | `True` | 压缩连续重复字符 |
| `DEDUP_ENABLED` | `True` | 训练前基于文本内容去重 |

---

## 模型

### TF-IDF + SVM

| 参数 | 值 |
|------|-----|
| 特征提取 | TF-IDF + bigram |
| 最大特征数 | 100,000 |
| 分类器 | LinearSVC (C=1.0, class_weight="balanced") |

- **优点**：训练快（分钟级），推理快（毫秒级/篇）
- **缺点**：无法利用词序信息

### BiLSTM

| 参数 | 值 |
|------|-----|
| Embedding | 300 维 |
| LSTM 隐层 | 512 维 × 2 层（双向） |
| Dropout | 0.2 |
| 序列长度 | 48 |
| 批大小 | 1024 |
| 学习率 | 1e-3 |
| 词表 | 80,000 |
| Pooling | mean + max 拼接 |

- **优点**：能建模序列信息
- **缺点**：训练比 SVM 慢，对小类别泛化一般

### BERT (Chinese RoBERTa)

| 参数 | 值 |
|------|-----|
| 预训练模型 | hfl/chinese-roberta-wwm-ext |
| 序列长度 | 256 |
| 批大小 | 64 |
| 学习率 | 2e-5 |
| 训练轮数 | 3（含 Early Stopping）|

- **优点**：利用预训练知识，综合表现最好
- **缺点**：训练最慢，需要 GPU（显存建议 ≥ 8GB）

---

## 训练命令

### 全量训练

```bash
# 每个模型使用全部数据训练
python src/tfidf_svm.py
python src/bilstm.py
python src/bert_model.py
```

训练完成后自动：
- 保存模型权重到 `models/<模型>/` 子目录
- 保存指标 JSON（含性能指标、超参数、数据集规模）
- 绘制损失曲线图到 `models/<模型>/loss_curve.png`（BiLSTM / BERT）

### 数据量实验

支持在数据子集上训练，探索数据量对模型性能的影响。三个数据集（train/valid/test）按 **同比例缩小**。

可用的数据量梯度（在 `src/config.py` 的 `DATA_SCALE_OPTIONS` 中配置）：

```
20000, 50000, 100000, 200000, 400000, 全量(~66.5w)
```

#### 一键跑全部

```bash
bash run_data_scale.sh              # 跑全部模型（SVM → BiLSTM → BERT）
bash run_data_scale.sh svm          # 只跑 SVM
bash run_data_scale.sh bilstm       # 只跑 BiLSTM
bash run_data_scale.sh bert         # 只跑 BERT
bash run_data_scale.sh clean        # 清理旧实验结果
```

#### 手动逐条运行

```bash
python src/tfidf_svm.py --data_scale 20000
python src/bilstm.py   --data_scale 50000
python src/bert_model.py --data_scale 100000
# 不传 --data_scale 表示全量
python src/tfidf_svm.py
```

#### 保存结构

```
models/svm/
├── model.pkl, metrics.json              ← 全量
├── 20000/model.pkl, metrics.json        ← 20k 条实验
├── 50000/...
├── 100000/...
├── 200000/...
└── 400000/...
```

---

## 数据量实验评估

单独分析指定模型在**不同数据量下的性能变化**，生成数据量 vs Macro-F1 曲线图。

### 仅分析数据量（跳过完整评估）

```bash
# 对比指定模型
python src/evaluate.py --data-scale-only --models svm,bilstm

# 只看某个模型
python src/evaluate.py --data-scale-only --models bert

# 全部模型
python src/evaluate.py --data-scale-only
```

### 评估并包含数据量分析

```bash
# 完整评估 + 数据量图（指定模型）
python src/evaluate.py --models svm,bilstm

# 完整评估 + 数据量图（全部模型）
python src/evaluate.py
```

输出：

| 文件 | 内容 |
| --- | --- |
| `results/data_scale_vs_f1.png` | 选定模型的数据量 vs Macro-F1 曲线 |
| `results/metrics.json` | 含数据量实验指标 |

会自动扫描 `models/*/[0-9]*/metrics.json`，绘制 **数据量 vs Macro-F1** 曲线图到 `results/data_scale_vs_f1.png`。

---

## 评估

```bash
python src/evaluate.py
```

### 代码架构

评估功能已拆分为四个模块，各司其职：

| 模块 | 职责 | 关键函数 |
| --- | --- | --- |
| `eval_models.py` | 模型加载、推理 | `load_label_mapping()`, `load_svm()`, `load_bilstm()`, `load_bert()`, `predict_with_model()` |
| `eval_visualize.py` | 图表绘制 | `plot_confusion_matrix()`, `plot_loss_curve()`, `plot_category_f1()`, `plot_data_scale_effect()`, `run_data_scale_analysis()` |
| `eval_report.py` | 报告生成 | `generate_report()` |
| `evaluate.py` | **编排入口**（~240 行） | 导入上述模块，串联评估流程 |

### 评估流程

`evaluate.py` 的 `run()` 函数按以下步骤执行：

1. 确定输出路径（支持 `data_scale` 子目录隔离）
2. 加载标签映射（`eval_models.load_label_mapping`）
3. 遍历指定模型，对每模型执行：
   - 检查模型文件是否存在
   - `predict_with_model()` 批量推理
   - 计算 Accuracy / Macro Precision / Recall / F1
   - 加载训练阶段的 `metrics.json`（含训练时间、损失历史）
4. 保存汇总指标到 `metrics.json`
5. `eval_visualize` — 绘制混淆矩阵、损失曲线、类别 F1 热力图
6. `run_data_scale_analysis` — 扫描数据量实验结果并绘图（仅全量评估）
7. `eval_report.generate_report` — 生成 Markdown 报告

### 输出

所有输出在 `results/` 目录：

| 文件 | 内容 |
|------|------|
| `metrics.json` | 三个模型的性能指标汇总 |
| `report.md` | 完整的实验报告（Markdown） |
| `confusion_matrix.png` | 三个模型的归一化混淆矩阵对比 |
| `loss_curve.png` | BiLSTM / BERT 训练损失曲线 |
| `category_f1.png` | 14 个类别在各模型上的 F1 热力图 |
| `data_scale_vs_f1.png` | 数据量对性能的影响曲线 |

---
