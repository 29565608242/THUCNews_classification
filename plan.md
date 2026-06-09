# THUCNews 新闻文本分类实验计划

## 限制

- 使用 `uav` 这个 conda 环境，如果有不存在的 python 包，使用清华源安装
- 所有代码使用 Python 编写

## 0. 项目目录结构

```
nlp/
├── plan.md                  # 本实验计划文档
├── requirements.txt         # Python 依赖
├── src/                     # 源代码目录
│   ├── config.py            # 全局配置（路径、超参数）
│   ├── data.py              # 数据预处理（清洗、分词、去停用词、划分）
│   ├── tfidf_svm.py         # TF-IDF + SVM 训练与评估
│   ├── bilstm.py            # BiLSTM 模型定义、训练与评估
│   ├── bert_model.py        # BERT 模型定义、训练与评估
│   ├── evaluate.py          # 评估指标、混淆矩阵、可视化
│   └── utils.py             # 工具函数（计时、日志等）
├── data/                    # 数据集（原始 THUCNews + 处理后 CSV）
│   ├── raw/                 # 原始 THUCNews 数据
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── models/                  # 保存训练好的模型文件
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── bilstm_model.pt
│   └── bert_model.pt
└── results/                 # 实验结果与可视化
    ├── metrics.json         # 所有指标汇总
    ├── confusion_matrix.png # 混淆矩阵热力图
    ├── loss_curve.png       # BiLSTM/BERT 损失曲线
    ├── category_f1.png      # 各类别 F1 热力图
    ├── data_scale_vs_f1.png # 数据量 vs Macro-F1 折线图
    └── report.md            # 实验报告草稿
```

## 1. 实验目标

- 基于 THUCNews 完整数据集，实现三个模型的新闻文本分类：
  1. **TF-IDF + SVM**（传统机器学习基线）
  2. **BiLSTM**（深度学习序列模型）
  3. **BERT**（预训练语言模型，作为性能最强模型）
- 对比三个模型在不同类别上的表现和数据量对分类性能的影响。
- 输出指标：准确率、宏平均精确率、召回率、F1、混淆矩阵、训练时间、推理时间。

---

## 2. 数据准备

- 数据集：THUCNews 完整数据集
  - 来源：THUCNews 包含约 74 万条新闻文本，涵盖 14 个类别（体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经、社会、星座、彩票、股票等）
  - 原始格式：每个类别一个文件夹，每个文件一条新闻（纯文本）
  - 如果总数据量过大（74 万条），可随机采样 5-10 万条保持类别平衡后再做划分，以控制实验时间
  - 最终类别数依据实际数据确定，建议不低于 10 个类别
- 数据处理（实现于 `src/data.py`）：
  - 文本清洗：去除多余空格、特殊符号、HTML 标签、URL
  - 分词：使用 jieba 精确模式分词
  - 去停用词：使用停用词表（如哈工大停用词表）
  - 标签编码：LabelEncoder 将类别名转为整数标签
  - 保存类别映射关系到 `models/label_mapping.json`
- 数据划分：
  - 训练集 80%
  - 验证集 10%
  - 测试集 10%
- 保存处理后的 CSV 文件到 `data/` 目录：
  - `train.csv`：text（分词后）, label（编码后）, label_name（原始类别名）
  - `valid.csv`：同上
  - `test.csv`：同上

## 3. 环境与依赖

- **conda 环境**：`uav`（Python 3.8+）
- **核心依赖**（写入 `requirements.txt`）：

| 包名            | 用途                         | 安装方式                              |
|-----------------|------------------------------|---------------------------------------|
| scikit-learn    | TF-IDF + SVM 模型            | `pip install scikit-learn`            |
| torch           | BiLSTM + BERT 训练           | `pip install torch`                   |
| transformers    | BERT 模型与 tokenizer        | `pip install transformers`            |
| jieba           | 中文分词                     | `pip install jieba`                   |
| pandas          | 数据处理                     | `pip install pandas`                  |
| numpy           | 数值计算                     | `pip install numpy`                   |
| matplotlib      | 可视化（混淆矩阵、曲线）     | `pip install matplotlib`              |
| seaborn         | 热力图美化                   | `pip install seaborn`                 |
| tqdm            | 进度条                       | `pip install tqdm`                    |

- 安装命令（清华源）：

  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

---

## 4. 模型定义与边界

### 4.1 TF-IDF + SVM

- 特征：TF-IDF (max_features=100000, ngram=(1,2))
- 分类器：LinearSVC
- 边界：
  - 不训练深度模型
  - 使用分词 + 去停用词文本作为输入
  - 输出训练时间、推理时间、指标

### 4.2 BiLSTM

- 输入：分词 + 去停用词文本转为词 ID 序列
- 模型结构：
  - Embedding 层
  - 双向 LSTM
  - 最大池化
  - Dropout
  - 全连接分类层
- 超参数：
  - max_len=300, embedding_dim=300, hidden_dim=256, num_layers=2, dropout=0.5
  - batch_size=128, epochs=10, optimizer=Adam
- 边界：
  - 不使用预训练 Transformer
  - 输出训练/验证指标及损失曲线

### 4.3 BERT

- 预训练模型：`hfl/chinese-roberta-wwm-ext`
- 输入：原始文本（使用 BERT tokenizer）
- 模型结构：
  - BERT encoder
  - Dropout
  - 全连接分类层
- 超参数：
  - max_len=256, batch_size=16, epochs=3-5, optimizer=AdamW, learning_rate=2e-5
- 边界：
  - 不做额外微调任务
  - 仅在训练集上微调
  - 输出训练/验证指标及损失曲线

---

## 5. 实验流程

### 5.1 数据预处理

- **脚本**：`python src/data.py`
- **步骤**：
  1. 读取 `data/raw/` 下的原始 THUCNews 数据（按类别文件夹遍历）
  2. 文本清洗：去空格、特殊符号、HTML 标签、URL
  3. jieba 分词 + 去停用词
  4. LabelEncoder 标签编码
  5. 随机打乱后按 8:1:1 划分 train/valid/test
  6. 保存到 `data/` 目录（train.csv / valid.csv / test.csv）
  7. 保存标签映射到 `models/label_mapping.json`
- **输出**：`data/train.csv`、`data/valid.csv`、`data/test.csv`、`models/label_mapping.json`
- **关键配置**（在 `src/config.py` 中定义）：
  - `RAW_DATA_DIR = "data/raw"`
  - `TRAIN_RATIO = 0.8`, `VAL_RATIO = 0.1`, `TEST_RATIO = 0.1`
  - `MAX_SAMPLES = None`（设为整数可限制总样本数以加速）
  - `RANDOM_SEED = 42`

### 5.2 模型训练

#### 5.2.1 TF-IDF + SVM

- **脚本**：`python src/tfidf_svm.py`
- **步骤**：
  1. 从 `data/train.csv` 读取训练文本与标签
  2. 使用 TfidfVectorizer 提取 TF-IDF 特征（max_features=100000, ngram_range=(1,2)）
  3. 训练 LinearSVC 分类器
  4. 在测试集上评估
  5. 保存模型与评估结果
- **输出**：
  - `models/tfidf_vectorizer.pkl`
  - `models/svm_model.pkl`
  - 测试集指标

#### 5.2.2 BiLSTM

- **脚本**：`python src/bilstm.py`
- **步骤**：
  1. 构建词表（基于训练集，设置 vocab_size 上限）
  2. 将文本转为词 ID 序列，padding 到 max_len=300
  3. 构建 BiLSTM 模型：Embedding → BiLSTM → MaxPooling → Dropout → FC
  4. 训练 10 个 epoch，记录每轮 train/valid loss 与 accuracy
  5. 保存最佳模型（基于 valid 指标）
- **输出**：
  - `models/bilstm_model.pt`
  - 损失曲线数据
  - 测试集指标
- **训练配置**：
  - GPU 可用时自动使用 CUDA
  - Early stopping（patience=3）防止过拟合
  - 保存最佳 checkpoint

#### 5.2.3 BERT

- **脚本**：`python src/bert_model.py`
- **步骤**：
  1. 加载 `hfl/chinese-roberta-wwm-ext` 预训练模型与 tokenizer
  2. 使用原始文本（不分词），BERT tokenizer 编码为 input_ids + attention_mask
  3. 构建 BERT 分类模型（BERT encoder + Dropout + FC）
  4. 训练 3-5 个 epoch，记录每轮损失与指标
  5. 保存最佳模型
- **输出**：
  - `models/bert_model.pt`
  - 损失曲线数据
  - 测试集指标
- **资源需求**：
  - 推荐至少 8GB 显存（batch_size=16, max_len=256）
  - 如显存不足，降低 batch_size 或 max_len

### 5.3 模型评估

- **脚本**：`python src/evaluate.py`
- **步骤**：
  1. 对三个模型分别加载最佳 checkpoint
  2. 在测试集上计算：Accuracy、Macro Precision、Macro Recall、Macro F1
  3. 绘制混淆矩阵热力图（归一化 + 原始计数两张子图）
  4. BiLSTM 和 BERT 绘制训练/验证损失曲线
  5. 输出每个类别的 Precision/Recall/F1，按 F1 排序
  6. 汇总所有指标到 `results/metrics.json`

### 5.4 数据量影响实验

- **执行方式**：通过修改 `src/config.py` 中的 `DATA_SCALE` 参数，或通过命令行参数 `--data_scale 5000` 控制
- **实验方案**：

| 数据量 | TF-IDF + SVM | BiLSTM | BERT |
|--------|:------------:|:------:|:----:|
| 5000   | ✅ | — | ✅ |
| 10000  | ✅ | ✅ | ✅ |
| 20000  | ✅ | ✅ | ✅ |
| 全量   | ✅ | ✅ | 视显存 |

- **输出**：`results/data_scale_vs_f1.png`（各模型 Macro-F1 随数据量变化折线图）

### 5.5 执行命令汇总

```bash
# 1. 数据预处理
python src/data.py

# 2. TF-IDF + SVM
python src/tfidf_svm.py

# 3. BiLSTM
python src/bilstm.py

# 4. BERT
python src/bert_model.py

# 5. 评估与可视化
python src/evaluate.py

# 6. 数据量实验（示例：5000 条）
python src/tfidf_svm.py --data_scale 5000
python src/bilstm.py --data_scale 10000
python src/bert_model.py --data_scale 5000
```

> **说明**：每个脚本独立可运行，不依赖前序脚本的执行状态（会自动加载已保存的数据和模型）。数据量实验通过 `--data_scale` 参数控制，不传则使用全量数据。

---

## 6. 输出与提交

所有输出文件按以下结构组织：

### 6.1 模型文件（`models/` 目录）

| 文件 | 来源 | 说明 |
|------|------|------|
| `models/svm_model.pkl` | TF-IDF + SVM | LinearSVC 分类器（pickle 格式） |
| `models/tfidf_vectorizer.pkl` | TF-IDF + SVM | TF-IDF 特征提取器（pickle 格式） |
| `models/bilstm_model.pt` | BiLSTM | PyTorch 模型权重（state_dict） |
| `models/bert_model.pt` | BERT | PyTorch 模型权重（state_dict） |
| `models/label_mapping.json` | 数据预处理 | 类别名到整数标签的映射 |

### 6.2 指标与可视化（`results/` 目录）

| 文件 | 说明 |
|------|------|
| `results/metrics.json` | 所有模型在各数据量下的完整指标汇总 |
| `results/confusion_matrix.png` | 三模型混淆矩阵对比图（2×2 子图布局） |
| `results/loss_curve.png` | BiLSTM 和 BERT 的训练/验证损失曲线 |
| `results/category_f1.png` | 各类别 F1 热力图（按模型分组） |
| `results/data_scale_vs_f1.png` | 数据量 vs Macro-F1 折线图 |

### 6.3 实验报告

生成 `results/report.md` 作为实验报告草稿，包含：
- **1. 引言**：实验目的、数据集简介
- **2. 数据集与预处理**：数据统计、预处理方法、类别分布
- **3. 模型结构与参数**：三个模型的结构描述与超参数表
- **4. 实验结果与分析**：
  - 三模型整体指标对比表（Accuracy / Macro-P / Macro-R / Macro-F1）
  - 训练时间与推理时间对比
  - 混淆矩阵分析
- **5. 类别差异分析**：各类别 F1 排序、难易类别分析、混淆模式
- **6. 数据量影响分析**：各模型 Macro-F1 随数据量变化趋势与分析
- **7. 结论**：三模型对比总结、最佳实践建议

可基于 `report.md` 后续导出为 Word/PDF。

---

## 7. 执行边界（Agent可控）

### ✅ Agent 负责范围

- 代码编写：实现 `src/` 目录下所有 Python 脚本
- 数据预处理：读取原始 THUCNews → 清洗 → 分词 → 去停用词 → 划分 → 保存 CSV
- 模型训练：TF-IDF + SVM / BiLSTM / BERT 三个模型独立训练与调优
- 结果评估：测试集上计算全部指标，输出可视化
- 数据量实验：按计划执行多个数据规模的对比实验并记录
- 实验报告：汇总结果，撰写 `results/report.md`

### ❌ Agent 不涉及范围

- 不做扩增模型或特征工程（如 Word2Vec、FastText、TextCNN 等额外模型）
- 不做模型部署（Flask/FastAPI/ONNX/TensorRT 等）
- 不做超参数大规模搜索（仅使用计划中指定的超参数）
- 不做分布式训练或多卡并行
- 不做生产级优化（量化、剪枝、蒸馏等）
- 不修改已有环境（不创建新 conda 环境，仅安装额外 Python 包）

### ⚠️ 注意事项

- 训练 BERT 时如 GPU 显存不足（< 8GB），自动降低 batch_size 至 8 或 max_len 至 128
- 数据量实验若全量数据过大，TF-IDF + SVM 可降采样至 5 万条训练（避免内存溢出）
- 所有随机操作设置 seed=42，保证实验可复现
- 记录每一步的执行时间和可能的报错信息

---

## 8. 可视化与分析

本实验共生成 5 张可视化图表，统一保存到 `results/` 目录：

### 8.1 混淆矩阵热力图（`results/confusion_matrix.png`）

- 布局：2×2 子图（TF-IDF+SVM / BiLSTM / BERT 各一张 + 右下角放图例或留空）
- 每张子图：横轴为预测类别，纵轴为真实类别，颜色深浅表示归一化后的值（0-1）
- 对角线为正确分类，非对角线为混淆模式
- 分析要点：找出最易混淆的类别对（如"家居"被误判为"房产"）

### 8.2 训练曲线（`results/loss_curve.png`）

- 双轴折线图，左轴为 Loss，右轴为 Accuracy（可选）
- BiLSTM：训练集 loss / 验证集 loss / 验证集 accuracy（10 epoch）
- BERT：训练集 loss / 验证集 loss / 验证集 accuracy（3-5 epoch）
- 分析要点：是否存在过拟合、收敛速度

### 8.3 类别 F1 热力图（`results/category_f1.png`）

- 横轴为三个模型，纵轴为所有类别
- 颜色深浅表示该模型在该类别上的 F1 值
- 分析要点：哪些类别所有模型都表现好/差、哪些类别模型间差异大

### 8.4 数据量影响折线图（`results/data_scale_vs_f1.png`）

- 横轴为数据量（5000 / 10000 / 20000 / 全量，对数刻度）
- 纵轴为 Macro-F1
- 每条线代表一个模型
- 分析要点：数据量增大时各模型收益是否递减、哪个模型在少量数据下表现更好

### 8.5 指标汇总表（`results/metrics.json`）

```json
{
  "tfidf_svm": {
    "accuracy": 0.95,
    "macro_precision": 0.94,
    "macro_recall": 0.93,
    "macro_f1": 0.94,
    "train_time": 120.5,
    "inference_time": 3.2
  },
  "bilstm": {
    "accuracy": 0.97,
    "macro_precision": 0.96,
    "macro_recall": 0.96,
    "macro_f1": 0.96,
    "train_time": 1800.0,
    "inference_time": 15.0
  },
  "bert": {
    "accuracy": 0.98,
    "macro_precision": 0.98,
    "macro_recall": 0.97,
    "macro_f1": 0.97,
    "train_time": 7200.0,
    "inference_time": 120.0
  }
}
```

（以上数据为预期参考值，以实际结果为准）
