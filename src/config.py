"""全局配置——路径、超参数、数据规模"""

import os
from pathlib import Path

# ── 项目根目录 ──
ROOT = Path(__file__).resolve().parent.parent

# ── 数据路径 ──
RAW_DATA_DIR = ROOT / "data" / "raw"
TRAIN_CSV = ROOT / "data" / "train.csv"
VALID_CSV = ROOT / "data" / "valid.csv"
TEST_CSV = ROOT / "data" / "test.csv"

# ── 模型保存路径（各模型分不同子目录）──
MODEL_DIR = ROOT / "models"
MODEL_SVM_DIR = MODEL_DIR / "svm"
MODEL_BILSTM_DIR = MODEL_DIR / "bilstm"
MODEL_BERT_DIR = MODEL_DIR / "bert"

SVM_MODEL_PATH = MODEL_SVM_DIR / "model.pkl"
TFIDF_PATH = MODEL_SVM_DIR / "tfidf_vectorizer.pkl"
SVM_METRICS_PATH = MODEL_SVM_DIR / "metrics.json"

BILSTM_MODEL_PATH = MODEL_BILSTM_DIR / "model.pt"
BILSTM_METRICS_PATH = MODEL_BILSTM_DIR / "metrics.json"
BILSTM_LOSS_CURVE = MODEL_BILSTM_DIR / "loss_curve.png"

BERT_MODEL_PATH = MODEL_BERT_DIR / "model"
BERT_METRICS_PATH = MODEL_BERT_DIR / "metrics.json"
BERT_LOSS_CURVE = MODEL_BERT_DIR / "loss_curve.png"

LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"

# ── 结果输出路径 ──
RESULT_DIR = ROOT / "results"
METRICS_JSON = RESULT_DIR / "metrics.json"
CONFUSION_MATRIX_PNG = RESULT_DIR / "confusion_matrix.png"
LOSS_CURVE_PNG = RESULT_DIR / "loss_curve.png"
CATEGORY_F1_PNG = RESULT_DIR / "category_f1.png"
DATA_SCALE_PNG = RESULT_DIR / "data_scale_vs_f1.png"
REPORT_MD = RESULT_DIR / "report.md"

# ── 数据预处理参数 ──
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
MAX_SAMPLES = None           # 从原始数据中最多读取的样本数；None 表示不限制
STOPWORDS_FILE = None        # 停用词表路径；None 使用内置默认列表
MIN_TEXT_LENGTH = 10          # 最短文本长度（字符数）
MIN_CHINESE_RATIO = 0.3       # 最低中文字符占比
MAX_TEXT_LENGTH = 50000       # 最长文本长度（字符数），超出截断

# ── 文本规范化开关（rebuild_data.py 使用）──
NORMALIZE_HTML = True         # 去除 HTML 标签
NORMALIZE_URL = True          # 去除 URL 和邮箱地址
NORMALIZE_FULLWIDTH = True    # 全角字母数字→半角
NORMALIZE_REPEAT = True       # 压缩连续重复字符
DEDUP_ENABLED = True          # 训练前基于文本去重

# ── TF-IDF + SVM 参数 ──
TFIDF_MAX_FEATURES = 100000
TFIDF_NGRAM_RANGE = (1, 2)

# ── BiLSTM 参数（全量数据 66w 适配）──
BILSTM_MAX_LEN = 48               # 200→48: 覆盖 P99.9(36)，大幅减少 padding 噪声
BILSTM_EMBEDDING_DIM = 300
BILSTM_HIDDEN_DIM = 512            # 256→512: 增大容量，解决欠拟合
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.2              # 0.3→0.2: 66w 数据下降一点正则化
BILSTM_DROPOUT_EMBED = 0.2        # Embedding 层 dropout
BILSTM_BATCH_SIZE = 1024           # 256→1024: 4 卡 DataParallel 每卡 256
BILSTM_EPOCHS = 20
BILSTM_LR = 1e-3
BILSTM_LR_MIN = 1e-5              # 最小学习率
BILSTM_WEIGHT_DECAY = 1e-4        # L2 正则化
BILSTM_USE_SCHEDULER = True       # 使用 ReduceLROnPlateau
BILSTM_VOCAB_SIZE = 80000         # 词表上限
BILSTM_POOLING = "mean_max"        # pooling 方式: "max" | "mean" | "mean_max" | "attention"

# ── BERT 参数（全量数据 66w 适配）──
BERT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
BERT_MAX_LEN = 256                 # 128→256: 保留更多上下文
BERT_BATCH_SIZE = 64               # 32→64: 充分利用 GPU
BERT_EPOCHS = 3
BERT_LR = 2e-5

# ── 训练通用参数 ──
EARLY_STOP_PATIENCE = 6
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ── 数据量实验配置 ──
DATA_SCALE_OPTIONS = [5000, 10000, 20000, None]  # None 表示全量


def ensure_dirs():
    """确保所有需要的目录存在"""
    for d in [RAW_DATA_DIR, MODEL_DIR, MODEL_SVM_DIR, MODEL_BILSTM_DIR, MODEL_BERT_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)
