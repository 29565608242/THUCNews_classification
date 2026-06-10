"""工具函数——计时、日志、指标收集、训练曲线绘图"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """固定所有随机种子以保证可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """可手动调用的计时器"""

    def __init__(self):
        self.start = None
        self.elapsed = None

    def tic(self):
        self.start = time.perf_counter()
        return self

    def toc(self, name: str = "Timer"):
        self.elapsed = time.perf_counter() - self.start
        print(f"[{name}] 耗时: {self.elapsed:.2f} 秒")
        return self.elapsed


def save_json(obj: Any, path: str):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MetricsCollector:
    """收集并汇总多个模型/数据量的指标"""

    def __init__(self):
        self._data: List[Dict[str, Any]] = []

    def add(self, model_name: str, data_scale: Optional[int], metrics: Dict[str, Any]):
        """添加一条指标记录"""
        record = {"model": model_name, "data_scale": data_scale, **metrics}
        self._data.append(record)

    def to_dict(self) -> Dict[str, list]:
        """转为按模型分组的字典"""
        grouped: Dict[str, list] = {}
        for rec in self._data:
            key = f"{rec['model']}@{rec['data_scale']}" if rec["data_scale"] else f"{rec['model']}@full"
            grouped[key] = {k: v for k, v in rec.items() if k not in ("model", "data_scale")}
        return self._data  # 返回原始列表

    def save(self, path: str):
        save_json(self._data, path)

    def get_all_records(self) -> List[Dict[str, Any]]:
        return self._data


def plot_training_history(history: Dict, save_path: str, model_name: str = "Model"):
    """绘制训练损失 + 准确率曲线并保存

    Args:
        history: 包含 train_loss, val_loss, train_acc, val_acc 的 dict
        save_path: 图片保存路径
        model_name: 模型名称（用于图标题）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── 左图: Loss ──
    ax1.plot(epochs, history["train_loss"], "o-", color="#3498db", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#e74c3c", label="Val Loss")
    ax1.set_title(f"{model_name} Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 右图: Accuracy ──
    ax2.plot(epochs, history["train_acc"], "o-", color="#2ecc71", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s--", color="#e67e22", label="Val Acc")
    ax2.set_title(f"{model_name} Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练曲线 -> {save_path}")
