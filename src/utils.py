"""工具函数——计时、日志、指标收集"""

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
