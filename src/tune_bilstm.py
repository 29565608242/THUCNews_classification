#!/usr/bin/env python3
"""BiLSTM 超参数搜索——系统地尝试多种参数组合，选出最佳配置

搜索维度：
  1. Pooling 策略: max, mean_max, attention
  2. hidden_dim: 192, 256
  3. dropout: 0.3, 0.4, 0.5
  4. lr: 3e-4, 5e-4, 1e-3
  5. weight_decay: 1e-5, 5e-5, 1e-4
  6. embedding_dim: 200, 300
"""

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bilstm import train as train_bilstm
from config import MODEL_DIR, RANDOM_SEED


def run_single_trial(params: dict, trial_name: str, data_scale: int = 20000) -> dict:
    """运行单次训练并返回指标"""
    print(f"\n{'='*60}")
    print(f"TRIAL: {trial_name}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    # 添加 trial_name 到参数，方便识别
    train_params = {**params, "epochs": 15}  # 搜索时用较少 epoch

    try:
        metrics = train_bilstm(data_scale=data_scale, params_override=train_params)
        return {
            "trial": trial_name,
            **params,
            "macro_f1": metrics.get("macro_f1", 0),
            "accuracy": metrics.get("accuracy", 0),
            "macro_precision": metrics.get("macro_precision", 0),
            "macro_recall": metrics.get("macro_recall", 0),
            "best_epoch": metrics.get("best_epoch", 0),
            "train_time_sec": metrics.get("train_time_sec", 0),
            "test_loss": metrics.get("test_loss", 0),
            "history": metrics.get("history", {}),
        }
    except Exception as e:
        print(f"  [FAIL] {e}")
        return {"trial": trial_name, **params, "macro_f1": -1, "error": str(e)}


def main():
    # ── 搜索空间（8 个组合）──
    search_space = [
        {
            "pooling": ["mean_max"],
            "hidden_dim": [192, 256],
            "dropout": [0.3, 0.5],
            "lr": [3e-4, 1e-3],
            "weight_decay": [1e-4],
            "embedding_dim": [300],
        },
    ]

    # 展开所有组合
    all_trials = []
    for space in search_space:
        keys = list(space.keys())
        for values in itertools.product(*space.values()):
            all_trials.append(dict(zip(keys, values)))

    print(f"总共 {len(all_trials)} 个参数组合待搜索")
    print(f"训练数据量: 20000（加速搜索）")
    print(f"每个组合最多 15 epochs，含 early stopping")
    print()

    # 运行所有 trials
    results = []
    for i, params in enumerate(all_trials):
        trial_name = f"trial_{i+1:02d}"
        result = run_single_trial(params, trial_name)
        results.append(result)

        # 实时输出当前最佳
        valid = [r for r in results if r.get("macro_f1", -1) > 0]
        if valid:
            best = max(valid, key=lambda r: r["macro_f1"])
            print(f"\n>>> 当前最佳: {best['trial']} | F1={best['macro_f1']:.4f} | "
                  f"Params: {{pooling={best['pooling']}, hidden={best['hidden_dim']}, "
                  f"dropout={best['dropout']}, lr={best['lr']}, wd={best['weight_decay']}}}")

    # ── 汇总结果 ──
    print(f"\n{'='*60}")
    print("搜索完成！结果汇总：")
    print(f"{'='*60}")

    # 按 F1 排序
    results.sort(key=lambda r: r.get("macro_f1", -1), reverse=True)

    summary = []
    for r in results:
        summary.append({
            "trial": r["trial"],
            "pooling": r["pooling"],
            "hidden_dim": r["hidden_dim"],
            "dropout": r["dropout"],
            "lr": r["lr"],
            "weight_decay": r["weight_decay"],
            "embedding_dim": r.get("embedding_dim", 300),
            "macro_f1": r.get("macro_f1", -1),
            "accuracy": r.get("accuracy", -1),
            "best_epoch": r.get("best_epoch", -1),
            "train_time_sec": r.get("train_time_sec", 0),
        })

    # 打印排行榜
    print(f"\n{'='*60}")
    print("排行榜 (Top 10)：")
    print(f"{'='*60}")
    print(f"{'Rank':>4} {'Trial':<10} {'Pooling':<10} {'Hidden':<8} {'Dropout':<8} "
          f"{'LR':<10} {'WD':<10} {'F1':<8} {'Acc':<8} {'BestEp':<6}")
    print("-" * 90)
    for rank, s in enumerate(summary[:10], 1):
        print(f"{rank:>4} {s['trial']:<10} {s['pooling']:<10} {s['hidden_dim']:<8} "
              f"{s['dropout']:<8} {s['lr']:<10} {s['weight_decay']:<10} "
              f"{s['macro_f1']:<8.4f} {s['accuracy']:<8.4f} {s['best_epoch']:<6}")

    # 保存结果
    save_path = MODEL_DIR / "bilstm_tuning_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {save_path}")

    # 输出最佳参数
    best = summary[0]
    print(f"\n{'='*60}")
    print("推荐最佳参数：")
    print(f"{'='*60}")
    print(f"  BILSTM_POOLING = '{best['pooling']}'")
    print(f"  BILSTM_HIDDEN_DIM = {best['hidden_dim']}")
    print(f"  BILSTM_DROPOUT = {best['dropout']}")
    print(f"  BILSTM_LR = {best['lr']}")
    print(f"  BILSTM_WEIGHT_DECAY = {best['weight_decay']}")
    print(f"  BILSTM_EMBEDDING_DIM = {best['embedding_dim']}")
    print(f"  Macro F1 = {best['macro_f1']:.4f}")
    print(f"  Accuracy = {best['accuracy']:.4f}")

    return best


if __name__ == "__main__":
    main()
