"""纯函数：漂移指标。不依赖 sklearn / fastapi，方便单独测。

把它和 ``drift.py`` 分开，是为了让「指标算得对不对」这件事
可以脱离数据加载、日志解析单独验证 —— 这就是**可测试性**的设计。
"""

from __future__ import annotations

import math

import numpy as np


def psi(expected: list[float], actual: list[float], eps: float = 1e-6) -> float:
    """Population Stability Index。

    PSI = Σ (actual - expected) * ln(actual / expected)

    * expected：训练集里每个箱子的样本占比
    * actual  ：当前数据用**同一组边界**分箱后的占比
    * eps       ：防止 log(0)；0.0001 量级的占比不会被它明显扭曲

    性质（也是测试要断言的）：
        * 分布完全一致 -> 0
        * 分布完全错位 -> 很大（> 1）
        * 永远 >= 0
    """
    if len(expected) != len(actual):
        raise ValueError(f"分箱数不一致: expected {len(expected)} vs actual {len(actual)}")
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(float(e), eps)
        a = max(float(a), eps)
        total += (a - e) * math.log(a / e)
    return total


def bin_shares(values: np.ndarray, edges: list[float] | np.ndarray) -> list[float]:
    """按给定边界分箱，返回每箱占比。

    注意：边界必须来自训练集（存在 reference_stats.json 里），
    绝不能用当前数据重新分位数 —— 那样每次算出来的分布都一样，漂移永远测不出来。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0] * (len(edges) - 1)
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    return [float(c / total) for c in counts]


def status(value: float) -> str:
    """PSI 的经验阈值。"""
    if value < 0.1:
        return "STABLE"
    if value < 0.25:
        return "WATCH"
    return "DRIFT"
