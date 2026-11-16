"""推理层：服务端唯一需要认识的类。

设计要点（面试常问、上线常踩）：
  1) 加载一次，全程复用 —— 模型进内存要几百毫秒到几秒，不能每个请求 load 一遍。
  2) 训练和推理走同一个 ``preprocessor`` 对象 —— 从根上消灭「列对不齐」。
  3) 只暴露 ``predict(records) -> list[float]``，上层（CLI / FastAPI / 批处理）
     完全不关心底下是 sklearn 还是 Keras。换模型不动接口，这叫解耦。
  4) 输出做业务兜底：房价不可能为负，clip 到 0。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from src.preprocess import as_frame


class HousingPredictor:
    """加载 model_bundle.joblib，提供单条 / 批量预测。"""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.kind: str = bundle.get("kind", "sklearn")
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.y_scaler = bundle["y_scaler"]
        self.features: list[str] = list(bundle["features"])
        self.metadata: dict = bundle.get("metadata", {})
        self._tf = None  # Keras 分支才懒加载 tensorflow

    # ---------------- 生命周期 ----------------
    @classmethod
    def load(cls, path: str | Path = "artifacts/model_bundle.joblib") -> "HousingPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到模型文件 {path}。先跑 `python -m src.train --synthetic` "
                f"或 `python -m src.train --data data/housing.csv` 生成。"
            )
        bundle = joblib.load(path)
        if bundle.get("bundle_version") != 1:
            raise ValueError(f"不认识的 bundle 版本: {bundle.get('bundle_version')}")
        return cls(bundle)

    # ---------------- 核心推理 ----------------
    def _raw_predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.kind == "keras":
            if self._tf is None:
                import tensorflow as tf  # 只有真的用 Keras 才 import，省启动时间

                self._tf = tf
            return np.asarray(self.model.predict(matrix, verbose=0)).reshape(-1)
        return np.asarray(self.model.predict(matrix)).reshape(-1)

    def predict(self, records: Iterable[dict[str, Any]]) -> list[float]:
        records = list(records)
        if not records:
            return []

        frame = as_frame(records)
        matrix = self.preprocessor.transform(frame)

        scaled = self._raw_predict(matrix)
        values = self.y_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)

        out: list[float] = []
        for v in values:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                raise ValueError("模型输出了 NaN/Inf，检查预处理或输入数据")
            out.append(max(v, 0.0))  # 房价不能是负数
        return out

    def predict_one(self, record: dict[str, Any]) -> float:
        return self.predict([record])[0]

    # ---------------- 给 /model-info 用的自描述 ----------------
    def info(self) -> dict:
        return {
            "kind": self.kind,
            "features": self.features,
            "target": self.bundle.get("target"),
            "metadata": self.metadata,
        }


def records_from_csv(path: str | Path) -> tuple[list[dict], list[float] | None]:
    """读批量预测用的 CSV；如果带真实房价列，一并返回用于评估。"""
    df = pd.read_csv(path)
    target = "median_house_value"
    truth = df[target].tolist() if target in df.columns else None
    records = df.drop(columns=[c for c in [target] if c in df.columns]).to_dict("records")
    return records, truth
