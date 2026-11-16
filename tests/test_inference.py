"""推理层测试：验证「预处理 -> 模型 -> 反缩放」这条链路本身是对的。

这类测试比「模型精度」更重要：
    精度会随数据波动，链路一断就是 100% 错。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import HousingPredictor

SAMPLE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def test_predict_returns_plausible_price(predictor: HousingPredictor) -> None:
    price = predictor.predict_one(SAMPLE)
    assert isinstance(price, float)
    # 不是精确断言，而是「业务上不可能」的边界：加州房价不会到 1 亿美元
    assert 1_000 < price < 100_000_000


def test_prediction_is_deterministic(predictor: HousingPredictor) -> None:
    a = predictor.predict_one(SAMPLE)
    b = predictor.predict_one(SAMPLE)
    assert a == b, "同一输入两次预测不一致，说明有随机性没固定"


def test_batch_matches_single(predictor: HousingPredictor) -> None:
    batch = predictor.predict([SAMPLE, SAMPLE, SAMPLE])
    assert len(batch) == 3
    assert batch[0] == pytest.approx(predictor.predict_one(SAMPLE))


def test_unseen_category_does_not_crash(predictor: HousingPredictor) -> None:
    """线上一定会遇到训练时没见过的类别。要求：不崩、给个数、别静默变 NaN。"""
    odd = {**SAMPLE, "ocean_proximity": "MOUNTAIN"}
    price = predictor.predict_one(odd)
    assert np.isfinite(price)


def test_missing_bedrooms_handled(predictor: HousingPredictor) -> None:
    """Kaggle 数据本身就缺 total_bedrooms，线上也会缺 —— 必须能用中位数补。"""
    price = predictor.predict_one({**SAMPLE, "total_bedrooms": None})
    assert np.isfinite(price)


def test_negative_prediction_clipped(predictor: HousingPredictor) -> None:
    """极端输入可能让模型吐出负数；业务上房价不能为负，推理层负责兜底。"""
    extreme = {**SAMPLE, "median_income": 0.5, "housing_median_age": 1, "total_rooms": 1}
    assert predictor.predict_one(extreme) >= 0


def test_empty_batch(predictor: HousingPredictor) -> None:
    assert predictor.predict([]) == []


def test_info_contains_metadata(predictor: HousingPredictor) -> None:
    info = predictor.info()
    assert info["kind"] in {"sklearn", "keras"}
    assert info["target"] == "median_house_value"
    assert "trained_at" in info["metadata"]
