"""契约测试：schema.py（训练侧）与 models.py（服务侧）必须是同一份真相。

这份测试存在的意义：
    你以后加字段时，如果只改了 pydantic 忘了 schema（或反过来），
    这里会立刻红 —— 而不是等上线后用户报「这个字段传了没用」。
"""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from app.models import HousingFeatures
from src.schema import (
    CATEGORICAL_FIELD,
    CATEGORIES,
    FEATURES,
    NUMERIC_FIELDS,
    validate_record,
)

VALID = {
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


def test_features_match_pydantic_fields() -> None:
    assert set(FEATURES) == set(HousingFeatures.model_fields), "训练特征与接口字段不一致"


def test_numeric_bounds_match() -> None:
    """pydantic 的 ge/le 必须等于 schema.py 里写的区间。"""
    for field in NUMERIC_FIELDS:
        meta = HousingFeatures.model_fields[field.name].metadata
        le = next((m.le for m in meta if hasattr(m, "le")), None)
        ge = next((m.ge for m in meta if hasattr(m, "ge")), None)
        assert le == field.hi, f"{field.name} 上界不一致: {le} != {field.hi}"
        if field.nullable:
            assert ge is None, f"{field.name} 允许 null，就不该有下界约束"
        else:
            assert ge == field.lo, f"{field.name} 下界不一致: {ge} != {field.lo}"


def test_category_literal_matches() -> None:
    annotation = HousingFeatures.model_fields[CATEGORICAL_FIELD].annotation
    allowed = set(typing.get_args(annotation))
    assert allowed == set(CATEGORIES)


def test_valid_record_passes_both_layers() -> None:
    assert validate_record(VALID) == []
    assert HousingFeatures(**VALID).total_rooms == 880


@pytest.mark.parametrize(
    "field, bad_value, hint",
    [
        ("latitude", 100, "纬度填到 100"),
        ("median_income", -1, "收入为负"),
        ("total_rooms", 0, "房间数为 0（下界 1）"),
    ],
)
def test_out_of_range_rejected_by_both_layers(field: str, bad_value: float, hint: str) -> None:
    bad = {**VALID, field: bad_value}
    assert validate_record(bad), f"schema 层没拦住：{hint}"
    with pytest.raises(ValidationError):
        HousingFeatures(**bad)


def test_unknown_category_rejected() -> None:
    bad = {**VALID, "ocean_proximity": "MOUNTAIN"}
    assert validate_record(bad)
    with pytest.raises(ValidationError):
        HousingFeatures(**bad)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        HousingFeatures(**VALID, foo=1)


def test_nullable_bedrooms() -> None:
    ok = {**VALID, "total_bedrooms": None}
    assert validate_record(ok) == []
    assert HousingFeatures(**ok).total_bedrooms is None
