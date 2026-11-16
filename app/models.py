"""接口层的输入契约（pydantic）。

为什么不能用「裸 dict」收参数？
    FastAPI 用 pydantic 模型声明请求体，就自动获得：
      1) 类型转换与校验 -> 你少写一堆 if；
      2) 错误信息自动定位到字段 -> 前端能看懂；
      3) /docs 里自动生成请求示例 -> 你等于免费得到一份接口文档；
      4) 生成的 OpenAPI 可以被前端 / 测试直接拿来用。

关键点：这里的取值范围**来自 src/schema.py**，不重复写死。
`tests/test_contract.py` 会检查两边是否一致 —— 这叫契约测试。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schema import CATEGORIES, FEATURES, NUMERIC_FIELDS, validate_record

# 用 schema.py 里的区间动态生成 Field 约束，避免「两处定义各写一遍」
_BOUNDS = {f.name: (f.lo, f.hi) for f in NUMERIC_FIELDS}
_NULLABLE = {f.name for f in NUMERIC_FIELDS if f.nullable}


def _field(name: str) -> "Field":
    lo, hi = _BOUNDS[name]
    nullable = name in _NULLABLE
    return Field(
        ge=None if nullable else lo,
        le=hi,
        description=f"取值范围 [{lo}, {hi}]" + ("，允许 null" if nullable else ""),
        examples=[(lo + hi) / 2],
    )


class HousingFeatures(BaseModel):
    """一条待预测的房屋街区特征。

    字段说明全部经过 schema.py 统一，新增字段时只需要改一处。
    """

    model_config = ConfigDict(extra="forbid")  # 多传字段直接报错，别悄悄忽略

    longitude: Annotated[float, _field("longitude")]
    latitude: Annotated[float, _field("latitude")]
    housing_median_age: Annotated[float, _field("housing_median_age")]
    total_rooms: Annotated[float, _field("total_rooms")]
    total_bedrooms: Annotated[float | None, _field("total_bedrooms")] = None
    population: Annotated[float, _field("population")]
    households: Annotated[float, _field("households")]
    median_income: Annotated[float, _field("median_income")]
    ocean_proximity: Literal[CATEGORIES]  # type: ignore[valid-type]

    def to_record(self) -> dict:
        return self.model_dump()


class PredictionResponse(BaseModel):
    predicted_house_value: float = Field(description="预测房价，单位：美元")
    currency: str = "USD"
    model_version: str = Field(description="模型训练时间戳，用于排查「哪版模型出的这个数」")


class BatchRequest(BaseModel):
    records: list[HousingFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[float]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    kind: str
    features: list[str]
    target: str | None
    metadata: dict


__all__ = [
    "FEATURES",
    "validate_record",
    "HousingFeatures",
    "PredictionResponse",
    "BatchRequest",
    "BatchResponse",
    "HealthResponse",
    "ModelInfoResponse",
]
