"""字段与取值范围的「唯一事实来源」（single source of truth）。

为什么要把这份定义单独抽出来？
    训练时：用它清洗数据；
    服务时：用它生成接口校验（app/models.py）；
    监控时：用它判断输入是否漂移（monitor/drift.py）。
三处共用一份定义，才不会出现「训练认 9 列、接口只收 8 列」这种上线后
排查一整天的经典事故。

本文件不依赖 sklearn / tensorflow / fastapi，谁都能 import。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TARGET = "median_house_value"
CATEGORICAL_FIELD = "ocean_proximity"

# Kaggle 的 california housing（camnugent 版）里 ocean_proximity 只有这 5 种取值。
CATEGORIES: tuple[str, ...] = (
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
)


@dataclass(frozen=True)
class NumericField:
    """一个数值字段：名字 + 合法区间 + 是否允许缺失。"""

    name: str
    lo: float
    hi: float
    nullable: bool = False
    note: str = ""


# 区间的依据：Kaggle 原始数据的 min / max（留了一点余量）。
# 注意：这不是「越多越好」的校验，而是拦住明显异常值 ——
# 比如前端把纬度填到 100、把 total_rooms 填成 -5。
NUMERIC_FIELDS: tuple[NumericField, ...] = (
    NumericField("longitude", -124.35, -114.31, note="经度，加州约 -124 ~ -114"),
    NumericField("latitude", 32.54, 41.95, note="纬度，加州约 32 ~ 42"),
    NumericField("housing_median_age", 1, 52, note="房龄中位数，数据里封顶 52"),
    NumericField("total_rooms", 1, 39320, note="街区总房间数"),
    NumericField("total_bedrooms", 1, 6445, nullable=True, note="缺失约 200 行"),
    NumericField("population", 1, 35682, note="街区人口"),
    NumericField("households", 1, 6082, note="街区户数"),
    NumericField("median_income", 0.4999, 15.0001, note="单位是万美元，已除以 1e4"),
)

FEATURES: tuple[str, ...] = tuple(f.name for f in NUMERIC_FIELDS) + (CATEGORICAL_FIELD,)

# 训练集里 median_house_value 被「封顶」在 500001 美元，这是数据集的已知缺陷。
TARGET_CAP = 500_001.0


def validate_record(record: dict[str, Any]) -> list[str]:
    """校验单条样本，返回错误信息列表（空列表 = 合法）。

    这里故意只做「便宜且明确」的检查：
      1. 字段是否齐全（少的补、多的忽略太宽容，直接报错更好排查）；
      2. 类型是否对得上；
      3. 数值是否在合法区间；
      4. 类别是否在已知集合内。
    真正的模型效果问题（比如经纬度落在海里）不在这里管，靠监控兜底。
    """
    errors: list[str] = []

    missing = [f.name for f in NUMERIC_FIELDS if f.name not in record]
    if CATEGORICAL_FIELD not in record:
        missing.append(CATEGORICAL_FIELD)
    if missing:
        errors.append(f"缺少字段: {', '.join(missing)}")
        return errors  # 缺字段就没必要继续往下比了

    if record[CATEGORICAL_FIELD] not in CATEGORIES:
        errors.append(
            f"{CATEGORICAL_FIELD} 取值 {record[CATEGORICAL_FIELD]!r} 未知，"
            f"必须是 {list(CATEGORIES)} 之一"
        )

    for field in NUMERIC_FIELDS:
        value = record[field.name]
        if value is None:
            if field.nullable:
                continue
            errors.append(f"{field.name} 不允许为空")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{field.name} 必须是数字，收到 {type(value).__name__}")
            continue
        if not (field.lo <= value <= field.hi):
            errors.append(f"{field.name}={value} 超出合理范围 [{field.lo}, {field.hi}]")

    return errors


def validate_dataframe(df: "Any") -> list[str]:
    """训练前的批量体检：列齐不齐、类别是否越界、数值区间是否离谱。

    只“报告”不“修改”，让训练脚本自己决定警告还是中断 ——
    静默改写数据是数据泄漏 / 静默翻车的高发区。
    """
    problems: list[str] = []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        problems.append(f"缺少特征列: {missing}")
        return problems
    if TARGET not in df.columns:
        problems.append(f"缺少目标列: {TARGET}")

    unseen = sorted(set(df[CATEGORICAL_FIELD].dropna().unique()) - set(CATEGORIES))
    if unseen:
        problems.append(f"出现未知类别: {unseen}")

    for field in NUMERIC_FIELDS:
        col = df[field.name]
        bad = ((col < field.lo) | (col > field.hi)).sum()
        if bad:
            problems.append(f"{field.name} 有 {bad} 行超出 [{field.lo}, {field.hi}]")
    return problems
