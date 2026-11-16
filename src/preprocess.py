"""特征预处理流水线。

核心思想（这是整个项目最关键的一课）：
    「预处理」不是几个零散的 transform，而是一个**一起 fit、一起保存**的对象。
    只要训练和推理用的是同一个 ColumnTransformer 实例，就不可能对不齐列。

对比一下你现在的写法：
    df.fillna(median) -> pd.get_dummies -> scaler.transform
    每一步都是裸代码，靠你手写顺序；一旦服务端漏写一步，预测就是错的，
    而且不报错 —— 这叫「静默失败」，生产环境最怕这个。

ColumnTransformer 版本：
    - 训练：preprocessor.fit_transform(X_train)
    - 推理：preprocessor.transform(X_new)   # 同一份，列顺序自动一致
    - 保存：joblib.dump(preprocessor)       # 词表/均值/中位数全在里面
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import CATEGORICAL_FIELD, FEATURES, NUMERIC_FIELDS

NUMERIC_COLUMNS = [f.name for f in NUMERIC_FIELDS]


def build_preprocessor() -> ColumnTransformer:
    """构建「缺失填充 + 标准化 + 独热编码」的可保存对象。"""
    numeric_pipe = Pipeline(
        steps=[
            # 中位数只在训练集上算出来，之后固定不变。
            # 千万不能对全量数据算 —— 那就是数据泄漏（leakage）。
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            # handle_unknown="ignore"：线上遇到训练时没见过的类别，
            # 输出全 0 并继续跑，而不是直接 500。这是「服务要抗造」的体现。
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLUMNS),
            ("cat", categorical_pipe, [CATEGORICAL_FIELD]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def numeric_columns() -> list[str]:
    return list(NUMERIC_COLUMNS)


def as_frame(records: list[dict]) -> pd.DataFrame:
    """把「一条条 JSON」变成固定列顺序、固定 dtype 的 DataFrame。

    这一步看起来无聊，其实是防翻车重点：
      1. 列顺序：pandas 的列顺序会影响 ColumnTransformer，
         而 dict 的 key 顺序是调用方随便给的；
      2. dtype：整列都是 None 时 pandas 会给 object 列，SimpleImputer 会直接报错。
    所以这里显式 to_numeric，让「缺值」统一表现为 NaN（而不是 None/object）。
    """
    frame = pd.DataFrame(records, columns=list(FEATURES))
    for col in NUMERIC_COLUMNS:
        # errors="raise"：真的传了个 "abc" 就当场报错（-> 422），
        # 而不是悄悄转成 NaN、被中位数填掉、然后给出一个错得很自然的预测。
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
    return frame
