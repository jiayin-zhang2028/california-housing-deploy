"""老资产的预处理逻辑：把 Kaggle 里手写的流程固化成可保存的对象。

为什么单独一个模块，而不是写在 `keras_bundle.py` 里？
────────────────────────────────────────────────
因为 `keras_bundle.py` 是**命令行入口**。用 `python -m src.keras_bundle` 运行它时，
它的模块名是 `__main__`，而不是 `src.keras_bundle`。

而 pickle / joblib 保存对象时，会把「类定义在哪个模块」一起写进文件。
如果类定义在 `__main__` 里，bundle 里记的就是：

    __main__.LegacyAssetPreprocessor

换任何一个别的入口去加载（`src.predict`、`src.inference`、uvicorn 启动的
`app.main`），都会报：

    AttributeError: Can't get attribute 'LegacyAssetPreprocessor'
                    on <module 'src.predict' ...>

把类放在一个「永远被 import、从不被当脚本执行」的模块里，就彻底避开了这个坑。
这条经验适用于任何要 pickle 的对象：**类不要定义在入口脚本里。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import CATEGORICAL_FIELD

CATEGORICAL_COLUMNS = (CATEGORICAL_FIELD,)


class LegacyAssetPreprocessor:
    """复刻「fillna(中位数) -> get_dummies -> reindex -> StandardScaler」这套老流程。

    它实现了和 ColumnTransformer 一样的 ``transform(df) -> ndarray`` 接口，
    所以可以直接塞进 bundle，被推理层无差别调用。
    """

    def __init__(self, assets: dict):
        self.scaler = assets["scaler"]
        self.train_columns: list[str] = list(assets["train_columns"])

        median = assets.get("train_median")
        if median is None:
            raise ValueError(
                "assets 里没有 train_median，无法复现训练时的缺失填充。\n"
                "请回到训练脚本，加上：\n"
                "    train_median = X_train.select_dtypes(include='number').median()\n"
                "并和 scaler 一起 joblib.dump。这就是「预处理必须整体保存」的意义。"
            )
        # 兼容 pd.Series（旧格式）和 dict（参数格式）
        self.train_median = pd.Series(median, dtype="float64")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        numeric_cols = [c for c in df.columns if c not in CATEGORICAL_COLUMNS]
        out = df.copy()

        # 0) 先确保数值列真的是数值类型。
        #    否则 pandas 3 在 fillna 时会警告 "Downcasting object dtype"，
        #    而且 object 列进 scaler 会直接报错。
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="raise")

        # 1) 缺失填充：只用训练集中位数，绝不看当前这批数据
        out[numeric_cols] = out[numeric_cols].fillna(self.train_median)

        # 2) 独热编码：reindex 到训练时的列，保证「列顺序 + 列数」完全一致
        out = pd.get_dummies(out, columns=list(CATEGORICAL_COLUMNS))
        out = out.reindex(columns=self.train_columns, fill_value=0)

        # 3) 标准化 + 转 float32（Keras 对 dtype 比较敏感）
        return self.scaler.transform(out.to_numpy(dtype="float64")).astype("float32")

    def __repr__(self) -> str:
        return f"LegacyAssetPreprocessor({len(self.train_columns)} columns)"


def rebuild_scaler(mean, scale):
    """用纯参数重建一个 StandardScaler，不需要和训练时的 sklearn 版本一致。"""
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.mean_ = np.asarray(mean, dtype="float64")
    sc.scale_ = np.asarray(scale, dtype="float64")
    sc.var_ = sc.scale_**2
    sc.n_features_in_ = int(sc.mean_.shape[0])
    return sc


def load_assets(path) -> dict:
    """读 assets，并把「参数格式」归一化成「对象格式」。

    支持两种来源：
      A. 对象格式（Kaggle 直接 dump 的 scaler / y_scaler），可能带环境依赖；
      B. 参数格式（只有 float / list / dict），任何机器都能读。
    """
    import joblib

    try:
        assets = joblib.load(path)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"加载 {path} 失败：本地缺少模块 `{exc.name}`。\n"
            "原因：Kaggle 里的对象是连着 Kaggle 环境一起序列化的，本地没有那个包。\n"
            "最快的修法（二选一）：\n"
            f"  1) python -m pip install {exc.name}\n"
            "  2) 更好：回 Kaggle 用「参数格式」重存一份（只有 float/list/dict）"
        ) from exc

    if not isinstance(assets, dict):
        raise SystemExit(f"{path} 里不是 dict，看起来不是本项目期望的 assets 格式")

    if "scaler" not in assets:
        if "scaler_mean" not in assets or "scaler_scale" not in assets:
            raise SystemExit(
                "assets 里既没有 scaler 对象，也没有 scaler_mean/scaler_scale 参数。\n"
                "请按 src/keras_bundle.py 顶部注释里的 A 或 B 格式重新保存。"
            )
        assets["scaler"] = rebuild_scaler(assets["scaler_mean"], assets["scaler_scale"])
        print("[assets] 用 scaler_mean/scaler_scale 重建了 StandardScaler")

    if "y_scaler" not in assets:
        if "y_mean" not in assets or "y_scale" not in assets:
            raise SystemExit(
                "assets 里没有 y_scaler（也没有 y_mean/y_scale），预测值无法还原成美元。\n"
                "训练时你对 y 做过 StandardScaler，就必须把它的参数一起存下来。"
            )
        assets["y_scaler"] = rebuild_scaler([assets["y_mean"]], [assets["y_scale"]])
        print("[assets] 用 y_mean/y_scale 重建了 y_scaler")

    return assets
