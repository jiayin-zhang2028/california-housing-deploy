"""把「你已有的 Keras 资产」包装成统一格式的 bundle。

用法：
    python -m src.keras_bundle
    # 或显式指定
    python -m src.keras_bundle --model xxx.keras --assets xxx.joblib

支持两种 assets 格式（脚本自动识别）：

    【A. 对象格式】直接存 sklearn 对象
        joblib.dump({"scaler": scaler, "y_scaler": y_scaler,
                     "train_columns": [...], "train_median": train_median}, path)

    【B. 参数格式】只存数字，不存对象（强烈推荐）
        joblib.dump({"scaler_mean": scaler.mean_.tolist(),
                     "scaler_scale": scaler.scale_.tolist(),
                     "y_mean": float(y_scaler.mean_[0]),
                     "y_scale": float(y_scaler.scale_[0]),
                     "train_columns": [...],
                     "train_median": {k: float(v) for k, v in train_median.items()}}, path)

为什么推荐 B：
    1) 对象是连着「训练那台机器」一起打包的，换机器可能缺依赖（比如 Kaggle 的 dill）；
    2) 跨 sklearn 版本反序列化会警告，甚至行为改变；
    3) B 只有 float / list / dict，十年后也能读。

注意：真正的预处理类定义在 `src/legacy_assets.py`，**不在这个文件里** ——
因为这个文件是命令行入口，用 `-m` 运行时模块名是 `__main__`，
类定义在这里会被 pickle 记成 `__main__.LegacyAssetPreprocessor`，
换个入口加载就报 AttributeError。详见 legacy_assets.py 顶部说明。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib

# 这两个名字从这里 re-export，方便外部（和测试）继续 `from src.keras_bundle import ...`
from src.legacy_assets import LegacyAssetPreprocessor, load_assets, rebuild_scaler
from src.schema import CATEGORICAL_FIELD, FEATURES, TARGET

__all__ = ["LegacyAssetPreprocessor", "build_bundle", "load_assets", "rebuild_scaler"]


def build_bundle(model_path: str | Path, assets_path: str | Path, out: str | Path) -> Path:
    import tensorflow as tf  # 延迟导入，让没装 TF 的人也能 import 本模块

    model = tf.keras.models.load_model(model_path)
    assets = load_assets(assets_path)
    preprocessor = LegacyAssetPreprocessor(assets)

    bundle = {
        "bundle_version": 1,
        "kind": "keras",
        "model": model,
        "preprocessor": preprocessor,
        "y_scaler": assets["y_scaler"],
        "features": list(FEATURES),
        "categorical_field": CATEGORICAL_FIELD,
        "target": TARGET,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "legacy-keras-assets",
            "model_file": str(model_path),
            "assets_file": str(assets_path),
            "n_input_columns": len(preprocessor.train_columns),
        },
        "reference_stats": {},
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f"[bundle] 已写出 {out}（输入列数 {len(preprocessor.train_columns)}）")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="把老的 Keras 资产打包成统一 bundle")
    parser.add_argument("--model", default="california_housing_model.keras")
    parser.add_argument("--assets", default="preprocess_assets.joblib")
    parser.add_argument("--out", default="artifacts/model_bundle.joblib")
    args = parser.parse_args()
    build_bundle(args.model, args.assets, args.out)


if __name__ == "__main__":
    main()
