"""训练脚本：把「notebook 里的实验」变成「一条命令能复现的流水线」。

用法：
    # 用 Kaggle 下载的 housing.csv
    python -m src.train --data data/housing.csv

    # 没有数据 / CI 里用：自动造一份合成数据，跑通整条链路
    python -m src.train --synthetic

产物（都写进 artifacts/）：
    model_bundle.joblib   # 模型 + 预处理器 + y 缩放器 + 元数据（部署只认这个文件）
    metrics.json          # 评估结果，人和 CI 都能读
    reference_stats.json  # 训练集分布快照，给线上漂移检测当基准

为什么要有 --synthetic？因为「链路能不能跑通」和「模型准不准」是两件事。
先保证链路通，再谈精度。CI 里每天用合成数据跑一遍，链路永远不会烂。
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocess import build_preprocessor, numeric_columns
from src.schema import (
    CATEGORICAL_FIELD,
    CATEGORIES,
    FEATURES,
    NUMERIC_FIELDS,
    TARGET,
    validate_dataframe,
)

ARTIFACT_DIR = Path("artifacts")


# --------------------------------------------------------------------------
# 1. 取数
# --------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件 {path}。请从 Kaggle 下载 housing.csv 放进 data/，"
            f"或改用 --synthetic 先跑通链路。"
        )
    df = pd.read_csv(path)
    print(f"[data] 读取 {path}，形状 {df.shape}")
    return df


def make_synthetic(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    """造一份「结构像真数据」的合成集，只为让链路能跑，不要用它比精度。"""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(32.54, 41.95, n_rows)
    lon = rng.uniform(-124.35, -114.31, n_rows)
    ocean = rng.choice(list(CATEGORIES), size=n_rows, p=[0.44, 0.48, 0.002, 0.04, 0.038])
    income = np.clip(rng.lognormal(mean=2.0, sigma=0.4, size=n_rows), 0.5, 15.0)
    age = rng.integers(1, 53, n_rows)
    rooms = rng.integers(2, 4000, n_rows)
    bedrooms = np.round(rooms * rng.uniform(0.15, 0.3, n_rows))
    households = np.maximum(1, np.round(rooms / rng.uniform(3.0, 8.0, n_rows)))
    population = np.maximum(3, np.round(households * rng.uniform(1.5, 4.0, n_rows)))

    # 人为埋一点「业务规律」，否则模型学不到东西、指标看着像假的。
    # 系数调过：让房价中位数落在 20 万美元量级，和真实加州数据接近。
    price = (
        20_000
        + 16_000 * income
        + 700 * age
        + 25 * rooms
        - 800 * np.abs(lat - 34.0)
        + 8_000 * (ocean == "NEAR OCEAN")
        - 25_000 * (ocean == "INLAND")
        + rng.normal(0, 30_000, n_rows)
    )
    price = np.clip(price, 15_000, 500_001)

    df = pd.DataFrame(
        {
            "longitude": lon,
            "latitude": lat,
            "housing_median_age": age,
            "total_rooms": rooms,
            "total_bedrooms": bedrooms,
            "population": population,
            "households": households,
            "median_income": np.round(income, 4),
            "ocean_proximity": ocean,
            TARGET: np.round(price, 0),
        }
    )
    # 故意挖掉 1% 的 total_bedrooms，模拟真实数据的缺失，验证 imputer 真的在工作。
    holes = rng.choice(n_rows, size=max(1, n_rows // 100), replace=False)
    df.loc[holes, "total_bedrooms"] = np.nan
    print(f"[data] 生成合成数据 {df.shape}（仅用于跑通链路）")
    return df


# --------------------------------------------------------------------------
# 2. 洗数（有痕迹地洗，不要静默改）
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 先硬性检查列是否齐全：缺列时立刻报错退出，
    # 比跑到一半 KeyError 好得多（这就是「fail loud, fail early」）。
    missing = [c for c in (*FEATURES, TARGET) if c not in df.columns]
    if missing:
        raise SystemExit(f"数据缺少列 {missing}\n当前列: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()               # 没标签的样本不能训练
    dropped_target = before - len(df)

    problems = validate_dataframe(df)
    for p in problems:
        print(f"[warn] {p}")

    # 超范围的行直接丢，因为它们的量级明显是脏数据（比如 total_rooms=39320 之外的极端值）
    for field in NUMERIC_FIELDS:
        col = df[field.name]
        if field.nullable:
            mask = col.notna() & ((col < field.lo) | (col > field.hi))
        else:
            mask = (col < field.lo) | (col > field.hi)
        if mask.any():
            print(f"[clean] {field.name}: 丢弃 {int(mask.sum())} 行超范围数据")
            df = df[~mask]

    # ocean_proximity 的异常值也丢掉
    df = df[df[CATEGORICAL_FIELD].isin(CATEGORIES)]

    capped = int((df[TARGET] >= 500_000).sum())
    if capped:
        print(
            f"[note] 有 {capped} 行目标值 >= 50 万美元（数据集的封顶缺陷）。"
            f"这里保留，但你要知道：模型在高端房价上天生偏低估。"
        )

    print(f"[clean] 行数 {before} -> {len(df)}（目标缺失丢掉 {dropped_target} 行）")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. 训练
# --------------------------------------------------------------------------
def train(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()

    df = make_synthetic(seed=args.seed) if args.synthetic else load_csv(Path(args.data))
    df = clean(df)

    X = df[list(FEATURES)]
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)      # 只在训练集上 fit！
    X_test_t = preprocessor.transform(X_test)

    # y 也缩放：Keras 时代必须做，否则 MSE 梯度尺度太大。
    # 用 sklearn 模型其实不必要，但这里保留，是为了让你理解
    # 「预测完必须反缩放」这个动作 —— 漏掉它就是预测出 0.3 美元的房子。
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        learning_rate=0.08,
        random_state=args.seed,
    )
    model.fit(X_train_t, y_train_s)

    # 统一走「缩放空间 -> 原始空间」，线上代码和评估代码是同一条路径。
    pred = y_scaler.inverse_transform(model.predict(X_test_t).reshape(-1, 1)).ravel()
    pred = np.clip(pred, 0, None)

    metrics = {
        "rmse": float(root_mean_rmse(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        # 基线：永远预测训练集中位数。你的模型如果打不过它，等于没学。
        "baseline_rmse": float(root_mean_rmse(y_test, np.full_like(y_test, np.median(y_train)))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }

    bundle = {
        "bundle_version": 1,
        "kind": "sklearn",
        "model": model,
        "preprocessor": preprocessor,
        "y_scaler": y_scaler,
        "features": list(FEATURES),
        "categorical_field": CATEGORICAL_FIELD,
        "categories": list(CATEGORIES),
        "target": TARGET,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "seed": args.seed,
            "data_source": "synthetic" if args.synthetic else str(args.data),
            "metrics": metrics,
        },
        # 漂移检测的基准：训练集上的分位点和类别分布
        "reference_stats": build_reference_stats(df),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_DIR / "model_bundle.joblib", compress=3)
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ARTIFACT_DIR / "reference_stats.json").write_text(
        json.dumps(bundle["reference_stats"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== 评估（测试集，单位：美元）===")
    print(f"  RMSE           : {metrics['rmse']:,.0f}")
    print(f"  MAE            : {metrics['mae']:,.0f}")
    print(f"  R2             : {metrics['r2']:.4f}")
    print(f"  基线 RMSE(中位数): {metrics['baseline_rmse']:,.0f}")
    print(f"  训练耗时        : {metrics['train_seconds']}s")
    print(f"\n[artifact] {ARTIFACT_DIR / 'model_bundle.joblib'}")
    return metrics


def root_mean_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def build_reference_stats(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """存下训练集的分箱边界与占比，线上用同样的边界算 PSI 做漂移对比。"""
    stats: dict = {"n_bins": n_bins, "numeric": {}, "categorical": {}}
    for col in numeric_columns():
        values = df[col].dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        counts, _ = np.histogram(values, bins=edges)
        stats["numeric"][col] = {
            "edges": [float(e) for e in edges],
            # 每个箱子里训练样本的占比，PSI 的「期望分布」就是它
            "proportions": [float(c / counts.sum()) for c in counts],
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
        }
    stats["categorical"][CATEGORICAL_FIELD] = {
        "share": df[CATEGORICAL_FIELD].value_counts(normalize=True).round(6).to_dict()
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="加州房价模型训练流水线")
    parser.add_argument("--data", default="data/housing.csv", help="Kaggle housing.csv 路径")
    parser.add_argument("--synthetic", action="store_true", help="用合成数据跑通链路")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
