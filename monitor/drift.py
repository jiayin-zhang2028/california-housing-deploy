"""线上监控：输入漂移检测（PSI）。

这是「检测部分」的核心一环。你要建立的概念：

    模型上线那天开始，它的准确率就只会下降 —— 因为世界在变，训练数据没变。
    这个过程叫 data drift（数据漂移）/ concept drift（概念漂移）。
    你不能等用户投诉才发现，所以需要**不看标签也能发现异常**的指标。

    为什么用 PSI 而不是准确率？因为线上往往拿不到真实房价（要等成交），
    但输入特征是立刻能拿到的。

用法：
    python -m monitor.drift --log logs/predictions.jsonl
    python -m monitor.drift --csv data/new_month.csv
    python -m monitor.drift --log logs/predictions.jsonl --fail-on-drift   # 给定时任务用
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from monitor.metrics import bin_shares, psi, status
from src.schema import CATEGORICAL_FIELD, NUMERIC_FIELDS

# 注意这里只 import schema，不 import preprocess：
# 监控脚本不该依赖 sklearn，否则「只想看一眼日志」也得装 500MB 的机器学习栈。
NUMERIC_COLUMNS = [f.name for f in NUMERIC_FIELDS]

DEFAULT_STATS = Path("artifacts/reference_stats.json")


def load_records(path: Path) -> pd.DataFrame:
    """支持两种输入：预测日志 jsonl，或一份新的 CSV。"""
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("event") == "predict":
                rows.append(item["input"])
        if not rows:
            raise SystemExit(f"{path} 里没有可用的 predict 记录")
        return pd.DataFrame(rows)

    df = pd.read_csv(path)
    drop = [c for c in ("median_house_value", "target") if c in df.columns]
    return df.drop(columns=drop)


def build_report(df: pd.DataFrame, stats: dict) -> list[dict]:
    rows: list[dict] = []

    for col in NUMERIC_COLUMNS:
        ref = stats["numeric"].get(col)
        if not ref or col not in df.columns:
            continue
        values = df[col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        actual = bin_shares(values, ref["edges"])
        value = psi(ref["proportions"], actual)
        rows.append(
            {
                "feature": col,
                "psi": round(value, 4),
                "status": status(value),
                "ref_median": round(ref["median"], 3),
                "now_median": round(float(pd.Series(values).median()), 3),
            }
        )

    cat_ref = stats.get("categorical", {}).get(CATEGORICAL_FIELD, {}).get("share", {})
    if cat_ref and CATEGORICAL_FIELD in df.columns:
        now_share = df[CATEGORICAL_FIELD].value_counts(normalize=True)
        cats = sorted(set(cat_ref) | set(now_share.index))
        expected = [float(cat_ref.get(c, 0.0)) for c in cats]
        actual = [float(now_share.get(c, 0.0)) for c in cats]
        value = psi(expected, actual)
        rows.append(
            {
                "feature": CATEGORICAL_FIELD,
                "psi": round(value, 4),
                "status": status(value),
                "ref_median": "-",
                "now_median": "-",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="输入漂移检测（PSI）")
    parser.add_argument("--log", default="logs/predictions.jsonl", help="预测日志 jsonl")
    parser.add_argument("--csv", help="或直接给一份新数据 CSV")
    parser.add_argument("--stats", default=str(DEFAULT_STATS), help="训练时产出的分布快照")
    parser.add_argument(
        "--fail-on-drift", action="store_true", help="有特征漂移时以非 0 退出（给定时任务/告警用）"
    )
    args = parser.parse_args()

    stats_path = Path(args.stats)
    if not stats_path.exists():
        raise SystemExit(f"缺少 {stats_path}，请先跑训练脚本生成基准分布")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    source = Path(args.csv) if args.csv else Path(args.log)
    if not source.exists():
        raise SystemExit(f"找不到数据源 {source}（先起服务打几个请求，日志就有了）")
    df = load_records(source)

    rows = build_report(df, stats)
    print(f"\n=== 漂移报告（样本 {len(df)} 条，基准 {stats_path}）===")
    print(f"{'特征':<20}{'PSI':>8}  {'结论':<8}{'训练中位数':>12}{'当前中位数':>12}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['feature']:<20}{r['psi']:>8.4f}  {r['status']:<8}"
            f"{str(r['ref_median']):>12}{str(r['now_median']):>12}"
        )

    drifted = [r["feature"] for r in rows if r["status"] == "DRIFT"]
    if drifted:
        print(f"\n[alarm] 明显漂移: {', '.join(drifted)} -> 该考虑重新训练了")
        if args.fail_on_drift:
            sys.exit(1)
    else:
        print("\n[ok] 没有特征触发 DRIFT 阈值")


if __name__ == "__main__":
    main()
