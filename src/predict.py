"""单条 / 批量预测的命令行入口（对应你截图里的「方案 1」）。

为什么有了 API 还要留这个脚本？
    因为排查问题时，你希望「不启服务、不连网络、不依赖前端」就能复现一个预测值。
    线上出问题时，第一步永远是：拿这条请求的数据，在本地脚本里跑一遍。
    脚本能复现 -> 问题在服务层（校验/序列化/网络）；
    脚本也错   -> 问题在模型/预处理。

用法：
    python -m src.predict                 # 用内置样例跑一条
    python -m src.predict --json '{"longitude":-122.23,...}'
    python -m src.predict --csv data/housing.csv --limit 5
    python -m src.predict --bundle artifacts/other.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.inference import HousingPredictor, records_from_csv
from src.schema import validate_record

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


def main() -> None:
    parser = argparse.ArgumentParser(description="加州房价单条/批量预测")
    parser.add_argument("--bundle", default="artifacts/model_bundle.joblib")
    parser.add_argument("--json", help="一条 JSON 样本")
    parser.add_argument("--csv", help="批量 CSV")
    parser.add_argument("--limit", type=int, default=5, help="CSV 只打印前 N 条")
    args = parser.parse_args()

    predictor = HousingPredictor.load(args.bundle)
    print(f"[model] kind={predictor.kind} trained_at={predictor.metadata.get('trained_at')}")

    if args.csv:
        records, truth = records_from_csv(args.csv)
        preds = predictor.predict(records)
        print(f"[batch] 共 {len(records)} 条")
        for i in range(min(args.limit, len(preds))):
            line = f"  #{i:<3} 预测 ${preds[i]:>12,.0f}"
            if truth is not None and not np.isnan(truth[i]):
                line += f" | 真实 ${truth[i]:>12,.0f} | 误差 ${abs(preds[i]-truth[i]):,.0f}"
            print(line)
        if truth is not None:
            rmse = np.sqrt(np.mean((np.asarray(preds) - np.asarray(truth, dtype=float)) ** 2))
            print(f"[batch] RMSE = ${rmse:,.0f}")
        return

    record = json.loads(args.json) if args.json else dict(SAMPLE)
    errors = validate_record(record)
    if errors:
        # 注意：脚本里「报错退出」是对的；服务里则要返回 422 并说清哪个字段错了。
        raise SystemExit("[invalid] " + "; ".join(errors))

    price = predictor.predict_one(record)
    print(json.dumps({"predicted_house_value": round(price, 2)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
