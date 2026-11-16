"""冒烟测试：对「真正跑起来的服务」发请求。

和 pytest 里的 TestClient 有什么区别？
    TestClient 在同一个进程里调函数，跑得快，但绕过网络/端口/序列化；
    冒烟测试打的是 http://127.0.0.1:8000，走完整链路。

上线检查清单里最后一步永远是它：部署完 -> 打一发 -> 看到数 -> 才算发布成功。

用法：
    python -m scripts.smoke_test --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

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


def call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    checks: list[tuple[str, bool, str]] = []

    status, health = call(f"{base}/health")
    checks.append(("服务存活", status == 200, f"HTTP {status}"))
    checks.append(("模型已加载", health.get("model_loaded") is True, str(health)))

    status, info = call(f"{base}/model-info")
    checks.append(("模型信息可读", status == 200 and info.get("target") == "median_house_value", ""))

    status, pred = call(f"{base}/predict", SAMPLE)
    price = pred.get("predicted_house_value", -1)
    checks.append(("单条预测有结果", status == 200 and 0 < price < 1e8, f"${price:,.0f}"))

    status, bad = call(f"{base}/predict", {**SAMPLE, "latitude": 200})
    checks.append(("非法输入被拒绝", status == 422, f"HTTP {status}"))

    print(f"冒烟测试 -> {base}")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")
        ok = ok and passed
    if not ok:
        sys.exit(1)
    print("全部通过：服务可用。")


if __name__ == "__main__":
    main()
