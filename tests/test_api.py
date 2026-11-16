"""接口测试：直接对 HTTP 层打请求，不关心里面是 sklearn 还是 Keras。

这就是「契约测试」的第二层含义：
    只要 /predict 的行为不变，我可以随便换模型、换框架，测试照样绿。
"""

from __future__ import annotations

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


def test_health_ok(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info(client) -> None:
    body = client.get("/model-info").json()
    assert body["target"] == "median_house_value"
    assert "longitude" in body["features"]


def test_predict_ok(client) -> None:
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_house_value"] > 0
    assert body["currency"] == "USD"
    assert resp.headers["X-Process-Time-Ms"]


def test_predict_rejects_out_of_range(client) -> None:
    resp = client.post("/predict", json={**VALID, "latitude": 100})
    assert resp.status_code == 422
    assert resp.json()["error"]["status"] == 422


def test_predict_rejects_missing_field(client) -> None:
    payload = {k: v for k, v in VALID.items() if k != "median_income"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_unknown_category(client) -> None:
    assert client.post("/predict", json={**VALID, "ocean_proximity": "MOON"}).status_code == 422


def test_predict_batch(client) -> None:
    resp = client.post("/predict_batch", json={"records": [VALID, VALID]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty_rejected(client) -> None:
    assert client.post("/predict_batch", json={"records": []}).status_code == 422


def test_batch_too_large_rejected(client) -> None:
    resp = client.post("/predict_batch", json={"records": [VALID] * 1001})
    assert resp.status_code == 422


def test_openapi_and_docs_available(client) -> None:
    """/docs 是白送的接口文档 —— 交付给前端 / 验收时非常有用。"""
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_404_uses_unified_error_format(client) -> None:
    """框架自己的 404 也要走统一格式，否则前端的错误处理要写两套。"""
    resp = client.get("/no-such-endpoint")
    assert resp.status_code == 404
    assert resp.json()["error"]["status"] == 404
