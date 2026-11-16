# 加州房价预测 · 从 Kaggle 到上线的完整参考实现

一个**能跑通全链路**的最小项目：训练 → 保存资产 → 打包 → HTTP 接口 → 测试 → 容器 → 监控。
专门用来补「我模型训练完了，然后呢？」这一段。

> 配套教材：[../加州房价项目-从Kaggle到落地-学习手册.md](../加州房价项目-从Kaggle到落地-学习手册.md)

## 30 秒跑通

```bash
cd housing-deploy
python -m venv .venv && .venv\Scripts\activate     # Windows
# source .venv/bin/activate                         # macOS / Linux
pip install -r requirements.txt

python -m src.train --synthetic     # 1) 训练（不需要 Kaggle 数据，先保证链路通）
python -m src.predict               # 2) 本地预测一条
python -m pytest                    # 3) 跑测试
python -m uvicorn app.main:app --port 8000   # 4) 起服务
```

浏览器打开 <http://127.0.0.1:8000/docs>，点 **POST /predict → Try it out → Execute**。

换成你自己的数据：

```bash
# 从 Kaggle 下载 "California Housing Prices" 的 housing.csv
mkdir data && copy housing.csv data\housing.csv
python -m src.train --data data/housing.csv
```

你已有的 Keras 模型也能直接接进来：

```bash
python -m src.keras_bundle --model california_housing_model.keras --assets preprocess_assets.joblib
python -m src.predict          # 走同一个推理层，服务代码一行不用改
```

## 文件地图

| 路径 | 干什么 | 对应教材章节 |
|---|---|---|
| `src/schema.py` | 字段与取值范围的唯一事实来源 | 第 2 讲 |
| `src/preprocess.py` | 可保存的预处理流水线（ColumnTransformer） | 第 3 讲 |
| `src/train.py` | 训练 + 评估 + 保存 bundle 与基准分布 | 第 3 讲 |
| `src/inference.py` | 推理层，换模型不动接口 | 第 4 讲 |
| `src/predict.py` | 本地预测 CLI（排障第一工具） | 第 4 讲 |
| `src/keras_bundle.py` | 把老的 .keras 资产包装成统一 bundle | 第 3 讲 |
| `app/models.py` | 接口输入契约（pydantic） | 第 5 讲 |
| `app/main.py` | FastAPI 服务：校验、日志、健康检查 | 第 5 讲 |
| `tests/` | 契约测试 / 推理测试 / 接口测试 | 第 6 讲 |
| `Dockerfile` | 镜像构建、非 root、healthcheck | 第 7 讲 |
| `scripts/smoke_test.py` | 部署后的冒烟验证 | 第 7 讲 |
| `monitor/metrics.py` | PSI 等漂移指标（纯函数，可单测） | 第 8 讲 |
| `monitor/drift.py` | 输入漂移检测报告 | 第 8 讲 |
| `.github/workflows/ci.yml` | 每次 push 自动验证链路 | 第 10 讲 |

## 接口

```bash
# 单条
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41,
       "total_rooms":880,"total_bedrooms":129,"population":322,
       "households":126,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}'
# -> {"predicted_house_value": 452473.19, "currency": "USD", "model_version": "2025-..."}

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

错误一律长这样，前端只要认一种格式：

```json
{"error": {"status": 422, "detail": [{"loc": ["body", "latitude"], "msg": "..."}], "path": "/predict"}}
```

## 容器

```bash
python -m src.train --data data/housing.csv   # 先在宿主机产出 artifacts/
docker build -t housing-pred:latest .
docker run -p 8000:8000 housing-pred:latest
python -m scripts.smoke_test --base-url http://127.0.0.1:8000
```

## 上线前自查（每一条都对应一个真实事故）

- [ ] 模型 + 预处理 + y 缩放器 **一起**保存，且版本号一致
- [ ] 训练/推理走同一个 `preprocessor` 对象，没有两套代码
- [ ] 非法输入返回 422，不是 500，更不是「照样给个数」
- [ ] 模型加载失败时 `/health` 返回 degraded，而不是假装正常
- [ ] 每次预测都留日志：输入、输出、延迟、模型版本
- [ ] 有 `/predict` 的回归测试（固定样本 → 固定量级）
- [ ] 镜像非 root，有 HEALTHCHECK，依赖版本已锁
- [ ] 知道怎么回滚到上一个镜像 tag

## 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: model_bundle.joblib` | 没训练 | 先跑 `python -m src.train --synthetic` |
| 预测值大得离谱（如 0.03） | 忘了 `y_scaler.inverse_transform` | 见 `src/inference.py` 的 `predict` |
| `ValueError: shapes not aligned` | 列顺序 / 列数不一致 | 用 `preprocessor.transform`，别手写 get_dummies |
| `/health` 一直 degraded | 模型路径没挂进容器 | 检查 `MODEL_PATH` 与 volume |
| 端口 8000 被占用 | 上次的服务没停 | `netstat -ano | findstr :8000` 后 kill |
