"""FastAPI 服务：把模型变成一个别人能调用的 HTTP 接口。

这份代码里包含了「能上线」和「只能演示」之间的全部差别，逐条对照：

  1. 启动时加载模型（lifespan），不是每个请求加载一次        -> 延迟从 2s 降到 5ms
  2. 输入有契约（pydantic），越界/缺字段返回 422             -> 不会拿脏数据喂模型
  3. 模型坏了返回 503 而不是 200 加一个瞎猜的数               -> 上游能感知故障
  4. 每个请求都记一条结构化日志（含模型版本、延迟、预测值）     -> 上线后能复盘
  5. /health 与 /model-info 分离：探活不该触发推理            -> K8s / 负载均衡要这个
  6. 环境变量配置模型路径                                     -> 同一个镜像能跑不同模型

跑起来：
    uvicorn app.main:app --reload --port 8000
    浏览器打开 http://127.0.0.1:8000/docs 直接点着测。
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    HousingFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from src.inference import HousingPredictor

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_bundle.joblib")
LOG_PATH = Path(os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))

# 全局只放一个预测器；用 lifespan 控制它的生命周期，测试里可以直接替换。
state: dict = {"predictor": None, "load_error": None}

_log_lock = threading.Lock()


def _append_log(payload: dict) -> None:
    """把预测记录追加到 jsonl。

    为什么不用 print？因为日志是要被机器消费的：
    以后算「输入漂移」、看「线上预测分布有没有变」全靠这个文件。
    一条日志至少要有：时间、请求体、预测值、延迟、模型版本。
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:  # 日志写不进去绝不能让预测失败
        print(f"[warn] 写日志失败: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 启动 ----
    try:
        state["predictor"] = HousingPredictor.load(MODEL_PATH)
        state["load_error"] = None
        print(f"[startup] 模型已加载: {MODEL_PATH}")
    except Exception as exc:  # noqa: BLE001 —— 故意兜住：让服务起来，由 /health 暴露问题
        state["predictor"] = None
        state["load_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[startup][error] 模型加载失败: {state['load_error']}")
    yield
    # ---- 关闭 ----
    state["predictor"] = None
    print("[shutdown] 已释放模型")


app = FastAPI(
    title="加州房价预测接口",
    version="1.0.0",
    description="输入一条街区的普查特征，输出预测房价（美元）。数据源：Kaggle California Housing。",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """给每个请求打一条访问日志，并把耗时塞进响应头。"""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    if request.url.path not in {"/docs", "/openapi.json", "/redoc", "/favicon.ico"}:
        print(
            f"[access] {request.method} {request.url.path} "
            f"{response.status_code} {elapsed_ms:.1f}ms"
        )
    return response


def _predictor_or_503() -> HousingPredictor:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"模型未加载: {state.get('load_error') or 'unknown'}",
        )
    return predictor


def _model_version(predictor: HousingPredictor) -> str:
    return str(predictor.metadata.get("trained_at", "unknown"))


@app.get("/health", response_model=HealthResponse, tags=["运维"])
def health() -> HealthResponse:
    """探活接口：只看进程和模型状态，不做推理。

    容器 healthcheck、负载均衡、K8s livenessProbe 都打它。
    模型没加载就返回 degraded —— 让调度器知道这个实例不该接流量。
    """
    predictor = state.get("predictor")
    return HealthResponse(
        status="ok" if predictor else "degraded",
        model_loaded=predictor is not None,
        model_version=_model_version(predictor) if predictor else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["运维"])
def model_info() -> ModelInfoResponse:
    """返回「当前线上到底是哪版模型」，出问题时第一件事就是看这个。"""
    predictor = _predictor_or_503()
    info = predictor.info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["推理"])
def predict(features: HousingFeatures) -> PredictionResponse:
    """单条预测。请求体是 JSON，不是 query 参数 —— 9 个字段塞 URL 里没人受得了。"""
    predictor = _predictor_or_503()
    started = time.perf_counter()
    record = features.to_record()

    try:
        price = predictor.predict_one(record)
    except ValueError as exc:
        # 输入合法但模型算不出来（NaN 等）-> 422，告诉我们数据有问题
        _append_log({"ts": _now(), "event": "predict_error", "error": str(exc), "input": record})
        raise HTTPException(status_code=422, detail=f"模型无法处理该输入: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        # 未知异常 -> 500，并且必须记日志，否则线上永远查不出原因
        _append_log({"ts": _now(), "event": "predict_crash", "error": repr(exc), "input": record})
        raise HTTPException(status_code=500, detail="内部推理错误") from exc

    _append_log(
        {
            "ts": _now(),
            "event": "predict",
            "input": record,
            "output": price,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model_version": _model_version(predictor),
        }
    )
    return PredictionResponse(
        predicted_house_value=round(price, 2),
        model_version=_model_version(predictor),
    )


@app.post("/predict_batch", response_model=BatchResponse, tags=["推理"])
def predict_batch(payload: BatchRequest) -> BatchResponse:
    """批量预测：一次请求多算几条，省掉网络往返。

    为什么限制 1000 条？因为无上限的批量接口 = 给攻击者一个免费 DDoS 按钮。
    """
    predictor = _predictor_or_503()
    records = [r.to_record() for r in payload.records]
    prices = predictor.predict(records)
    _append_log(
        {
            "ts": _now(),
            "event": "predict_batch",
            "count": len(records),
            "output_mean": sum(prices) / len(prices),
            "model_version": _model_version(predictor),
        }
    )
    return BatchResponse(
        predictions=[round(p, 2) for p in prices],
        count=len(prices),
        model_version=_model_version(predictor),
    )


DEMO_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>加州房价预测</title>
<style>
 body{font-family:-apple-system,"Segoe UI",sans-serif;margin:0;padding:16px;background:#f6f7f9;color:#111}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:#666;font-size:13px;margin:0 0 14px}
 label{display:block;font-size:13px;margin:10px 0 4px;color:#333}
 input,select{width:100%;box-sizing:border-box;padding:10px;font-size:16px;
   border:1px solid #dcdfe6;border-radius:8px;background:#fff}
 button{margin-top:18px;width:100%;padding:14px;font-size:16px;border:0;
   border-radius:8px;background:#2563eb;color:#fff}
 button:disabled{background:#9ca3af}
 #out{margin-top:18px;padding:14px;border-radius:8px;background:#fff;
   font-size:15px;white-space:pre-wrap;min-height:22px;border:1px solid #eef0f3}
 .ok{color:#065f46}.err{color:#b91c1c}
</style></head><body>
<h1>加州房价预测</h1>
<p class="sub">输入街区特征，返回预测房价（美元）。无外部依赖，手机也能开。</p>
<div id="fields"></div>
<label>ocean_proximity</label>
<select id="ocean_proximity">
 <option>NEAR BAY</option><option>&lt;1H OCEAN</option><option>INLAND</option>
 <option>NEAR OCEAN</option><option>ISLAND</option>
</select>
<button id="go">预测</button>
<div id="out">等待输入…</div>
<script>
const F=[["longitude","-122.23"],["latitude","37.88"],["housing_median_age","41"],
["total_rooms","880"],["total_bedrooms","129"],["population","322"],
["households","126"],["median_income","8.3252"]];
const box=document.getElementById("fields");
for(const [k,v] of F){
  box.insertAdjacentHTML("beforeend",'<label>'+k+'</label><input id="'+k+'" value="'+v+'">');
}
const out=document.getElementById("out"), go=document.getElementById("go");
go.onclick=async()=>{
  const body={};
  for(const [k] of F){
    const raw=document.getElementById(k).value.trim();
    body[k]= raw===""? null : Number(raw);
  }
  body.ocean_proximity=document.getElementById("ocean_proximity").value;
  go.disabled=true; out.className=""; out.textContent="请求中…";
  try{
    const r=await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const j=await r.json();
    if(r.ok){
      out.className="ok";
      out.textContent="预测房价：$"+Number(j.predicted_house_value).toLocaleString()+
        "\\n模型版本："+j.model_version;
    }else{
      out.className="err";
      out.textContent="HTTP "+r.status+"\\n"+JSON.stringify(j.error?j.error.detail:j,null,2);
    }
  }catch(e){ out.className="err"; out.textContent="请求失败："+e; }
  go.disabled=false;
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    """一个完全本地的小表单页。

    为什么需要它？FastAPI 自带的 /docs 会去 CDN 下载 Swagger 的 JS，
    在手机、内网、或没外网的机器上经常一直转圈。这个页面零外部依赖，
    打开就能填数字点按钮 —— 演示和跨设备验收都靠它。
    """
    return DEMO_PAGE


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """把 pydantic 的校验错误也收进统一格式。

    这里是个容易忽略的坑：``RequestValidationError`` 不是 ``HTTPException`` 的子类，
    所以只写 HTTPException 的处理器，422 会走 FastAPI 的默认格式，
    前端就得多写一套解析逻辑。

    另外故意只挑 loc / msg / type 三个字段：pydantic 的 errors() 里
    ctx 可能带不可序列化的对象，直接 json.dumps 会 500。
    """
    detail = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": {"status": 422, "detail": detail, "path": request.url.path}},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """统一错误结构，前端只需要认 {error: {...}} 一种格式。

    注意注册的是 **starlette 的** HTTPException，不是 fastapi 的那个：
    fastapi.HTTPException 是它的子类，所以注册父类能一次覆盖
    「我们自己抛的 503/422」和「框架抛的 404/405」，
    否则你的 404 会突然变成另一种格式，前端又得多写一个分支。
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"status": exc.status_code, "detail": exc.detail, "path": request.url.path}},
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
