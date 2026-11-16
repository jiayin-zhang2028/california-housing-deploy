"""pytest 共享夹具。

关键点：测试**不要**依赖你本地那份 artifacts/model_bundle.joblib。
每个测试会话现场训一个「迷你模型」，这样：
  * CI 机器上没有你的模型文件也能跑；
  * 训练代码改了，测试立刻失败 —— 这才是测试的价值。
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """用合成数据现场训练一个小模型，返回 bundle 路径。"""
    import src.train as train_mod

    out_dir = tmp_path_factory.mktemp("artifacts")
    train_mod.ARTIFACT_DIR = out_dir  # 把产物写进临时目录，不污染工作区

    args = argparse.Namespace(
        data=None,
        synthetic=True,
        test_size=0.2,
        max_iter=80,  # 测试只要「能跑通」，不追求精度
        seed=7,
    )
    train_mod.train(args)
    path = out_dir / "model_bundle.joblib"
    assert path.exists(), "训练脚本没有产出 bundle"
    return path


@pytest.fixture(scope="session")
def predictor(bundle_path: Path):
    from src.inference import HousingPredictor

    return HousingPredictor.load(bundle_path)


@pytest.fixture(scope="session")
def client(bundle_path: Path, tmp_path_factory: pytest.TempPathFactory):
    """启动一个真正的 FastAPI 应用（含 lifespan），用 TestClient 打请求。"""
    os.environ["MODEL_PATH"] = str(bundle_path)
    os.environ["PREDICTION_LOG"] = str(tmp_path_factory.mktemp("logs") / "predictions.jsonl")

    import app.main as main_mod

    main_mod = importlib.reload(main_mod)  # 让模块级常量读到刚设置的环境变量

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as test_client:
        yield test_client
