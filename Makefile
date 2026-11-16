# 用 `make xxx` 把「一堆记不住的命令」变成 5 个词。
# .RECIPEPREFIX 是为了让 recipe 用 '>' 开头而不是 Tab（Tab 在各种编辑器里太容易被吃掉）。
.RECIPEPREFIX = >
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install data train train-synthetic predict test api smoke drift docker-build docker-run all

help:
> @echo "make install          安装依赖"
> @echo "make train-synthetic  用合成数据训练（不需要 Kaggle 数据，先跑通链路）"
> @echo "make train            用 data/housing.csv 正式训练"
> @echo "make predict          本地单条预测"
> @echo "make test             跑 pytest（含接口测试）"
> @echo "make api              启动 uvicorn（前台，Ctrl+C 退出）"
> @echo "make smoke            对已启动的服务发冒烟请求"
> @echo "make drift            用预测日志算输入漂移"
> @echo "make docker-build     构建镜像 housing-pred:latest"
> @echo "make docker-run       起容器，映射 8000 端口"

install:
> $(PY) -m pip install -r requirements.txt

train-synthetic:
> $(PY) -m src.train --synthetic

train:
> $(PY) -m src.train --data data/housing.csv

predict:
> $(PY) -m src.predict

test:
> $(PY) -m pytest

api:
> $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

smoke:
> $(PY) -m scripts.smoke_test --base-url http://127.0.0.1:8000

drift:
> $(PY) -m monitor.drift --log logs/predictions.jsonl

docker-build:
> docker build -t housing-pred:latest .

docker-run:
> docker run --rm -p 8000:8000 --name housing-pred housing-pred:latest

all: install train-synthetic test
