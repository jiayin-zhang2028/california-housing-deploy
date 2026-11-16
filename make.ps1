# Windows 上没有 make 时用这个（PowerShell）。
# 用法：.\make.ps1 train-synthetic   或   .\make.ps1 test
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'install', 'train-synthetic', 'train', 'predict', 'test', 'api', 'smoke', 'drift', 'docker-build', 'docker-run')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'
$py = if ($env:PY) { $env:PY } else { 'python' }

switch ($Task) {
    'help' {
        Write-Host @'
.\make.ps1 install          安装依赖
.\make.ps1 train-synthetic  用合成数据训练（不需要 Kaggle 数据，先跑通链路）
.\make.ps1 train            用 data/housing.csv 正式训练
.\make.ps1 predict          本地单条预测
.\make.ps1 test             跑 pytest（含接口测试）
.\make.ps1 api              启动 uvicorn
.\make.ps1 smoke            对已启动的服务发冒烟请求
.\make.ps1 drift            用预测日志算输入漂移
.\make.ps1 docker-build     构建镜像
.\make.ps1 docker-run       起容器
'@
    }
    'install' { & $py -m pip install -r requirements.txt }
    'train-synthetic' { & $py -m src.train --synthetic }
    'train' { & $py -m src.train --data data/housing.csv }
    'predict' { & $py -m src.predict }
    'test' { & $py -m pytest }
    'api' { & $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload }
    'smoke' { & $py -m scripts.smoke_test --base-url http://127.0.0.1:8000 }
    'drift' { & $py -m monitor.drift --log logs/predictions.jsonl }
    'docker-build' { docker build -t housing-pred:latest . }
    'docker-run' { docker run --rm -p 8000:8000 --name housing-pred housing-pred:latest }
}
