#!/usr/bin/env bash
# 自动排队：等 E3 训练真正完成后，依次跑 E1、E2，最后评估。
# 用 history.json 作为完成标志（train.py 正常退出才写），避免 pgrep 自匹配误判。
# 串行执行，避免多进程抢 CPU。
set -u
cd "$(dirname "$0")" || exit 1
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY="python"

E3_DONE="runs/E3_resnet18_finetune/history.json"
echo "[$(date '+%H:%M:%S')] 等待 E3 完成标志: $E3_DONE"
while [ ! -f "$E3_DONE" ]; do
  sleep 30
done
echo "[$(date '+%H:%M:%S')] E3 已完成"
ls -lh models/best_model.pth 2>/dev/null || echo "[警告] best_model.pth 未找到"

echo "[$(date '+%H:%M:%S')] === 启动 E1_cnn ==="
"$PY" -u src/train.py --exp E1_cnn > runs/E1_train.log 2>&1
tail -3 runs/E1_train.log

echo "[$(date '+%H:%M:%S')] === 启动 E2_resnet18_frozen ==="
"$PY" -u src/train.py --exp E2_resnet18_frozen > runs/E2_train.log 2>&1
tail -3 runs/E2_train.log

echo "[$(date '+%H:%M:%S')] === 运行评估 ==="
"$PY" -u src/evaluate.py --checkpoint models/best_model.pth --out report > runs/evaluate.log 2>&1
tail -12 runs/evaluate.log

echo "[$(date '+%H:%M:%S')] 全部完成"
ls -lh models/
find runs -name history.json
