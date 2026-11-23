#!/usr/bin/env bash
# 自动排队完成剩余实验：等待当前 E3 结束后，依次跑 E1、E2，最后评估。
# 用法：bash run_remaining_experiments.sh
# 说明：串行执行，避免多进程抢 CPU。日志分别写入 runs/*.log。
#       每组训练把 stdout/stderr 重定向到日志文件并加 -u 关闭缓冲，方便实时查看进度。

set -u
cd "$(dirname "$0")" || exit 1
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY="python"

echo "[$(date '+%H:%M:%S')] 等待 E3 训练进程结束..."
while pgrep -f "train.py --exp E3_resnet18_finetune" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date '+%H:%M:%S')] E3 已结束，检查产物..."
ls -lh models/best_model.pth 2>/dev/null || echo "[警告] best_model.pth 未找到，请检查 E3 日志"

echo "[$(date '+%H:%M:%S')] === 启动 E1_cnn ==="
"$PY" -u src/train.py --exp E1_cnn > runs/E1_train.log 2>&1
echo "[$(date '+%H:%M:%S')] E1 完成，末尾日志："
tail -3 runs/E1_train.log

echo "[$(date '+%H:%M:%S')] === 启动 E2_resnet18_frozen ==="
"$PY" -u src/train.py --exp E2_resnet18_frozen > runs/E2_train.log 2>&1
echo "[$(date '+%H:%M:%S')] E2 完成，末尾日志："
tail -3 runs/E2_train.log

echo "[$(date '+%H:%M:%S')] === 运行评估 ==="
"$PY" -u src/evaluate.py --checkpoint models/best_model.pth --out report > runs/evaluate.log 2>&1
echo "[$(date '+%H:%M:%S')] 评估完成，末尾日志："
tail -10 runs/evaluate.log

echo "[$(date '+%H:%M:%S')] 全部实验完成。"
echo "=== 汇总 ==="
ls -lh models/
find runs -name history.json -exec echo {} \;
