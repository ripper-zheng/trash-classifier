#!/usr/bin/env bash
# 从零复现的完整命令流（Linux/Mac；Windows 用 Git Bash 或照 README 逐条执行）
set -e

echo "=== 1. 环境 ==="
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "=== 2. 数据 ==="
python src/download_data.py --output dataset/raw
python src/prepare_data.py --raw dataset/raw --out dataset --seed 42

echo "=== 3. 三组训练实验 ==="
python src/train.py --exp E1_cnn
python src/train.py --exp E2_resnet18_frozen
python src/train.py --exp E3_resnet18_finetune

echo "=== 4. 查看曲线（另开终端） ==="
echo "tensorboard --logdir runs --port 6006"

echo "=== 5. 评估 ==="
python src/evaluate.py --checkpoint models/best_model.pth --out report

echo "=== 6. 单图推理 ==="
python src/predict.py --image dataset/samples/某张图片.jpg --checkpoint models/best_model.pth

echo "=== 7. 前端 ==="
python src/app.py
echo "浏览器打开 http://127.0.0.1:5000"
