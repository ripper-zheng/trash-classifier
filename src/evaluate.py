"""评估脚本：测试集指标、混淆矩阵、Top-3、正确/错误案例。

用法：
    python src/evaluate.py --checkpoint models/best_model.pth --out report

产出（供报告第 7、9 章与演示视频使用）：
    report/metrics.json          测试指标
    report/confusion_matrix.png  6×6 混淆矩阵
    report/cases/                正确与错误案例图
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from config import BEST_MODEL, CLASSES, DATA_DIR, CLASS_CN, IMG_SIZE, REPORT_DIR, get_device, get_transforms
from dataset import build_loader
from model import build_model

# 中文字体兜底（若无中文字体，混淆矩阵用英文标签也能正常生成）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(checkpoint: Path, device):
    model = build_model("resnet18", pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(BEST_MODEL))
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(REPORT_DIR))
    args = parser.parse_args()

    device = get_device()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到权重 {ckpt}，请先运行训练。")

    model = load_model(ckpt, device)
    test_loader = build_loader("test", Path(args.data), batch_size=32, shuffle=False)
    print(f"[数据] test={len(test_loader.dataset)}  [设备] {device}")

    ys, preds, probs_list = [], [], []
    correct_imgs, wrong_imgs = [], []
    paths = []
    # 记录每张测试图路径（ImageFolder 的 samples 顺序与 loader 一致，shuffle=False）
    for images, labels in test_loader:
        with torch.no_grad():
            out = model(images.to(device))
            probs = out.softmax(dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        ys.extend(labels.numpy().tolist())
        preds.extend(pred.tolist())
        probs_list.extend(probs.tolist())

    ys = np.array(ys)
    preds = np.array(preds)
    probs = np.array(probs_list)

    # ---------- 指标 ----------
    metrics = {
        "checkpoint": str(ckpt),
        "test_size": int(len(ys)),
        "top1_accuracy": float(accuracy_score(ys, preds)),
        "macro_precision": float(precision_score(ys, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(ys, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "classification_report": classification_report(
            ys, preds, target_names=CLASSES, output_dict=True, zero_division=0),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[指标]")
    print(f"  Top-1 Accuracy : {metrics['top1_accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")

    # ---------- 单图耗时（CPU, batch=1，去掉首次预热） ----------
    infer_transform = get_transforms("infer")
    from PIL import Image
    sample = next((Path(args.data) / "test").rglob("*.jpg"), None)
    latency_ms = None
    if sample:
        x = infer_transform(Image.open(sample).convert("RGB")).unsqueeze(0).to(device)
        model(x)  # 预热
        t0 = time.time()
        reps = 20
        with torch.no_grad():
            for _ in range(reps):
                model(x)
        latency_ms = (time.time() - t0) / reps * 1000
        metrics["latency_ms_per_image"] = round(latency_ms, 2)
        print(f"  单图推理耗时  : {latency_ms:.1f} ms ({device})")

    # ---------- 混淆矩阵 ----------
    cm = confusion_matrix(ys, preds, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("测试集混淆矩阵")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"[图] {out_dir / 'confusion_matrix.png'}")

    # ---------- 案例图（≥5 正确 + ≥2 错误） ----------
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    # 通过 ImageFolder 的有序 samples 对齐图片路径
    ds = build_loader("test", Path(args.data), shuffle=False).dataset
    img_paths = [Path(p) for p, _ in ds.samples]
    correct_idx = np.where(preds == ys)[0].tolist()
    wrong_idx = np.where(preds != ys)[0].tolist()
    for tag, idxs in [("correct", correct_idx[:5]), ("wrong", wrong_idx[:2])]:
        for n, i in enumerate(idxs):
            src = img_paths[i]
            name = f"{tag}_{n+1}_true-{CLASSES[ys[i]]}_pred-{CLASSES[preds[i]]}_{src.name}"
            shutil.copy2(src, cases_dir / name)
    print(f"[案例] 正确 {len(correct_idx[:5])} 张、错误 {len(wrong_idx[:2])} 张 → {cases_dir}")

    # ---------- 错误案例表（文本） ----------
    err_lines = ["| 真实类别 | 预测类别 | 置信度 | 文件 |", "|---|---|---|---|"]
    for i in wrong_idx[:10]:
        err_lines.append(f"| {CLASS_CN.get(CLASSES[ys[i]], CLASSES[ys[i]])} | "
                         f"{CLASS_CN.get(CLASSES[preds[i]], CLASSES[preds[i]])} | "
                         f"{probs[i][preds[i]]:.3f} | {img_paths[i].name} |")
    (out_dir / "error_cases.md").write_text("\n".join(err_lines), encoding="utf-8")
    print(f"[表] {out_dir / 'error_cases.md'}")


if __name__ == "__main__":
    main()
