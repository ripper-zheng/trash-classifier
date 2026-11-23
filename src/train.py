"""训练脚本：三组对照实验、TensorBoard 记录、固定种子、保存最佳权重。

用法：
    python src/train.py --exp E1_cnn
    python src/train.py --exp E2_resnet18_frozen
    python src/train.py --exp E3_resnet18_finetune

每次实验写入独立 runs/<实验名>/ 目录，互不覆盖。
最佳权重按验证集 Macro-F1 选取，保存到 models/best_model.pth（仅 is_final 实验写入最终文件）。
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from config import (
    BEST_MODEL, CLASSES, DATA_DIR, EXPERIMENTS, MODEL_DIR, RUNS_DIR,
    SEED, set_seed, get_device,
)
from dataset import build_loader
from model import build_model, count_parameters


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0))
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0),
        )
    raise ValueError(f"未知优化器: {cfg['optimizer']}")


def build_scheduler(optimizer, cfg):
    if cfg.get("scheduler") == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if cfg.get("scheduler") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ys, preds = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            ys.extend(labels.cpu().tolist())
            preds.extend(pred.cpu().tolist())
    acc = correct / total if total else 0.0
    macro_f1 = f1_score(ys, preds, average="macro", labels=list(range(len(CLASSES))), zero_division=0)
    return total_loss / total if total else 0.0, acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()),
                        help="实验名，对应 config.EXPERIMENTS")
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(EXPERIMENTS[args.exp])
    if args.epochs:
        cfg["epochs"] = args.epochs
    batch_size = args.batch_size or 32

    set_seed(SEED)
    device = get_device()
    print(f"[实验] {args.exp}: {cfg['desc']}")
    print(f"[设备] {device}  [种子] {SEED}  [Epochs] {cfg['epochs']}")

    # 数据
    train_loader = build_loader("train", Path(args.data), batch_size=batch_size, shuffle=True)
    val_loader = build_loader("val", Path(args.data), batch_size=batch_size, shuffle=False)
    print(f"[数据] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    # 模型
    model = build_model(
        cfg["model"],
        pretrained=cfg.get("pretrained", False),
        freeze_backbone=cfg.get("freeze_backbone", False),
    ).to(device)
    print(f"[模型] {cfg['model']} 可训练参数 {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    log_dir = RUNS_DIR / args.exp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_f1, best_epoch = -1.0, -1
    history = {"exp": args.exp, "cfg": cfg, "epochs": []}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # 每组实验都保存自己的最优权重；最终实验另存为 best_model.pth
    exp_ckpt = MODEL_DIR / f"{args.exp}_best.pth"

    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer, device, False)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", va_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.flush()

        history["epochs"].append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "train_acc": tr_acc, "val_acc": va_acc, "val_macro_f1": va_f1, "lr": current_lr,
        })
        print(f"[{epoch:02d}/{cfg['epochs']}] "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  acc {tr_acc:.4f}/{va_acc:.4f}  "
              f"val_f1 {va_f1:.4f}  lr {current_lr:.2e}")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            torch.save(model.state_dict(), exp_ckpt)
            if cfg.get("is_final"):
                torch.save(model.state_dict(), BEST_MODEL)
            # 保存类别映射等元信息，避免部署时拿到无法解释的 .pth
            (MODEL_DIR / f"{args.exp}_meta.json").write_text(json.dumps({
                "classes": CLASSES,
                "model": cfg["model"],
                "input_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    writer.close()
    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = best_f1
    (log_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] {args.exp} 最佳 Epoch={best_epoch} val_macro_f1={best_f1:.4f} "
          f"耗时 {time.time()-t0:.1f}s")
    print(f"[权重] {exp_ckpt}" + (f"；最终模型 {BEST_MODEL}" if cfg.get("is_final") else ""))
    print(f"[日志] tensorboard --logdir runs --port 6006")


if __name__ == "__main__":
    main()
