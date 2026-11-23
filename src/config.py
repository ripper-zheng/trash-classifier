"""全局共享配置：类别映射、预处理、路径与实验超参。

【重要】训练、评估、推理、前端都从本文件读取类别顺序与预处理参数，
保证“三条链路一致性”（manifest 类别顺序 = Dataset 索引 = predict 输出映射）。
任何一处都不要另写一份类别列表或 transforms。
"""
from pathlib import Path

import torch
import torchvision.transforms as T

# ---------------- 路径 ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "models"
BEST_MODEL = MODEL_DIR / "best_model.pth"
REPORT_DIR = ROOT / "report"
FRONTEND_DIR = ROOT / "frontend"

# ---------------- 类别 ----------------
# 顺序即为 Dataset 类别索引、模型输出索引、predict 返回映射，三者同源。
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# 中文名与投放建议（前端展示用，由后端返回）
CLASS_CN = {
    "cardboard": "纸板",
    "glass": "玻璃",
    "metal": "金属",
    "paper": "纸张",
    "plastic": "塑料",
    "trash": "其他垃圾",
}
RECYCLABLE = {"cardboard", "glass", "metal", "paper", "plastic"}
ADVICE = {
    **{c: "可回收物：请清洁后投入对应回收箱。" for c in RECYCLABLE},
    "trash": "其他垃圾：不属于上述可回收类别，请投入其他垃圾桶。",
}

# ---------------- 预处理 ----------------
IMG_SIZE = 224
# ImageNet 标准化参数，与预训练权重一致
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    """返回指定阶段的 transforms。split ∈ {train, val, test, infer}。

    训练阶段使用随机增强；验证/测试/推理阶段只做确定性缩放裁剪，
    保证评估稳定、可复现，并与网页推理保持一致。
    """
    split = split.lower()
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    if split in ("val", "test", "infer"):
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    raise ValueError(f"未知 split: {split}")


# ---------------- 实验超参 ----------------
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0  # 课程环境建议 0，避免 Windows 多进程数据加载问题

EXPERIMENTS = {
    "E1_cnn": {
        "model": "simple_cnn",
        "pretrained": False,
        "freeze_backbone": False,
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "scheduler": "step",
        "desc": "自定义轻量 CNN，从零训练建立基线",
    },
    "E2_resnet18_frozen": {
        "model": "resnet18",
        "pretrained": True,
        "freeze_backbone": True,
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 20,
        "scheduler": "cosine",
        "desc": "ResNet18 冻结卷积层，只训练分类头",
    },
    "E3_resnet18_finetune": {
        "model": "resnet18",
        "pretrained": True,
        "freeze_backbone": False,
        "optimizer": "sgd",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "epochs": 30,
        "scheduler": "cosine",
        "desc": "ResNet18 全量微调，产出最终 best_model.pth",
        "is_final": True,
    },
}

# 最终部署模型名称/版本（前端与 /model-info 返回）
MODEL_NAME = "ResNet18-finetuned"
MODEL_VERSION = "v1.0"


def set_seed(seed: int = SEED) -> None:
    """固定随机种子，保证可复现。"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
