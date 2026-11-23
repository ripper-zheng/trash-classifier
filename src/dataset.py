"""Dataset 封装：用 torchvision.datasets.ImageFolder 读取划分目录。

关键点：ImageFolder 会按目录名排序生成 classes，必须校验它与 config.CLASSES
完全一致；若不一致立即报错，避免类别索引错位（三条链路一致性）。
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import CLASSES, BATCH_SIZE, NUM_WORKERS, get_transforms


def build_dataset(split: str, data_dir: Path):
    root = data_dir / split
    if not root.exists():
        raise FileNotFoundError(f"未找到数据划分目录 {root}，请先运行 prepare_data.py")
    ds = ImageFolder(str(root), transform=get_transforms(split))
    if ds.classes != CLASSES:
        raise ValueError(
            f"类别顺序不一致！目录={ds.classes}，配置={CLASSES}。"
            "请检查 dataset/<split>/ 下的类别子目录名与顺序。"
        )
    return ds


def build_loader(split: str, data_dir: Path, batch_size: int = BATCH_SIZE,
                 shuffle: bool = False):
    ds = build_dataset(split, data_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
