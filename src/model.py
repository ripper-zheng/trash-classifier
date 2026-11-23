"""模型定义：自定义轻量 CNN 与 ResNet18 迁移学习。

训练与推理共同导入本文件（禁止复制两份不同版本），保证结构一致。
"""
import torch.nn as nn
from torchvision import models

from config import CLASSES


class SimpleCNN(nn.Module):
    """E1 基线：从零训练的轻量 CNN。输入 3×224×224。"""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TrashClassifierResNet(nn.Module):
    """E2/E3：ImageNet 预训练 ResNet18，替换 fc 为 6 类输出。"""

    def __init__(self, num_classes: int = len(CLASSES), pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
        if freeze_backbone:
            # 冻结除 fc 之外的所有卷积层，只训练分类头
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


def build_model(name: str, pretrained: bool = True, freeze_backbone: bool = False,
                num_classes: int = len(CLASSES)) -> nn.Module:
    """按名称构造模型。name ∈ {simple_cnn, resnet18}。"""
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return TrashClassifierResNet(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"未知模型: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
