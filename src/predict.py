"""独立推理模块：加载最佳权重，对单张图片预测。

既可作为命令行工具使用，也被 app.py 导入。训练和推理共用 config.get_transforms('infer')，
保证预处理一致（三条链路一致性）。
"""
from pathlib import Path

import torch
from PIL import Image

from config import (
    ADVICE, BEST_MODEL, CLASS_CN, CLASSES, IDX_TO_CLASS,
    MODEL_NAME, MODEL_VERSION, ROOT, get_device, get_transforms,
)
from model import build_model

_model = None
_device = None


def load_model(checkpoint: Path = None):
    """全局加载一次模型；加载失败抛异常，供 app 启动时拒绝服务。"""
    global _model, _device
    ckpt = Path(checkpoint) if checkpoint else BEST_MODEL
    if not ckpt.exists():
        raise FileNotFoundError(f"模型权重不存在: {ckpt}")
    _device = get_device()
    model = build_model("resnet18", pretrained=False).to(_device)
    state = torch.load(ckpt, map_location=_device)
    model.load_state_dict(state)
    model.eval()
    _model = model
    return model


def get_model():
    if _model is None:
        load_model()
    return _model, _device


def predict_image(image: Image.Image, topk: int = 3) -> dict:
    """对 PIL 图片推理，返回结构化结果（不含耗时，耗时由调用方统计）。"""
    model, device = get_model()
    transform = get_transforms("infer")
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = out.softmax(dim=1).cpu().numpy()[0]
    order = probs.argsort()[::-1][:topk]
    top3 = [
        {"class": IDX_TO_CLASS[int(i)], "cn": CLASS_CN[IDX_TO_CLASS[int(i)]],
         "score": round(float(probs[int(i)]), 4)}
        for i in order
    ]
    top1 = IDX_TO_CLASS[int(order[0])]
    confidence = float(probs[int(order[0])])
    # 低置信度提示人工确认（不能直接把高置信度当正确）
    need_review = confidence < 0.6
    return {
        "status": "success",
        "class": top1,
        "cn": CLASS_CN[top1],
        "confidence": round(confidence, 4),
        "top3": top3,
        "category": "可回收物" if top1 in {"cardboard", "glass", "metal", "paper", "plastic"} else "其他垃圾",
        "advice": ADVICE[top1] + ("  （置信度偏低，建议人工确认）" if need_review else ""),
        "need_review": need_review,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


def main():
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    load_model(args.checkpoint)
    img = Image.open(args.image)
    t0 = time.time()
    result = predict_image(img)
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
