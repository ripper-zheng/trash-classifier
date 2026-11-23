"""Flask 后端：提供前端页面与推理接口。

接口：
    GET  /            首页
    POST /predict     上传图片并推理，返回 JSON
    GET  /health      服务与模型状态
    GET  /model-info  模型名称、类别与输入尺寸

关键约束：
- 服务启动时加载模型，加载失败则拒绝启动（不提供带病服务）。
- 上传文件做扩展名、真实格式与大小校验，随机化文件名，处理后立即删除。
- 结果由 best_model.pth 实时推理得到，绝不在后端返回固定类别。
"""
import io
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# 允许从项目根运行：python src/app.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASSES, CLASS_CN, FRONTEND_DIR, IMG_SIZE, MODEL_NAME, MODEL_VERSION, ROOT  # noqa: E402
from predict import get_model, load_model, predict_image  # noqa: E402

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

# 启动时加载模型；失败则打印错误并退出进程（拒绝启动）
try:
    load_model()
    print(f"[OK] 模型加载成功: {MODEL_NAME} {MODEL_VERSION}")
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] 模型加载失败，服务拒绝启动: {exc}", file=sys.stderr)
    sys.exit(1)


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.route("/health")
def health():
    model, device = get_model()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    })


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "classes": CLASSES,
        "classes_cn": CLASS_CN,
        "input_size": IMG_SIZE,
    })


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "未收到文件字段 file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "未选择有效文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "不支持的文件类型，仅允许 JPG/PNG/WEBP"}), 400

    data = file.read()
    if len(data) > 10 * 1024 * 1024:
        return jsonify({"status": "error", "message": "文件超过 10 MB"}), 413
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()  # 校验真实格式，防止改扩展名的文本文件
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001
        return jsonify({"status": "error", "message": "图片损坏或格式不合法"}), 400

    t0 = time.time()
    try:
        result = predict_image(image)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": f"推理失败: {exc}"}), 500
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    # 不记录原始文件名（隐私保护）
    app.logger.info("predict ok: %s (%.1f ms)", result["class"], result["latency_ms"])
    return jsonify(result)


if __name__ == "__main__":
    # 课程现场演示用开发服务器即可；生产可用 gunicorn（见 Dockerfile）
    app.run(host="127.0.0.1", port=5000, debug=False)
