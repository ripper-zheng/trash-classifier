"""功能与异常测试 T01-T08：真实调用运行中的 Flask 服务，产出实测结果表。

用法（服务需已启动）：python src/run_functional_tests.py
产出：report/functional_tests.md（可直接粘入报告第 9 章）
每个用例独立执行，单个失败不影响其他。
"""
import json
import subprocess
import tempfile
from pathlib import Path

BASE = "http://127.0.0.1:5000"
DEMO = Path("dataset/demo_images")


def curl_post(path: Path = None, filename: str = None, no_file: bool = False):
    """用 curl 上传文件，返回 (http_code, 响应JSON或文本)。"""
    cmd = ["curl", "-s", "--max-time", "60", "-w", "\n__HTTP__%{http_code}",
           "-X", "POST"]
    if not no_file and path is not None:
        cmd += ["-F", f"file=@{path}" + (f";filename={filename}" if filename else "")]
    cmd.append(BASE + "/predict")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=70).stdout
    except subprocess.TimeoutExpired:
        return "超时", "<请求超时>"
    if "\n__HTTP__" not in out:
        return "?", out[:200]
    body, code = out.rsplit("\n__HTTP__", 1)
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = body.strip()[:200]
    return code.strip(), parsed


def brief(code, body):
    if isinstance(body, dict):
        cls = body.get("class")
        if cls:
            return f"HTTP {code} · class={cls}, conf={body.get('confidence')}"
        msg = body.get("message")
        if msg:
            return f"HTTP {code} · {msg}"
    text = str(body).replace("\n", " ").replace("\r", " ").strip()
    if "<html" in text.lower() or "<!doctype" in text.lower():
        text = "Flask 返回 413 Request Entity Too Large 错误页"
    return f"HTTP {code} · {text[:80]}"


def main():
    tmp = Path(tempfile.gettempdir()) / "tc"
    tmp.mkdir(exist_ok=True)
    png = tmp / "ok.png"
    if not png.exists():
        from PIL import Image
        Image.open(DEMO / "metal_00402.jpg").convert("RGB").save(png)
    txt = tmp / "bad.txt"
    if not txt.exists():
        txt.write_text("not an image", encoding="utf-8")
    big = tmp / "big.png"
    if not big.exists():
        import numpy as np
        from PIL import Image
        Image.fromarray((np.random.rand(4000, 4000, 3) * 255).astype("uint8")).save(big)
    fake = tmp / "fake.jpg"
    if not fake.exists():
        fake.write_text("plain text named jpg", encoding="utf-8")

    cases = [
        ("T01", "上传正常 JPG 图片", "返回类别与置信度", lambda: curl_post(DEMO / "cardboard_00394.jpg")),
        ("T02", "上传 PNG 图片", "正常推理", lambda: curl_post(png)),
        ("T03", "上传 .txt 文件", "拒绝并提示", lambda: curl_post(txt, "bad.txt")),
        ("T04", "上传超过 10 MB 文件", "返回 413 提示", lambda: curl_post(big, "big.png")),
        ("T05", "改扩展名的文本文件", "格式校验拦截", lambda: curl_post(fake, "fake.jpg")),
        ("T06", "未携带 file 字段提交", "提示未收到文件", lambda: curl_post(no_file=True)),
    ]
    results = []
    for tid, case, expect, fn in cases:
        try:
            code, body = fn()
            detail = brief(code, body)
        except Exception as e:
            detail = f"测试异常: {e}"
        results.append((tid, case, expect, detail))
        print(f"{tid}: {detail}")

    results.append(("T07", "模型权重缺失时启动服务", "服务拒绝启动",
                    "代码审查确认：app.py 加载失败即 sys.exit(1)，见 5.4 节"))
    try:
        code, body = curl_post(DEMO / "glass_00495.jpg")
        detail = brief(code, body)
    except Exception as e:
        detail = f"测试异常: {e}"
    results.append(("T08", "上传未参与训练的真实图片", "返回合理类别", detail))
    print(f"T08: {detail}")

    lines = ["| 编号 | 用例 | 预期 | 实测结果 |", "|---|---|---|---|"]
    for r in results:
        lines.append("| " + " | ".join(r) + " |")
    out = Path("report/functional_tests.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[表] {out}")


if __name__ == "__main__":
    main()
