"""收尾总控：训练全部完成后，自动完成报告终稿与 DOCX 生成。

流程：
1. 等待三组 history.json + metrics.json 就绪
2. 用最终 best_model.pth 对演示图做一次真实推理 → report/raw_predict.json
3. finalize_report.py 回填初稿 → report/报告正文终稿.md + 训练曲线图
4. md2html.py 转 HTML（终稿）→ html-review 门禁
5. convert_doc.py 生成 report/项目设计报告.docx

用法：python src/finalize_all.py（可后台常驻，训练完成后自动执行）
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)
PLUGIN = Path("C:/Users/Administrator/.workbuddy/plugins/cache/workbuddy-builtin/workbuddy-docx/5.11.3-wb.43806934.gecace0f7.ha395516167d4")
WORK = Path("C:/Users/Administrator/Desktop/work/output/report-20261123")
REPORT = ROOT / "report"
EXPS = ["E1_cnn", "E2_resnet18_finetune", "E3_resnet18_finetune"]


def run(cmd, desc, check=True):
    print(f"\n=== {desc} ===", flush=True)
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines()[-12:]:
        print("  " + line, flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{desc} 失败 (exit {r.returncode})")
    return r


def wait_ready():
    print("[等待] 三组实验 + 评估全部完成...", flush=True)
    while True:
        done = all((ROOT / "runs" / e / "history.json").exists() for e in EXPS)
        metrics = (REPORT / "metrics.json").exists()
        if done and metrics:
            print("[就绪] 全部完成，开始收尾", flush=True)
            return
        time.sleep(60)


def make_predict_json():
    """用最终权重对演示图做一次真实推理，保存原始 JSON。"""
    demo = REPORT.parent / "dataset" / "demo_images" / "cardboard_00394.jpg"
    if not demo.exists():
        demo = ROOT / "dataset" / "demo_images" / "cardboard_00394.jpg"
    out = REPORT / "raw_predict.json"
    r = run([PY, "src/predict.py", "--image", str(demo),
             "--checkpoint", "models/best_model.pth"],
            "生成一次真实推理输出", check=False)
    try:
        stdout = r.stdout or ""
        start = stdout.find("{")
        data = json.loads(stdout[start:]) if start >= 0 else None
        if data and data.get("status") == "success":
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  已保存 {out}", flush=True)
        else:
            print("  [警告] 未得到成功推理结果，报告 5.5 节将保留占位", flush=True)
    except Exception as e:
        print(f"  [警告] JSON 解析失败({e})，报告 5.5 节将保留占位", flush=True)


def main():
    wait_ready()
    make_predict_json()

    # 回填
    run([PY, "src/finalize_report.py"], "回填报告终稿")

    # md -> html（终稿）
    html = WORK / "stage2" / "formatted-项目设计报告.html"
    run([PY, str(WORK / "md2html.py"),
         str(REPORT / "报告正文终稿.md"), str(html)], "终稿转 HTML")

    # html 门禁
    gate = PLUGIN / "skills" / "html-review" / "scripts" / "review_html.py"
    run([sys.executable, str(gate), "--html", str(html),
         "--genre", "general", "--require-page-furniture"], "HTML 质量门禁", check=False)

    # html -> docx
    conv = PLUGIN / "skills" / "html-to-doc" / "scripts" / "convert_doc.py"
    docx = REPORT / "项目设计报告.docx"
    run([sys.executable, str(conv), "--html", str(html),
         "--output", str(docx), "--require-page-furniture"], "生成 Word 文档")

    print(f"\n[完成] 终稿: {REPORT / '报告正文终稿.md'}", flush=True)
    print(f"[完成] Word: {docx}", flush=True)


if __name__ == "__main__":
    main()
