"""提交前自检脚本：核对硬性要求的文件是否齐备。

用法：python src/check_submission.py
只做文件存在性检查，不校验内容真实性（真实性须人工确认）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 硬性要求必须存在的文件/目录
REQUIRED = [
    "README.md",
    "requirements.txt",
    "source_repository.txt",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/model.py",
    "src/app.py",
    "frontend/index.html",
    "frontend/style.css",
    "frontend/script.js",
    "dataset/dataset_manifest.csv",
    "dataset/samples",
    "runs",
    "models/best_model.pth",
    "report/项目设计报告.docx",
    "video/演示视频链接.txt",
]

# 报告五大必含小节关键词（对 .docx 做纯文本粗查）
REPORT_KEYWORDS = ["真实数据集证明", "PyTorch 模型文件", "TensorBoard 日志", "Loss", "运行说明"]


def check_report(docx_path: Path):
    """用 zipfile 读取 docx 的 document.xml，粗查关键词是否出现。"""
    import re
    import zipfile

    if not docx_path.exists():
        return False, []
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml))
    missing = [k for k in REPORT_KEYWORDS if k not in text]
    return len(missing) == 0, missing


def main():
    print("=== 文件齐备性检查 ===")
    missing = []
    for rel in REQUIRED:
        p = ROOT / rel
        ok = p.exists()
        tag = "[OK]" if ok else "[缺失]"
        count = ""
        if ok and p.is_dir() and rel.startswith("runs"):
            n = len([d for d in p.iterdir() if d.is_dir()])
            count = f" ({n} 个实验目录)"
            if n < 3:
                tag = "[不足]"
        print(f"  {tag} {rel}{count}")
        if not ok:
            missing.append(rel)

    print("\n=== 报告五大必含小节粗查 ===")
    docx = ROOT / "report" / "项目设计报告.docx"
    ok, miss = check_report(docx)
    if docx.exists():
        print("  [OK] 五个关键词均在报告中" if ok else f"  [缺失关键词] {miss}")
    else:
        print("  [跳过] 报告尚未生成（report/项目设计报告.docx）")

    print("\n=== 源码链接检查 ===")
    src_txt = (ROOT / "source_repository.txt").read_text(encoding="utf-8")
    if "<你的用户名>" in src_txt:
        print("  [待填] source_repository.txt 仍是占位地址，请替换为真实仓库链接")
    else:
        print("  [OK] 仓库地址已填写（请确认浏览器可打开）")

    print("\n=== 结论 ===")
    if not missing:
        print("  硬性文件齐备。请继续人工确认：数据真实性、指标真实性、曲线截图、视频现场演示。")
    else:
        print(f"  仍缺 {len(missing)} 项：{', '.join(missing)}")


if __name__ == "__main__":
    main()
