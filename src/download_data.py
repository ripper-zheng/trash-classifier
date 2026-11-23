"""下载 TrashNet 六分类数据集并整理到类别子目录。

数据源（已实测可用，2026-11）：
    官方仓库 https://github.com/garythung/trashnet
    文件 data/dataset-resized.zip（约 41 MB，2527 张已缩放图像，六分类）
    通过 GitHub API 原始内容端点下载（网络代理环境下 raw.githubusercontent 常被拦截）：
    https://api.github.com/repos/garythung/trashnet/contents/data/dataset-resized.zip
    请求头 Accept: application/vnd.github.raw+json

用法：
    python src/download_data.py --output dataset/raw
    python src/prepare_data.py --raw dataset/raw --out dataset

若 API 端点失效，可手动下载：
    浏览器打开上述 API 地址（或仓库 data/ 目录）下载 dataset-resized.zip，
    放到 output 目录，本脚本会检测到并直接解压整理。
"""
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

API_URL = (
    "https://api.github.com/repos/garythung/trashnet/contents/data/dataset-resized.zip"
)
ARCHIVE_NAME = "dataset-resized.zip"

# TrashNet 文件名前缀到类别的映射（官方命名：cardboard1.jpg, glass2.jpg, ...）
PREFIX_TO_CLASS = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
}


def download(dst: Path) -> bool:
    print(f"[下载] {API_URL}")
    req = urllib.request.Request(API_URL, headers={
        "Accept": "application/vnd.github.raw+json",
        "User-Agent": "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, dst.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        print(f"[完成] {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {exc}")
        return False


def classify_by_name(filename: str):
    lower = filename.lower()
    for prefix, cls in PREFIX_TO_CLASS.items():
        if lower.startswith(prefix):
            return cls
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/raw")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / ARCHIVE_NAME

    if not archive.exists():
        if not download(archive):
            print("\n自动下载失败，请按文件顶部说明手动下载 dataset-resized.zip 放到该目录。")
            return

    print("[解压] ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)

    # 解压后图片在 dataset-resized/<类别>/ 下，已按类别分好子目录。
    # 处理方式：把每个类别子目录整体移动到 output/<类别>/。
    extracted = out / "dataset-resized"
    src_dir = extracted if extracted.exists() else out
    image_exts = {".jpg", ".jpeg", ".png"}
    moved = 0
    skipped = 0
    for sub in sorted(src_dir.iterdir()):
        cls = sub.name.lower() if sub.is_dir() else None
        if cls not in PREFIX_TO_CLASS:
            # 平铺文件的情况：按文件名前缀归类（兼容旧版 dataset.tar.gz）
            if sub.is_file() and sub.suffix.lower() in image_exts:
                detected = classify_by_name(sub.name)
                if detected:
                    cls_dir = out / detected
                    cls_dir.mkdir(exist_ok=True)
                    shutil.move(str(sub), str(cls_dir / sub.name))
                    moved += 1
                else:
                    skipped += 1
            continue
        cls_dir = out / cls
        if cls_dir.exists():
            shutil.rmtree(cls_dir)
        shutil.move(str(sub), str(cls_dir))
        moved += len([f for f in cls_dir.iterdir() if f.suffix.lower() in image_exts])

    print(f"\n[整理完成] 归类 {moved} 张，跳过 {skipped} 张（无法识别类别）。")
    counts = {}
    for cls in PREFIX_TO_CLASS.values():
        cnt = len(list((out / cls).glob("*"))) if (out / cls).exists() else 0
        counts[cls] = cnt
        print(f"  {cls}: {cnt}")
    (out / "download_info.json").write_text(json.dumps({
        "source": "https://github.com/garythung/trashnet (data/dataset-resized.zip)",
        "api_endpoint": API_URL,
        "archive": ARCHIVE_NAME,
        "counts": counts,
        "license": "见原仓库 LICENSE；仅用于课程学习与模型评估",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n下一步：python src/prepare_data.py --raw dataset/raw --out dataset")


if __name__ == "__main__":
    main()
