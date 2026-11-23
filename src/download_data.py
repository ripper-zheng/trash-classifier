"""下载 TrashNet 六分类数据集。

TrashNet 官方仓库：https://github.com/garythung/trashnet
数据文件：dataset.tar.gz（约 300MB），含 six categories:
    cardboard, glass, metal, paper, plastic, trash

用法：
    python src/download_data.py --output dataset/raw

若自动下载失败（网络受限），请手动下载后解压到 output 目录：
    1. 浏览器打开 https://github.com/garythung/trashnet
    2. 下载 dataset.tar.gz
    3. 解压后得到 class-*.jpg 命名的图片（如 glass1.jpg）
    4. 把图片按类别放入 output/<类别>/ 子目录，再运行 prepare_data.py
"""
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

DATA_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset.tar.gz"
# 备选镜像（如官方链接失效）
MIRROR_URL = "https://dataset.cmu.edu/trashnet/dataset.tar.gz"

# TrashNet 文件名前缀到类别的映射（官方命名：cardboard1.jpg, glass2.jpg, ...）
PREFIX_TO_CLASS = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
}


def download(url: str, dst: Path) -> bool:
    print(f"[下载] {url}")
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"[完成] 保存到 {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[失败] {e}")
        return False


def classify_by_name(filename: str):
    lower = filename.lower()
    for prefix, cls in PREFIX_TO_CLASS.items():
        if lower.startswith(prefix):
            return cls
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/raw", help="原始数据输出目录")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "dataset.tar.gz"

    if not archive.exists():
        ok = download(DATA_URL, archive)
        if not ok:
            ok = download(MIRROR_URL, archive)
        if not ok:
            print("\n自动下载失败，请按文件顶部说明手动下载并解压。")
            return

    print("[解压] ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out)
    archive.unlink()  # 解压后删除压缩包，节省空间

    # TrashNet 解压后是平铺的 class-*.jpg，整理到类别子目录
    image_exts = {".jpg", ".jpeg", ".png"}
    moved = 0
    for f in out.iterdir():
        if f.suffix.lower() not in image_exts or not f.is_file():
            continue
        cls = classify_by_name(f.name)
        if cls is None:
            continue
        cls_dir = out / cls
        cls_dir.mkdir(exist_ok=True)
        shutil.move(str(f), str(cls_dir / f.name))
        moved += 1

    print(f"\n[整理完成] 共归类 {moved} 张图片到 6 个类别子目录。")
    for cls in PREFIX_TO_CLASS.values():
        cnt = len(list((out / cls).glob("*"))) if (out / cls).exists() else 0
        print(f"  {cls}: {cnt}")
    print("\n下一步：python src/prepare_data.py --raw dataset/raw --out dataset")


if __name__ == "__main__":
    main()
