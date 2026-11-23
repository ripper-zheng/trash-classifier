"""数据清洗、分层划分、生成 manifest 与样例图。

流程：
    1. 从 raw/<类别>/ 读取图片，统一转 RGB/JPG
    2. 感知哈希（dHash）去重，剔除损坏与过小图片
    3. 按类别分层划分 train/val/test = 70/15/15
    4. 写出 dataset_manifest.csv（每行标注 split / sha256 / 来源）
    5. 从 val/test 抽取 10 张样例图到 samples/（每类≥1，文件名带类别）

用法：
    python src/prepare_data.py --raw dataset/raw --out dataset --seed 42
"""
import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from config import CLASSES, SEED

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIDE = 96  # 最短边小于此值人工复核，不参与训练
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(img: Image.Image, hash_size: int = 8) -> str:
    """感知哈希：缩小为 (hash_size+1)×hash_size 灰度图，比较相邻像素。"""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(g.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            a = pixels[row * (hash_size + 1) + col]
            b = pixels[row * (hash_size + 1) + col + 1]
            bits.append("1" if a > b else "0")
    return "".join(bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="dataset/raw")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    random.seed(args.seed)

    # ---------- 1. 收集与清洗 ----------
    by_class = defaultdict(list)
    seen_hashes = {}
    removed = {"corrupt": 0, "small": 0, "duplicate": 0, "unknown_class": 0}

    for cls in CLASSES:
        src_dir = raw / cls
        if not src_dir.exists():
            print(f"[警告] 未找到 {src_dir}，跳过该类")
            continue
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS or not f.is_file():
                continue
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if min(w, h) < MIN_SIDE:
                        removed["small"] += 1
                        continue
                    hsh = dhash(im)
            except Exception:  # noqa: BLE001
                removed["corrupt"] += 1
                continue
            # 去重（同类内 + 跨类）；阈值 5 视为近似重复
            is_dup = False
            for prev_hash in seen_hashes:
                if hamming(hsh, prev_hash) <= 5:
                    is_dup = True
                    break
            if is_dup:
                removed["duplicate"] += 1
                continue
            seen_hashes[hsh] = cls
            by_class[cls].append(f)

    total = sum(len(v) for v in by_class.values())
    if total == 0:
        print("[错误] 未读取到任何图片，请检查 --raw 目录结构（应含 6 个类别子目录）。")
        return
    print(f"[清洗] 有效 {total} 张；移除：损坏 {removed['corrupt']}、"
          f"过小 {removed['small']}、重复 {removed['duplicate']}")

    # ---------- 2. 分层划分 ----------
    manifest_rows = []
    samples_pool = []
    rng = random.Random(args.seed)
    for cls in CLASSES:
        files = sorted(by_class[cls])
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * SPLIT_RATIO["train"])
        n_val = int(n * SPLIT_RATIO["val"])
        split_of = {}
        for i, f in enumerate(files):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            split_of[f] = split

            # 复制到划分目录，统一文件名 <类别>_<序号>.jpg
            dst_dir = out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{cls}_{i:05d}.jpg"
            with Image.open(f) as im:
                im.convert("RGB").save(dst, "JPEG", quality=92)

            manifest_rows.append({
                "image_id": f"{cls}_{i:05d}",
                "relative_path": f"{split}/{cls}/{dst.name}",
                "class": cls,
                "split": split,
                "source": "TrashNet (github.com/garythung/trashnet)",
                "license": "见原仓库许可说明；仅用于课程学习",
                "sha256": sha256_file(dst),
                "note": "",
            })
            if split in ("val", "test"):
                samples_pool.append((cls, dst))

        print(f"  {cls}: 共 {n} → train {n_train}, val {n_val}, "
              f"test {n - n_train - n_val}")

    # ---------- 3. 写 manifest ----------
    manifest = out / "dataset_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "image_id", "relative_path", "class", "split",
            "source", "license", "sha256", "note",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[完成] manifest: {manifest} ({len(manifest_rows)} 行)")

    # ---------- 4. 抽样样例图（每类≥1，共10张） ----------
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng.shuffle(samples_pool)
    chosen = []
    per_class = defaultdict(int)
    # 先保证每类至少 1 张
    for item in samples_pool:
        cls, path = item
        if per_class[cls] == 0 and len(chosen) < len(CLASSES):
            chosen.append(item)
            per_class[cls] += 1
    # 再补足到 10 张
    for item in samples_pool:
        if len(chosen) >= 10:
            break
        if item in chosen:
            continue
        chosen.append(item)
    for cls, path in chosen:
        dst = samples_dir / f"{cls}_{path.stem.split('_')[-1]}.jpg"
        shutil.copy2(path, dst)
    print(f"[完成] 样例图 {len(chosen)} 张 → {samples_dir}")

    # ---------- 5. 清洗记录 ----------
    log = out / "clean_log.txt"
    log.write_text(
        "数据清洗与划分记录\n"
        f"随机种子: {args.seed}\n"
        f"划分比例: train {SPLIT_RATIO['train']} / val {SPLIT_RATIO['val']} / test {SPLIT_RATIO['test']}\n"
        f"有效图片: {total}\n"
        f"移除-损坏: {removed['corrupt']}\n"
        f"移除-过小: {removed['small']}\n"
        f"移除-重复(感知哈希汉明距离≤5): {removed['duplicate']}\n"
        "防泄漏措施: 先按感知哈希去重再分层划分；增强只作用于 train；同一哈希不跨集合。\n",
        encoding="utf-8",
    )
    print(f"[完成] 清洗记录: {log}")


if __name__ == "__main__":
    main()
