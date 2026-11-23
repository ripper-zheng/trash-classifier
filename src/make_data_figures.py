"""从真实 dataset_manifest.csv 生成报告第3章所需图表（课程作业，数据真实，禁止编造）。

生成：
  report/figures/fig2_class_distribution.png   六类总量柱状图
  report/figures/fig3_split_stacked.png         train/val/test 分层堆叠图
  report/figures/fig4_samples_grid.png          10张样例图拼图（标注类别+文件名）
  report/table3_dataset_split.md                划分统计表（Markdown，可直接粘进报告）

用法：python src/make_data_figures.py
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from plot_style import setup_chinese_font

setup_chinese_font()

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset" / "dataset_manifest.csv"
SAMPLES = ROOT / "dataset" / "samples"
FIGDIR = ROOT / "report" / "figures"
REPORT = ROOT / "report"
FIGDIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SPLITS = ["train", "val", "test"]
# 颜色仅用于区分类别，与 Stock 涨跌无关
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load_manifest():
    rows = []
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fig_distribution(counts):
    vals = [counts[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(CLASSES, vals, color=COLORS, edgecolor="#333", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("图片数量（张）", fontsize=11)
    ax.set_xlabel("类别", fontsize=11)
    ax.set_title("图2  数据集六类别分布（真实统计，n=%d）" % sum(vals), fontsize=12)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "fig2_class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_split_stacked(by_split):
    """堆积柱状图：每类 train/val/test 数量。"""
    import numpy as np
    train = [by_split[c]["train"] for c in CLASSES]
    val = [by_split[c]["val"] for c in CLASSES]
    test = [by_split[c]["test"] for c in CLASSES]
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    w = 0.55
    ax.bar(x, train, w, label="train", color="#4C72B0", edgecolor="#333", linewidth=0.5)
    ax.bar(x, val, w, bottom=train, label="val", color="#DD8452", edgecolor="#333", linewidth=0.5)
    bottoms = [t + v for t, v in zip(train, val)]
    ax.bar(x, test, w, bottom=bottoms, label="test", color="#55A868", edgecolor="#333", linewidth=0.5)
    for i, c in enumerate(CLASSES):
        total = train[i] + val[i] + test[i]
        ax.text(i, total + 6, str(total), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("图片数量（张）", fontsize=11)
    ax.set_title("图3  各类别 train/val/test 分层划分（70/15/15）", fontsize=12)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "fig3_split_stacked.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_samples():
    imgs = sorted([p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    n = len(imgs)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), dpi=150)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, imgs):
        im = Image.open(p).convert("RGB")
        ax.imshow(im)
        cls = p.name.split("_")[0]
        ax.set_title(f"{cls}\n{p.name}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"图4  数据集样例图（共 {n} 张，均来自真实数据集，标注类别与文件名）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGDIR / "fig4_samples_grid.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def markdown_table(counts, by_split):
    lines = [
        "| 类别 | 总数 | train | val | test |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        d = by_split[c]
        lines.append(f"| {c} | {counts[c]} | {d['train']} | {d['val']} | {d['test']} |")
    lines.append(
        f"| **合计** | **{sum(counts.values())}** | "
        f"**{sum(d['train'] for d in by_split.values())}** | "
        f"**{sum(d['val'] for d in by_split.values())}** | "
        f"**{sum(d['test'] for d in by_split.values())}** |"
    )
    out = REPORT / "table3_dataset_split.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    rows = load_manifest()
    counts = Counter(r["class"] for r in rows)
    by_split = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_split[r["class"]][r["split"]] += 1
    for c in CLASSES:
        for s in SPLITS:
            by_split[c][s]  # 确保键存在

    outputs = [
        fig_distribution(counts),
        fig_split_stacked(by_split),
        fig_samples(),
        markdown_table(counts, by_split),
    ]
    print("已生成（全部来自真实 manifest / samples，无编造）：")
    for o in outputs:
        print(f"  {o.relative_to(ROOT)}  ({o.stat().st_size} bytes)")
    print("\n划分统计表内容：")
    print((REPORT / "table3_dataset_split.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
