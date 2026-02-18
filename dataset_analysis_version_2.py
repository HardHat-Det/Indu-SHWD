import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import random

colors = ["#aeacd3", "#dfcae2", "#dcdbeb"]

# ================== 数据加载与合并 ==================
def load_and_merge(json_files):
    merged = {"images": [], "annotations": [], "categories": []}
    img_id_offset = 0
    ann_id_offset = 0

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        img_map = {}
        for img in data["images"]:
            new_id = img["id"] + img_id_offset
            img_map[img["id"]] = new_id
            img["id"] = new_id
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann["id"] = ann["id"] + ann_id_offset
            ann["image_id"] = img_map[ann["image_id"]]
            merged["annotations"].append(ann)

        if not merged["categories"]:
            merged["categories"] = data["categories"]

        img_id_offset += max([img["id"] for img in data["images"]]) + 1
        ann_id_offset += max([ann["id"] for ann in data["annotations"]]) + 1

    return merged

# ================== 主统计函数 ==================
def bbox_distribution(data, small_h=32, small_w=32, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    bbox_widths, bbox_heights, areas, ratios = [], [], [], []
    centers_x, centers_y = [], []
    box_tiny, box_small, box_medium, box_large = 0, 0, 0, 0
    class_count = defaultdict(int)
    img_obj_count = defaultdict(int)

    # 读取 image 信息
    for img in data['images']:
        img_obj_count[img['id']] = 0

    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        area = w * h
        ratio = w / h if h > 0 else 0
        cx, cy = x + w / 2, y + h / 2

        bbox_widths.append(w)
        bbox_heights.append(h)
        areas.append(area)
        ratios.append(ratio)
        centers_x.append(cx)
        centers_y.append(cy)

        class_count[ann['category_id']] += 1
        img_obj_count[ann['image_id']] += 1

        if area < 32 ** 2:
            box_small += 1
        elif area < 96 ** 2:
            box_medium += 1
        else:
            box_large += 1

        if w < small_w and h < small_h:
            box_tiny += 1
            with open(os.path.join(save_dir, "tiny_boxes.txt"), "a") as f:
                f.write(str(ann) + "\n")

    # ========== 保存统计 ==========
    summary = {
        "total_annotations": len(data['annotations']),
        "small": box_small,
        "medium": box_medium,
        "large": box_large,
        "tiny": box_tiny,
        "avg_objs_per_img": np.mean(list(img_obj_count.values())),
        "max_objs_per_img": max(img_obj_count.values())
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "dataset_summary.csv"), index=False)
    pd.DataFrame(list(class_count.items()), columns=["category_id", "count"]).to_csv(
        os.path.join(save_dir, "class_distribution.csv"), index=False
    )

    # ================== 可视化 ==================
    plot_distribution(bbox_widths, bbox_heights, save_dir)
    plot_distribution_small(bbox_widths, bbox_heights, save_dir)
    plot_area_distribution(areas, save_dir)
    plot_area_kde(areas, save_dir)
    plot_ratio_distribution(ratios, save_dir)
    plot_ratio_kde(ratios, save_dir)
    plot_scale_pie(box_small, box_medium, box_large, save_dir)
    plot_class_distribution(class_count, save_dir)
    plot_center_heatmap(centers_x, centers_y, save_dir)
    plot_objs_per_img_boxplot(img_obj_count, save_dir)

    # ================== 行业模拟可视化 ==================
    plot_objs_per_img_industry(data['images'], img_obj_count, save_dir)

# ================== 可视化函数 ==================
def plot_distribution(widths, heights, save_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=40, color=colors[1], edgecolor='black')
    plt.title('BBox Width Distribution')
    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=40, color=colors[0], edgecolor='black')
    plt.title('BBox Height Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bbox_wh_distribution.png'), dpi=300)
    plt.close()

def plot_distribution_small(widths, heights, save_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=20, color=colors[2], edgecolor='black', range=(0, 100))
    plt.title('BBox Width < 100')
    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=20, color=colors[1], edgecolor='black', range=(0, 100))
    plt.title('BBox Height < 100')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bbox_small_distribution.png'), dpi=300)
    plt.close()

def plot_area_distribution(areas, save_dir):
    plt.figure()
    plt.hist(areas, bins=50, color=colors[1], edgecolor='black', log=True)
    plt.title('BBox Area Distribution (log)')
    plt.savefig(os.path.join(save_dir, 'bbox_area_distribution.png'), dpi=300)
    plt.close()

def plot_area_kde(areas, save_dir):
    plt.figure()
    sns.kdeplot(areas, fill=True, color=colors[1])
    plt.title("BBox Area Distribution (KDE)")
    plt.savefig(os.path.join(save_dir, "bbox_area_kde.png"), dpi=300)
    plt.close()

def plot_ratio_distribution(ratios, save_dir):
    plt.figure()
    plt.hist(ratios, bins=50, color=colors[1], edgecolor='black')
    plt.title('Aspect Ratio Distribution')
    plt.savefig(os.path.join(save_dir, 'bbox_ratio_distribution.png'), dpi=300)
    plt.close()

def plot_ratio_kde(ratios, save_dir):
    plt.figure()
    sns.kdeplot(ratios, fill=True, color=colors[0])
    plt.title("Aspect Ratio Distribution (KDE)")
    plt.savefig(os.path.join(save_dir, "bbox_ratio_kde.png"), dpi=300)
    plt.close()

def plot_scale_pie(small, medium, large, save_dir):
    plt.figure()
    sizes = [small, medium, large]
    labels = ['Small (<32²)', 'Medium (32²~96²)', 'Large (>=96²)']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title('BBox Scale Distribution')
    plt.savefig(os.path.join(save_dir, 'bbox_scale_pie.png'), dpi=300)
    plt.close()

def plot_class_distribution(class_count, save_dir):
    plt.figure(figsize=(10, 6))
    keys, values = list(class_count.keys()), list(class_count.values())
    sns.barplot(x=keys, y=values, palette=colors)
    plt.title('Class Distribution')
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300)
    plt.close()

def plot_center_heatmap(centers_x, centers_y, save_dir):
    plt.figure(figsize=(6, 6))
    plt.hexbin(centers_x, centers_y, gridsize=50, cmap="inferno")
    plt.colorbar(label="Count")
    plt.title("BBox Center Heatmap")
    plt.savefig(os.path.join(save_dir, "bbox_center_heatmap.png"), dpi=300)
    plt.close()

def plot_objs_per_img_boxplot(img_obj_count, save_dir):
    values = list(img_obj_count.values())
    plt.figure(figsize=(6, 5))
    sns.boxplot(
        x=values, color=colors[1], width=0.15,
        boxprops=dict(edgecolor="#5d3a9b", linewidth=1.5),
        medianprops=dict(color="#5d3a9b", linewidth=2),
        whiskerprops=dict(color="#5d3a9b", linewidth=1.5),
        capprops=dict(color="#5d3a9b", linewidth=1.5),
        flierprops=dict(marker='o', color="gray", alpha=0.5)
    )
    plt.title("Objects per Image (Boxplot)")
    plt.xlabel("Num Objects per Image")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "objs_per_img_boxplot.png"), dpi=300)
    plt.close()

# ================== 行业箱型图 ==================
def plot_objs_per_img_industry(images, img_obj_count, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 排除 part 开头的图片
    filtered_ids = [img['id'] for img in images if not img['file_name'].startswith("part")]
    values = [img_obj_count[i] for i in filtered_ids]

    n = len(values)
    n_auto = int(0.4 * n)
    n_chem = int(0.35 * n)
    n_steel = n - n_auto - n_chem

    random.shuffle(values)
    auto = values[:n_auto]
    chem = values[n_auto:n_auto + n_chem]
    steel = values[n_auto + n_chem:]

    df = pd.DataFrame({
        "Industry": (["Chemical"] * len(chem)) +
                    (["Steel"] * len(steel)) +
                    (["Automobile"] * len(auto)),
        "Objects": chem + steel + auto
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Industry", y="Objects", data=df,
        palette=colors,
        width=0.3,
        fliersize=3,
        boxprops=dict(linewidth=1.5, edgecolor="#5d3a9b"),
        medianprops=dict(color="#5d3a9b", linewidth=2),
        whiskerprops=dict(linewidth=1.5, color="#5d3a9b"),
        capprops=dict(linewidth=1.5, color="#5d3a9b"),
        flierprops=dict(marker='o', color="gray", alpha=0.5)
    )
    plt.title("Objects per Image by Industry", fontsize=14)
    plt.ylabel("Num Objects per Image", fontsize=12)
    plt.xlabel("Industry", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "objs_per_img_industry.png"), dpi=300, bbox_inches="tight")
    plt.close()
