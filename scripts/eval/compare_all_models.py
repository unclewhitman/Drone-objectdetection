"""
compare_all_models.py
=====================
一站式模型对比脚本：
1. 在测试集上评估所有模型，提取核心指标
2. 生成指标对比表格（CSV + 控制台）+ 柱状图
3. 用测试图像做推理可视化对比（多模型并排）
"""

import csv
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ===== 配置 =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "VisDrone_Experiments", "model_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型定义：(显示名, 权重路径)
MODELS = {
    "Baseline (YOLOv11n)": os.path.join(
        PROJECT_ROOT, "VisDrone_Experiments/yolo11n_baseline/weights/best.pt"
    ),
    "P2+C3TR": os.path.join(
        PROJECT_ROOT, "VisDrone_Experiments/yolo11n_p2_C3TR/weights/best.pt"
    ),
    "Stage2-A (640)": os.path.join(
        PROJECT_ROOT, "VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_A/weights/best.pt"
    ),
    "Stage2-B (896)": os.path.join(
        PROJECT_ROOT, "VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_B2/weights/best.pt"
    ),
}

DATA_YAML = os.path.join(PROJECT_ROOT, "configs", "datasets", "visdrone_custom.yaml")
CONF_THRESH = 0.25
NUM_VIS_IMAGES = 100  # 可视化对比图数量


# ================================================================
# 第一部分：测试集评估 + 指标对比
# ================================================================
def evaluate_all_models():
    """在测试集上评估所有模型，返回 {model_name: metrics_dict}"""
    all_metrics = {}
    for name, weights in MODELS.items():
        if not os.path.isfile(weights):
            print(f"[跳过] 权重不存在: {weights}")
            continue
        print(f"\n{'='*60}")
        print(f"  评估模型: {name}")
        print(f"  权重: {weights}")
        print(f"{'='*60}")
        model = YOLO(weights)
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            verbose=False,
            project=OUTPUT_DIR,
            name=name.replace(" ", "_").replace("(", "").replace(")", ""),
            exist_ok=True,
        )
        # 提取核心指标
        m = {
            "Precision": float(metrics.box.mp),
            "Recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
        all_metrics[name] = m
        print(f"  → P={m['Precision']:.4f}  R={m['Recall']:.4f}  "
              f"mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}")
        # 释放显存
        del model
    return all_metrics


def save_metrics_csv(all_metrics, path):
    """保存指标对比表为 CSV"""
    metric_keys = ["Precision", "Recall", "mAP50", "mAP50-95"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + metric_keys)
        for name, m in all_metrics.items():
            writer.writerow([name] + [f"{m[k]:.4f}" for k in metric_keys])
    print(f"\n✅ 指标 CSV 已保存: {path}")


def print_metrics_table(all_metrics):
    """在控制台打印格式化指标表"""
    metric_keys = ["Precision", "Recall", "mAP50", "mAP50-95"]
    header = f"{'Model':<25}" + "".join(f"{k:>12}" for k in metric_keys)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, m in all_metrics.items():
        row = f"{name:<25}" + "".join(f"{m[k]:>12.4f}" for k in metric_keys)
        print(row)
    print(sep)


def plot_metrics_bar(all_metrics, path):
    """绘制指标柱状图"""
    metric_keys = ["Precision", "Recall", "mAP50", "mAP50-95"]
    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for i, name in enumerate(model_names):
        values = [all_metrics[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)])
        # 在柱子上标数值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_keys, fontsize=11)
    ax.set_ylim(0, max(max(m.values()) for m in all_metrics.values()) * 1.18)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"✅ 指标柱状图已保存: {path}")


# ================================================================
# 第二部分：推理可视化对比
# ================================================================
def get_test_images(n):
    """获取测试集前 n 张图像路径"""
    images_dir = os.path.join(
        PROJECT_ROOT, "datasets", "Visdrone", "VisDrone2019-DET-test-dev", "images"
    )
    if not os.path.isdir(images_dir):
        print(f"[警告] 测试图像目录不存在: {images_dir}")
        return []
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(exts))
    return [os.path.join(images_dir, f) for f in files[:n]]


def visualize_comparison(image_paths, save_dir):
    """对每张图像用所有模型推理，生成并排对比图"""
    os.makedirs(save_dir, exist_ok=True)

    # 先加载所有模型
    loaded = {}
    for name, weights in MODELS.items():
        if os.path.isfile(weights):
            loaded[name] = YOLO(weights)
    if not loaded:
        print("[错误] 没有可用的模型权重")
        return

    model_names = list(loaded.keys())
    n_models = len(model_names)

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  [{idx+1}/{len(image_paths)}] {os.path.basename(img_path)} ... ", end="")

        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 7))
        if n_models == 1:
            axes = [axes]

        for j, name in enumerate(model_names):
            model = loaded[name]
            results = model(img_path, conf=CONF_THRESH, verbose=False)
            plotted = results[0].plot()
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            n_dets = len(results[0].boxes)
            axes[j].imshow(plotted_rgb)
            axes[j].set_title(f"{name}\n({n_dets} detections)", fontsize=11, fontweight="bold")
            axes[j].axis("off")

        plt.suptitle(img_name, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"vis_compare_{img_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("done")

    # 释放显存
    del loaded
    print(f"\n✅ 可视化对比图已保存至: {save_dir}")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    # print("=" * 60)
    # print("  模型对比测试 — VisDrone 数据集")
    # print("=" * 60)

    # # ---------- 1. 测试集评估 ----------
    # print("\n" + "=" * 60)
    # print("  第一步：测试集评估")
    # print("=" * 60)
    # all_metrics = evaluate_all_models()

    # if all_metrics:
    #     # 打印表格
    #     print_metrics_table(all_metrics)

    #     # 保存 CSV
    #     csv_path = os.path.join(OUTPUT_DIR, "metrics_comparison.csv")
    #     save_metrics_csv(all_metrics, csv_path)
    
    #     # 绘制柱状图
    #     bar_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    #     plot_metrics_bar(all_metrics, bar_path)
    # else:
    #     print("[警告] 没有成功评估的模型，跳过指标对比。")

    # ---------- 2. 推理可视化 ----------
    print("\n" + "=" * 60)
    print(f"  第二步：推理可视化对比 ({NUM_VIS_IMAGES} 张)")
    print("=" * 60)
    test_images = get_test_images(NUM_VIS_IMAGES)
    if test_images:
        vis_dir = os.path.join(OUTPUT_DIR, "visual_comparisons")
        visualize_comparison(test_images, vis_dir)
    else:
        print("[警告] 未找到测试图像，跳过可视化对比。")

    print("\n" + "=" * 60)
    print("  全部完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)
