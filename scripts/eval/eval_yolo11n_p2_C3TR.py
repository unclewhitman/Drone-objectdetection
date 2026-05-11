# compare_models.py
import csv
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

# 以仓库根目录解析路径，保证无论从哪运行都能找到数据
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compare_inference(image_path, baseline_weights, improved_weights):
    print(f"正在处理图像: {image_path}")

    # ==========================================
    # 1. 基线模型 (YOLOv11n) - 常规推理
    # ==========================================
    print("运行基线模型...")
    baseline_model = YOLO(baseline_weights)
    baseline_results = baseline_model(image_path, conf=0.5, verbose=False)
    baseline_img_bgr = baseline_results[0].plot()
    baseline_img_rgb = cv2.cvtColor(baseline_img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # 2. 改进模型 (P2+Trans) + SAHI 切片推理
    # ==========================================
    print("运行改进模型 + SAHI 切片推理...")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=improved_weights,
        confidence_threshold=0.5,
        device="cuda:0"  # 没有 GPU 改为 "cpu"
    )

    sahi_result = get_sliced_prediction(
        image_path,
        sahi_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
        postprocess_type="NMM"
    )

    sahi_result.export_visuals(export_dir="temp_sahi", file_name="temp_output")
    sahi_img_bgr = cv2.imread("temp_sahi/temp_output.png")
    sahi_img_rgb = cv2.cvtColor(sahi_img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # 3. Matplotlib 并排显示对比结果
    # ==========================================
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(baseline_img_rgb)
    plt.title("Baseline (YOLOv11n Original)", fontsize=16)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sahi_img_rgb)
    plt.title("Improved (P2+C3TR) + SAHI", fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(PROJECT_ROOT, "VisDrone_Experiments", "comparison_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"comparison_{img_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"对比图已保存至: {save_path}")
    return save_path


def _resolve_test_images(n=5):
    """从测试集目录取前 n 张图像路径。"""
    rel = "datasets/Visdrone/VisDrone2019-DET-test-dev/images"
    images_dir = os.path.join(PROJECT_ROOT, rel)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"测试图像目录不存在: {images_dir}")
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(exts)]
    if not files:
        raise FileNotFoundError(f"目录 {images_dir} 下没有图像文件。")
    selected = [os.path.join(images_dir, f) for f in files[:n]]
    return selected


def _save_metrics_csv(metrics, save_path):
    """将 val 返回的 metrics 保存为 CSV。"""
    results_dict = getattr(metrics, "results_dict", None)
    if results_dict is None:
        try:
            results_dict = dict(metrics)
        except Exception:
            return False
    headers = list(results_dict.keys())
    values = []
    for k in headers:
        v = results_dict[k]
        try:
            values.append(f"{float(v):.6f}")
        except (TypeError, ValueError):
            values.append(str(v))
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(values)
    return True


if __name__ == '__main__':
    NUM_IMAGES = 100

    BASELINE_WEIGHTS = os.path.join(PROJECT_ROOT, "VisDrone_Experiments/yolo11n_baseline/weights/best.pt")
    IMPROVED_WEIGHTS = os.path.join(PROJECT_ROOT, "VisDrone_Experiments/yolo11n_p2_C3TR/weights/best.pt")
    STAGE2_WEIGHTS = os.path.join(
        PROJECT_ROOT,
        "VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_B/weights/best.pt"
    )

    save_dir = os.path.join(PROJECT_ROOT, "VisDrone_Experiments", "comparison_results")
    os.makedirs(save_dir, exist_ok=True)

    # ========== 1. 测试集评估，结果存 CSV ==========
    print("=" * 50)
    print("1. 测试集评估 (recall、mAP 等)，并保存 CSV")
    print("=" * 50)

    print("基线模型 (YOLOv11n) 评估中...")
    baseline_model = YOLO(BASELINE_WEIGHTS)
    baseline_metrics = baseline_model.val(
        data=os.path.join(PROJECT_ROOT, "configs", "datasets", "visdrone_custom.yaml"),
        split="test",
        verbose=False,
    )
    baseline_csv = os.path.join(save_dir, "metrics_baseline.csv")
    if _save_metrics_csv(baseline_metrics, baseline_csv):
        print(f"  已保存 {baseline_csv}")

    print("改进模型 (P2+C3TR) 评估中...")
    improved_model = YOLO(IMPROVED_WEIGHTS)
    improved_metrics = improved_model.val(
        data=os.path.join(PROJECT_ROOT, "configs", "datasets", "visdrone_custom.yaml"),
        split="test",
        verbose=False,
    )
    improved_csv = os.path.join(save_dir, "metrics_improved.csv")
    if _save_metrics_csv(improved_metrics, improved_csv):
        print(f"  已保存 {improved_csv}")

    if os.path.isfile(STAGE2_WEIGHTS):
        print("新方法 (sobj_stage2) 评估中...")
        stage2_model = YOLO(STAGE2_WEIGHTS)
        stage2_metrics = stage2_model.val(
            data=os.path.join(PROJECT_ROOT, "configs", "datasets", "visdrone_custom.yaml"),
            split="test",
            verbose=False,
        )
        stage2_csv = os.path.join(save_dir, "metrics_sobj_stage2.csv")
        if _save_metrics_csv(stage2_metrics, stage2_csv):
            print(f"  已保存 {stage2_csv}")

    # ========== 2. 可视化对比 ==========
    print("\n" + "=" * 50)
    print(f"2. 可视化对比 (共 {NUM_IMAGES} 张)")
    print("=" * 50)

    TEST_IMAGES = _resolve_test_images(NUM_IMAGES)
    print(f"图像列表: {[os.path.basename(p) for p in TEST_IMAGES[:5]]}{'...' if len(TEST_IMAGES) > 5 else ''}")

    for i, img_path in enumerate(TEST_IMAGES):
        print(f"\n[{i+1}/{len(TEST_IMAGES)}] ", end="")
        compare_inference(img_path, BASELINE_WEIGHTS, IMPROVED_WEIGHTS)

    print("\n全部完成。")
    print(f"  - 测试指标 CSV: {save_dir}/metrics_baseline.csv, metrics_improved.csv, metrics_sobj_stage2.csv")
    print(f"  - 对比图: {save_dir}/comparison_*.png")
