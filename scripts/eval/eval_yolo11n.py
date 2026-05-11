from ultralytics import YOLO
import csv
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    # 1. 加载训练得到的最好权重
    # 如有需要，可将该路径改成绝对路径，避免工作目录不一致带来的问题
    best_model_path = os.path.join(
        "VisDrone_Experiments",
        "yolo11n_baseline",
        "weights",
        "best.pt",
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"未找到权重文件: {best_model_path}，请先运行 train_visdrone.py 完成训练。"
        )

    print(f"加载最优权重: {best_model_path}")
    best_model = YOLO(best_model_path)

    # 2. 在测试集上评估
    print("开始在测试集上评估...")
    metrics = best_model.val(
        data=os.path.join(PROJECT_ROOT, "configs", "datasets", "visdrone_custom.yaml"),
        split="test",  # 指定使用 test 集评估
        project="VisDrone_Experiments",
        name="yolo11n_test_eval",
    )
    print("测试集评估完成，主要指标如下：")
    print(metrics)

    # 2.1 将评估指标保存到文件中（包含 recall、mAP 等）
    # Ultralytics 通常会在 metrics.results_dict 中提供一个完整的字典
    results_dict = getattr(metrics, "results_dict", None)
    if results_dict is None:
        # 某些版本 metrics 本身就是 dict 或可转为 dict
        try:
            results_dict = dict(metrics)
        except Exception:
            results_dict = None

    # 评估输出目录（与 val 中的 project/name 保持一致）
    save_dir = getattr(metrics, "save_dir", os.path.join("VisDrone_Experiments", "yolo11n_test_eval"))
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir    , "metrics_summary.csv")

    if results_dict is not None:
        # 与训练 results.csv 类似的格式：表头一行，数值一行
        headers = list(results_dict.keys())
        values = []
        for k in headers:
            v = results_dict[k]
            try:
                values.append(f"{float(v):.6f}")
            except (TypeError, ValueError):
                values.append(str(v))

        with open(metrics_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(values)
        print(f"评估数值已保存到: {metrics_path}")
    else:
        print("当前 Ultralytics 版本未提供 results_dict，未能保存 CSV。")

    # 3. 可视化输出：挑选几张测试集图片进行预测并保存直观结果
    print("开始进行可视化推理...")
    test_images_dir = os.path.join(PROJECT_ROOT, "datasets", "Visdrone", "VisDrone2019-DET-test-dev", "images")

    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(
            f"测试图片目录不存在: {test_images_dir}，请检查 visdrone_custom.yaml 中的路径配置。"
        )

    # 随便挑3张图片做可视化演示
    sample_files = os.listdir(test_images_dir)[:3]
    sample_images = [os.path.join(test_images_dir, f) for f in sample_files]

    best_model.predict(
        source=sample_images,
        save=True,  # 将画好框的图片保存下来
        project="VisDrone_Experiments",
        name="yolo11n_visual_preds",
        conf=0.25,  # 置信度阈值：只显示置信度大于 0.25 的预测框
    )
    print("可视化推理完成！")


if __name__ == "__main__":
    main()
