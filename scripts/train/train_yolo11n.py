from pathlib import Path
import sys

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.paths import resolve_pretrained


def main():
    # 1. 加载最小参数量的 YOLOv11 Nano 模型
    # 如果本地没有 yolo11n.pt，它会自动从官方下载
    model = YOLO(resolve_pretrained("yolo11n.pt"))

    # 2. 模型训练 (加入了资源优化参数)
    print("开始训练...")
    results = model.train(
        data="configs/datasets/visdrone_custom.yaml",  # 指定刚才写的数据集配置
        epochs=150,                   # 最大训练轮数
        patience=20,                  # ★ 早停机制：如果验证集指标 20 轮不下降，自动停止训练，节省时间
        batch=8,                      # ★ 批次大小：根据显存调整。如果显存极小，可改为 4；如果还可以，改为 16
        imgsz=640,                    # 输入图像尺寸：VisDrone 原图很大，640 是个折中的选择，降低显存压力
        workers=2,                    # ★ 数据加载线程数：降低此值可以大幅减少 CPU 和 内存 的占用 (默认为8，低配建议2或4)
        device="0",                   # 使用 GPU (如果没有 GPU，填 "cpu")
        cache=False,                  # ★ 关键：不要将数据集缓存到内存中，避免内存溢出
        project="VisDrone_Experiments",  # 实验主目录：为了后续做对比实验，统一保存在这里
        name="yolo11n_baseline",      # 本次实验的名称，结果会保存在 VisDrone_Experiments/yolo11n_baseline 下
        save=True,                    # 保存模型权重
        plots=True                    # ★ 自动生成训练过程的各类可视化图表 (Loss曲线, PR曲线, 混淆矩阵等)
    )

    print("训练完成，最优权重已保存到 VisDrone_Experiments/yolo11n_baseline/weights/best.pt")


if __name__ == "__main__":
    # 在 Windows 下运行多进程 DataLoader 需要放到 __main__ 保护里
    main()
