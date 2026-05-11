# train_improved.py
from ultralytics import YOLO

def main():
    # 1. 核心技巧：加载自定义的 yaml 结构，但同时载入官方的预训练权重
    # YOLO 会自动匹配结构相同的层（如 Backbone 的前几层）并加载权重，
    # 新增加的 P2 和 Transformer 层会自动随机初始化。这能大幅加速收敛！
    model = YOLO("configs/models/yolo11n_p2_C3TR.yaml").load("yolo11n.pt")

    print("开始训练改进版模型 (P2 Head + C3TR)...")
    
    # 2. 启动训练
    results = model.train(
        data="configs/datasets/visdrone_custom.yaml",
        epochs=150,
        patience=30,                 # 结构变复杂了，多给一点耐心等待收敛
        batch=4,                     # 注意：加了 P2 头显存占用会变大，此处现存不够，改为 4
        imgsz=640,
        workers=2,
        device="0", 
        project="VisDrone_Experiments",
        name="yolo11n_p2_C3TR", # 保存到新的独立文件夹中
        save=True,
        plots=True,
        optimizer="AdamW",           # 加入 Transformer 后，AdamW 优化器通常比默认的 SGD 表现更好
        lr0=0.001                    # 适当调低初始学习率，保护随机初始化的新层
    )
    print("改进版模型训练完成！")

if __name__ == '__main__':
    main()
