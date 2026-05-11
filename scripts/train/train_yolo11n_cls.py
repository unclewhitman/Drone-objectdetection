from ultralytics import YOLO

def main():
    # 1. Load YOLOv11 nano classification model
    model = YOLO("yolo11n-cls.pt")

    # 2. Train model
    print("开始训练 YOLO11 分类模型...")
    results = model.train(
        data="datasets/Visdrone_Cls", 
        epochs=30,                     
        patience=10,                  
        batch=32,                     
        imgsz=64,                    
        workers=2,                   
        device="0",                  
        project="VisDrone_Experiments",  
        name="yolo11n_cls_baseline",     
        save=True,                   
        plots=True                   
    )

    print("分类器训练完成，最优权重已保存到 VisDrone_Experiments/yolo11n_cls_baseline/weights/best.pt")

if __name__ == "__main__":
    main()
