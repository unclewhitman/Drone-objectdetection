from ultralytics import YOLO


def stage_a():
    # 原始全新训练方式（保留参考）
    # model = YOLO("configs/models/yolo11n_p2_C3TR.yaml").load("yolo11n.pt")
    # return model.train(
    #     data="configs/datasets/visdrone_smallobj.yaml",
    #     epochs=150,
    #     imgsz=640,
    #     batch=4,
    #     optimizer="AdamW",
    #     lr0=0.001,
    #     project="VisDrone_Experiments",
    #     name="yolo11n_p2_C3TR_sobj_stage2_A",
    #     save=True,
    #     plots=True,
    # )

    # 断点续训：从 last.pt 继续
    model = YOLO("VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_A/weights/last.pt")
    return model.train(resume=True)


def stage_b():
    # 原始 Stage B（保留参考）
    # model = YOLO("VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_A/weights/best.pt")
    # return model.train(
    #     data="configs/datasets/visdrone_smallobj.yaml",
    #     epochs=40,
    #     imgsz=896,
    #     batch=2,
    #     close_mosaic=10,
    #     mixup=0.05,
    #     project="VisDrone_Experiments",
    #     name="yolo11n_p2_C3TR_sobj_stage2_B",
    #     save=True,
    #     plots=True,
    # )

    # 断点续训：从 last.pt 继续
    model = YOLO("VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_B2/weights/last.pt")
    return model.train(resume=True)


if __name__ == "__main__":
    # Stage A 断点续训
    # stage_a()
    # Stage B 断点续训
    stage_b()
