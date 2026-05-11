# Drone Object Detection / VisDrone YOLO11 实验

这个仓库整理了基于 VisDrone 数据集的 YOLO11 小目标检测实验，包括基线训练、P2+C3TR 结构改动、小目标增强、tile/sliced training 数据生成，以及 sliced inference 评估脚本。

## 目录结构

```text
configs/
  datasets/      数据集 YAML 配置
  models/        YOLO 模型结构 YAML
docs/
  plans/         实验设计和迭代方案
  reports/       实验结果报告
scripts/
  common/        通用几何、切片等工具函数
  data/          数据集转换、增强、tile 数据生成
  eval/          评估、对比、sliced inference
  infer/         推理流程
  train/         训练入口脚本
tests/           单元测试
```

## 本地数据和权重

下面这些内容体积大或属于个人材料，不进入 Git：

- `datasets/`、`dataset/`
- `VisDrone_Experiments/`
- `models/`
- `*.pt`、`*.pth`、`*.onnx`、`*.engine`

默认数据目录约定为：

```text
datasets/Visdrone/
  VisDrone2019-DET-train/
  VisDrone2019-DET-val/
  VisDrone2019-DET-test-dev/
```

## 常用命令

训练 baseline：

```powershell
& E:\Anaconda3\envs\yolov11\python.exe scripts\train\train_yolo11n.py
```

训练 P2+C3TR：

```powershell
& E:\Anaconda3\envs\yolov11\python.exe scripts\train\train_yolo11n_p2_C3TR.py
```

生成 tile768 混合训练集：

```powershell
& E:\Anaconda3\envs\yolov11\python.exe scripts\data\build_tile_dataset.py `
  --src datasets\Visdrone\VisDrone2019-DET-train `
  --dst datasets\Visdrone\VisDrone2019-DET-train-tile768-mix `
  --tile-size 768 `
  --overlap 0.20 `
  --empty-ratio 0.10 `
  --keep-original
```

sliced inference 评估：

```powershell
& E:\Anaconda3\envs\yolov11\python.exe scripts\eval\sliced_yolo_eval.py `
  --mode sliced `
  --weights VisDrone_Experiments\yolo11n_p2_C3TR_sobj_stage2_B2\weights\best.pt `
  --slice-size 1536 `
  --overlap 0.20 `
  --merge-iou 0.45 `
  --conf 0.001
```

详细实验结论见：

```text
docs/reports/visdrone-sliced-inference-and-tile-training-report.md
```
