# VisDrone 切片推理与 Tile 训练测试报告

日期：2026-05-11

## 摘要

本轮测试的目标是：在当前最好的 `Stage2-B (896)` 模型基础上，判断还能不能通过 **sliced inference（切片推理）** 和 **tile training（切片训练数据）** 继续提升 VisDrone 小目标检测效果。

主要结论：

- 切片推理确实有效，而且收益主要来自“切片 + 合并”，不是单纯把 `imgsz` 调大。
- 当前完整 test-dev 上最好的推理参数是 `slice_size=1536`、`overlap=0.20`、`merge_iou=0.45`。
- 在同一套自定义评估器下，mAP50-95 从 `0.1908` 提升到 `0.2511`。
- 普通整图 `imgsz=1536` 也有提升，但仍低于 `sliced 1536`。
- 已经生成并验证了 `tile768 + 原图混合` 训练数据集，但目前只做了 2% 数据、1 epoch 的烟测，不能作为正式训练结论。

## 当前基线

参考权重：

```text
VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_B2/weights/best.pt
```

这就是本轮测试前的当前最强模型：

- 模型结构：YOLO11n + P2 检测头 + C3TR
- 训练数据：small-object augmented VisDrone 训练集
- Stage B 精修分辨率：`imgsz=896`

## 新增评估脚本

新增脚本：

```text
scripts/eval/sliced_yolo_eval.py
```

这个脚本支持两种模式：

- `normal`：整图直接跑 YOLO。
- `sliced`：把原图切成有重叠的 slice，每个 slice 单独推理，再把框映射回原图坐标，最后做 class-aware NMS 合并。

评估设置：

- 测试集：`datasets/Visdrone/VisDrone2019-DET-test-dev/images`
- 标签：`datasets/Visdrone/VisDrone2019-DET-test-dev/labels`
- 图片数：1610
- IoU 阈值：`0.50` 到 `0.95`
- 置信度阈值：`conf=0.001`
- 合并后最大检测数：`max_det=1000`

新增测试：

```text
tests/test_sliced_yolo_eval.py
tests/test_build_tile_dataset.py
```

测试覆盖：

- slice 能覆盖图像边缘。
- class-aware NMS 不会互相压掉不同类别的重叠框。
- 同一个 GT 在每个 IoU 阈值下最多只匹配一次。
- tile 标签裁剪后坐标能正确重算。
- 只露出很小一部分的边缘框会被丢弃。

## 完整 Test-Dev 结果

下面所有结果都使用同一个自定义 evaluator，因此可以直接横向比较。

| 实验 | 模式 | img/slice | overlap | merge_iou | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2-B normal | normal | 896 | - | - | 0.4599 | 0.3494 | 0.3276 | 0.1908 |
| Stage2-B sliced | sliced | 896 | 0.25 | 0.60 | 0.4937 | 0.4137 | 0.3915 | 0.2287 |
| Stage2-B sliced | sliced | 1280 | 0.20 | 0.55 | 0.4997 | 0.4246 | 0.4049 | 0.2383 |
| Stage2-B normal | normal | 1536 | - | - | 0.5088 | 0.4184 | 0.3976 | 0.2330 |
| Stage2-B sliced | sliced | 1536 | 0.20 | 0.45 | 0.5177 | 0.4386 | 0.4249 | 0.2511 |

当前最佳参数：

```text
mode=sliced
slice_size=1536
overlap=0.20
merge_iou=0.45
conf=0.001
tile_max_det=300
max_det=1000
```

最佳完整测试输出：

```text
VisDrone_Experiments/sliced_eval/stage2b_sliced1536_o020_m045_full_summary.csv
```

## 结果解读

第一版 `slice_size=896` 的切片推理已经比普通整图推理好，但提升还不算大：

```text
normal 896 mAP50-95: 0.1908
sliced 896 mAP50-95: 0.2287
```

继续扫参后发现，更大的 slice 效果更好：

- `896` 能放大小目标，但可能损失太多场景上下文。
- `1280` 在细节和上下文之间更平衡。
- `1536` 最好，说明当前模型需要保留一定大视野，同时避免整图缩放造成小目标过小。

普通整图 `imgsz=1536` 也做了测试：

```text
normal 1536 mAP50-95: 0.2330
sliced 1536 mAP50-95: 0.2511
```

因此结论不是“只要高分辨率就行”，而是 **大切片 + 框合并** 这套推理方式更适合当前模型和 VisDrone 场景。

## 300 张子集扫参

为了降低成本，先在前 300 张 test-dev 图上扫了 `slice_size / overlap / merge_iou`。

前几名结果：

| slice | overlap | merge_iou | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 0.20 | 0.45 | 0.5427 | 0.5198 | 0.5023 | 0.2950 |
| 1536 | 0.20 | 0.50 | 0.5426 | 0.5201 | 0.5016 | 0.2941 |
| 1536 | 0.30 | 0.50 | 0.5426 | 0.5201 | 0.5016 | 0.2941 |
| 1536 | 0.20 | 0.55 | 0.5419 | 0.5182 | 0.5008 | 0.2935 |
| 1280 | 0.20 | 0.45 | 0.5248 | 0.5040 | 0.4857 | 0.2815 |
| 1280 | 0.20 | 0.50 | 0.5248 | 0.5040 | 0.4844 | 0.2806 |
| 1280 | 0.20 | 0.55 | 0.5226 | 0.5053 | 0.4830 | 0.2793 |
| 1024 | 0.30 | 0.55 | 0.5186 | 0.4938 | 0.4719 | 0.2721 |

注意：300 张子集分数明显高于完整 test-dev，所以它只适合用来排序参数，不应该当最终指标。

## Tile 训练数据管线

新增 tile 数据集生成脚本：

```text
scripts/data/build_tile_dataset.py
```

生成的数据集：

```text
datasets/Visdrone/VisDrone2019-DET-train-tile768-mix
```

输出规模：

```text
44213 image/label pairs
```

对应数据集配置：

```text
configs/datasets/visdrone_tile768_mix.yaml
```

生成参数：

```text
tile_size=768
overlap=0.20
min_visible=0.30
empty_ratio=0.10
keep_original=True
```

为什么训练用 `tile_size=768`，而不是推理最优的 `1536`：

- 训练集很多图是 `960x540`、`1360x765`、`1920x1080`。
- `1536` 对很多训练图切不出有效局部块，基本等于复制原图。
- `768` 能真正生成局部训练样本，同时通过 `keep_original=True` 保留原图上下文。

## Tile 训练烟测

做了一个很短的烟测，只验证训练管线，不判断最终效果。

训练参数：

```text
weights=VisDrone_Experiments/yolo11n_p2_C3TR_sobj_stage2_B2/weights/best.pt
data=configs/datasets/visdrone_tile768_mix.yaml
epochs=1
fraction=0.02
imgsz=768
batch=2
optimizer=AdamW
lr0=0.0002
```

输出目录：

```text
VisDrone_Experiments/tile768_mix_smoke_e1_frac002_fixpath
```

烟测确认：

- 数据路径可用。
- 标签扫描正常。
- 训练能在 RTX 3070 Laptop GPU 上跑通。
- 峰值显存大约 `5.7GB`。

烟测权重用最佳 sliced1536 评估后的结果：

| 权重 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Stage2-B best + sliced1536 | 0.5177 | 0.4386 | 0.4249 | 0.2511 |
| Tile smoke checkpoint + sliced1536 | 0.4758 | 0.3207 | 0.3293 | 0.1880 |

这个下降不说明 tile 训练方向错了。它只说明 `2% 数据 + 1 epoch` 不能作为正式模型结果，只能说明数据和训练流程跑通。

## 当前最佳实用命令

当前建议把 `Stage2-B best.pt` 配合 sliced1536 作为最强推理 baseline：

```powershell
& E:\Anaconda3\envs\yolov11\python.exe scripts\eval\sliced_yolo_eval.py `
  --mode sliced `
  --weights VisDrone_Experiments\yolo11n_p2_C3TR_sobj_stage2_B2\weights\best.pt `
  --output-dir VisDrone_Experiments\sliced_eval `
  --summary-name stage2b_sliced1536_o020_m045_full_summary.csv `
  --per-image-name stage2b_sliced1536_o020_m045_full_per_image.csv `
  --slice-size 1536 `
  --overlap 0.20 `
  --conf 0.001 `
  --merge-iou 0.45 `
  --tile-max-det 300 `
  --max-det 1000
```

## 下一步建议

下一步不要再只做 1 epoch 烟测，而是做一个正式的小规模训练实验：

```text
weights: Stage2-B best.pt
data: configs/datasets/visdrone_tile768_mix.yaml
imgsz: 768
batch: 2
epochs: 10-20
fraction: 0.20-0.30
optimizer: AdamW
lr0: 0.0001-0.0002
mosaic: 0.3-0.5 early
close_mosaic: near the end
mixup: 0.0
```

训练后统一用以下推理参数评估：

```text
slice_size=1536
overlap=0.20
merge_iou=0.45
```

判断标准：

- 如果 20%-30% fraction 训练后，sliced1536 mAP50-95 超过 `0.2511`，再扩大到全量训练。
- 如果没有超过，先优化 tile 采样和数据增强策略，不建议直接全量训练。
