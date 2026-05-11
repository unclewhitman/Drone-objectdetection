# yolo11n_p2_C3TR_sobj_stage2 设计文档

日期：2026-03-10
负责人：用户 + 助手

## 概述
在有限算力（笔记本 RTX 3070 8GB）条件下，提升 VisDrone 小目标召回与 mAP50-95。方法为“小目标数据强化 + 两阶段训练”。实验名为 yolo11n_p2_C3TR_sobj_stage2。

## 目标
- 提升小目标召回。
- 提升整体 mAP50-95。
- 训练成本可控，适配 8GB 显存。

## 非目标
- 不改变现有 P2 + C3TR 的模型结构路线。
- 不采用 1280+ 这种高分辨率全量训练。

## 指标
主要指标：
- 小目标 recall。
- 小目标 mAP50-95。

次要指标：
- 整体 recall。
- 整体 mAP50-95。

小目标定义（可配置）：
- bbox 面积 < 0.01 × 图像面积，或 bbox 边长 < 32×32 像素。

## 约束
- 笔记本 RTX 3070 8GB。
- 训练时间可接受。
- 避免数据增强造成背景污染。

## 总体方案
1. 离线小目标增强数据集（目标感知裁剪 + 受限 Copy-Paste + 小目标过采样）。
2. 两阶段训练：
   - Stage A：imgsz=640，强增广。
   - Stage B：imgsz=896/960，低增广精修。
3. 评估复用现有 baseline / improved 权重，保证可比性。

## 数据管线改动
新增离线增强脚本，生成训练集：
- 输入：datasets/Visdrone/VisDrone2019-DET-train/images + labels。
- 输出：datasets/Visdrone/VisDrone2019-DET-train-aug/images + labels。
- 新数据配置：visdrone_smallobj.yaml，train 指向 -train-aug，val/test 不变。

## 增强规则
目标感知裁剪：
- 以小目标为中心裁剪，保证目标完整保留。

受限 Copy-Paste：
- 只在同一张图内部 Copy-Paste。
- 粘贴位置限制在目标附近局部范围。
- 与已有目标 IoU < 0.1。
- 粘贴目标必须完整落在图内。
- 每张图粘贴次数有限（如 <= 2）。
- 可选更严格规则：避免天空区域或用纹理相似度筛选。

小目标过采样：
- 对包含小目标的图像进行过采样，提升出现频率。

## 两阶段训练
Stage A（鲁棒性）：
- imgsz=640
- 强增广（mosaic / scale / crop）
- epochs：120~160

Stage B（精修）：
- imgsz=896 或 960（取显存可承受的最高值）
- 低增广（close_mosaic，降低 mixup）
- epochs：30~60
- 从 Stage A best 权重继续训练

## 评估计划
- 复用现有 baseline / improved 权重进行对比评估。
- 新模型在同一测试集上评估。
- 输出整体指标和小目标指标 CSV 到 comparison_results。

## 风险与控制
- 背景污染：同图 + 局部 + IoU 约束。
- 过拟合小目标：保留部分原始样本混合训练。
- 显存压力：Stage B 降低 batch，优先 imgsz=896。

## 交付物
- 离线增强脚本。
- 新数据集 YAML（visdrone_smallobj.yaml）。
- 两阶段训练脚本（yolo11n_p2_C3TR_sobj_stage2）。
- 小目标评估脚本与 CSV 输出。
