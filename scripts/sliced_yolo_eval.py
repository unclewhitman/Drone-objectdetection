import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class


CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def generate_slices(width, height, slice_size=896, overlap=0.25):
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(round(slice_size * (1.0 - overlap))))
    xs = list(range(0, max(width - slice_size, 0) + 1, step))
    ys = list(range(0, max(height - slice_size, 0) + 1, step))
    if not xs or xs[-1] != max(width - slice_size, 0):
        xs.append(max(width - slice_size, 0))
    if not ys or ys[-1] != max(height - slice_size, 0):
        ys.append(max(height - slice_size, 0))

    boxes = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + slice_size, width)
            y2 = min(y1 + slice_size, height)
            boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-16)


def nms_numpy(boxes, scores, iou_thresh=0.6):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break
        ious = box_iou(boxes[idx], boxes[order[1:]])
        order = order[1:][ious <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def class_aware_nms(preds, iou_thresh=0.6, max_det=300):
    if preds.size == 0:
        return preds
    kept = []
    for cls_id in np.unique(preds[:, 5]).astype(int):
        cls_mask = preds[:, 5] == cls_id
        cls_preds = preds[cls_mask]
        keep_local = nms_numpy(cls_preds[:, :4], cls_preds[:, 4], iou_thresh)
        kept.append(cls_preds[keep_local])
    merged = np.concatenate(kept, axis=0) if kept else np.empty((0, 6), dtype=np.float32)
    if merged.size == 0:
        return merged
    order = np.argsort(-merged[:, 4])[:max_det]
    return merged[order]


def load_yolo_labels(label_path, width, height):
    labels = []
    if not label_path.is_file():
        return np.empty((0, 5), dtype=np.float32)
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = map(float, parts)
            x1 = (x - w / 2.0) * width
            y1 = (y - h / 2.0) * height
            x2 = (x + w / 2.0) * width
            y2 = (y + h / 2.0) * height
            labels.append((x1, y1, x2, y2, cls_id))
    return np.array(labels, dtype=np.float32)


def match_predictions(preds, targets, iouv):
    tp = np.zeros((len(preds), len(iouv)), dtype=bool)
    if len(preds) == 0:
        return tp
    if len(targets) == 0:
        return tp

    pred_order = np.argsort(-preds[:, 4])
    matched = {float(t): set() for t in iouv}
    for pred_idx in pred_order:
        pred = preds[pred_idx]
        same_cls = np.where(targets[:, 4] == pred[5])[0]
        if same_cls.size == 0:
            continue
        ious = box_iou(pred[:4], targets[same_cls, :4])
        if ious.size == 0:
            continue
        local_order = np.argsort(-ious)
        for t_idx, threshold in enumerate(iouv):
            for local_idx in local_order:
                target_idx = int(same_cls[local_idx])
                if target_idx in matched[float(threshold)]:
                    continue
                if ious[local_idx] >= threshold:
                    tp[pred_idx, t_idx] = True
                    matched[float(threshold)].add(target_idx)
                break
    return tp


def predict_sliced(model, image, slice_size, overlap, conf, tile_iou, merge_iou, tile_max_det, max_det, device):
    height, width = image.shape[:2]
    all_preds = []
    for x1, y1, x2, y2 in generate_slices(width, height, slice_size, overlap):
        tile = image[y1:y2, x1:x2]
        results = model.predict(
            tile,
            imgsz=slice_size,
            conf=conf,
            iou=tile_iou,
            max_det=tile_max_det,
            device=device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        xyxy[:, [0, 2]] += x1
        xyxy[:, [1, 3]] += y1
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
        confs = boxes.conf.cpu().numpy().astype(np.float32)[:, None]
        classes = boxes.cls.cpu().numpy().astype(np.float32)[:, None]
        all_preds.append(np.concatenate([xyxy, confs, classes], axis=1))

    if not all_preds:
        return np.empty((0, 6), dtype=np.float32)
    preds = np.concatenate(all_preds, axis=0)
    return class_aware_nms(preds, merge_iou, max_det)


def predict_normal(model, image, imgsz, conf, iou, max_det, device):
    height, width = image.shape[:2]
    results = model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    confs = boxes.conf.cpu().numpy().astype(np.float32)[:, None]
    classes = boxes.cls.cpu().numpy().astype(np.float32)[:, None]
    return np.concatenate([xyxy, confs, classes], axis=1)


def evaluate_sliced(args):
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    model = YOLO(args.weights)
    iouv = np.linspace(0.5, 0.95, 10)
    stats = []
    target_cls_all = []
    rows = []

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        targets = load_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height)
        if args.mode == "sliced":
            preds = predict_sliced(
                model,
                image,
                args.slice_size,
                args.overlap,
                args.conf,
                args.tile_iou,
                args.merge_iou,
                args.tile_max_det,
                args.max_det,
                args.device,
            )
        else:
            preds = predict_normal(
                model,
                image,
                args.imgsz,
                args.conf,
                args.tile_iou,
                args.max_det,
                args.device,
            )
        tp = match_predictions(preds, targets, iouv)
        if len(preds):
            stats.append((tp, preds[:, 4], preds[:, 5]))
        if len(targets):
            target_cls_all.append(targets[:, 4])
        rows.append((image_path.name, len(targets), len(preds)))
        if idx <= 5 or idx == len(image_paths) or idx % args.progress_every == 0:
            print(f"[{idx}/{len(image_paths)}] {image_path.name}: gt={len(targets)} pred={len(preds)}")

    if stats:
        tp = np.concatenate([s[0] for s in stats], axis=0)
        conf = np.concatenate([s[1] for s in stats], axis=0)
        pred_cls = np.concatenate([s[2] for s in stats], axis=0)
    else:
        tp = np.zeros((0, len(iouv)), dtype=bool)
        conf = np.zeros((0,), dtype=np.float32)
        pred_cls = np.zeros((0,), dtype=np.float32)
    target_cls = np.concatenate(target_cls_all, axis=0) if target_cls_all else np.zeros((0,), dtype=np.float32)

    _, _, precision, recall, _, ap, ap_class = ap_per_class(
        tp, conf, pred_cls, target_cls, names=CLASS_NAMES
    )[:7]
    mp = float(np.mean(precision)) if len(precision) else 0.0
    mr = float(np.mean(recall)) if len(recall) else 0.0
    map50 = float(np.mean(ap[:, 0])) if len(ap) else 0.0
    map5095 = float(np.mean(ap)) if len(ap) else 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "weights",
                "mode",
                "images",
                "imgsz",
                "slice_size",
                "overlap",
                "conf",
                "tile_iou",
                "merge_iou",
                "tile_max_det",
                "max_det",
                "precision",
                "recall",
                "mAP50",
                "mAP50-95",
            ]
        )
        writer.writerow(
            [
                args.weights,
                args.mode,
                len(image_paths),
                args.imgsz,
                args.slice_size,
                args.overlap,
                args.conf,
                args.tile_iou,
                args.merge_iou,
                args.tile_max_det,
                args.max_det,
                f"{mp:.6f}",
                f"{mr:.6f}",
                f"{map50:.6f}",
                f"{map5095:.6f}",
            ]
        )

    per_image_path = output_dir / args.per_image_name
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "gt_count", "pred_count"])
        writer.writerows(rows)

    print(
        f"Saved sliced summary to {summary_path}: "
        f"P={mp:.4f} R={mr:.4f} mAP50={map50:.4f} mAP50-95={map5095:.4f}"
    )
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO weights with tiled/sliced inference.")
    parser.add_argument("--mode", choices=["normal", "sliced"], default="sliced")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", default="datasets/Visdrone/VisDrone2019-DET-test-dev/images")
    parser.add_argument("--labels", default="datasets/Visdrone/VisDrone2019-DET-test-dev/labels")
    parser.add_argument("--output-dir", default="VisDrone_Experiments/sliced_eval")
    parser.add_argument("--summary-name", default="stage2b_sliced_summary.csv")
    parser.add_argument("--per-image-name", default="stage2b_sliced_per_image.csv")
    parser.add_argument("--slice-size", type=int, default=896)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--tile-iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.6)
    parser.add_argument("--tile-max-det", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_sliced(parse_args())
