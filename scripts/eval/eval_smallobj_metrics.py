import argparse
import csv
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class


def load_labels(label_path):
    labels = []
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = parts
            labels.append((int(cls_id), float(x), float(y), float(w), float(h)))
    return labels


def filter_small_gt(gts, area_ratio_thresh=0.01):
    return [g for g in gts if (g[3] * g[4]) < area_ratio_thresh]


def collect_gt(pred_results, labels_dir, area_ratio_thresh=0.01):
    # Build GT arrays aligned to predictions for ap_per_class
    gts = []
    for r in pred_results:
        name = Path(r.path).stem
        label_path = labels_dir / f"{name}.txt"
        labels = load_labels(label_path)
        labels = filter_small_gt(labels, area_ratio_thresh)
        gts.append(labels)
    return gts


def main():
    parser = argparse.ArgumentParser(description="Evaluate small-object metrics.")
    parser.add_argument("--weights", required=True, help="Path to weights")
    parser.add_argument("--data", required=True, help="Dataset yaml")
    parser.add_argument("--split", default="test", help="split: train/val/test")
    parser.add_argument("--area-ratio-thresh", type=float, default=0.01)
    parser.add_argument("--csv", required=True, help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.val(data=args.data, split=args.split, verbose=False)

    # Use model.val to get predictions and use labels for small-object filtering
    pred_results = results.pred
    # Fallback if pred is not available
    if pred_results is None:
        print("Warning: results.pred unavailable. This script expects results.pred from Ultralytics.")
        return

    # Find labels directory from dataset
    data = results.args
    data_path = Path(data.data)
    data_dir = data_path.parent
    labels_dir = data_dir / "labels"

    gts = collect_gt(pred_results, labels_dir, args.area_ratio_thresh)

    # Gather preds in format required by ap_per_class
    stats = []
    for r, gt in zip(pred_results, gts):
        if r is None or r.boxes is None:
            continue
        boxes = r.boxes
        if boxes.xyxy is None or boxes.conf is None or boxes.cls is None:
            continue
        predn = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        pred_cls = boxes.cls.cpu().numpy()

        # Build target array
        tcls = np.array([g[0] for g in gt], dtype=np.int32)
        tbox = np.array([[g[1], g[2], g[3], g[4]] for g in gt], dtype=np.float32)

        stats.append((predn, conf, pred_cls, tbox, tcls))

    if not stats:
        print("No stats collected; check predictions.")
        return

    # Run ap_per_class
    p, r, ap, f1, ap_class = ap_per_class(*zip(*stats))
    ap50_95 = ap.mean() if ap is not None and len(ap) else 0.0

    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["small_recall", "small_mAP50-95"])
        writer.writerow([f"{float(np.mean(r)):.6f}", f"{float(ap50_95):.6f}"])

    print(f"Saved small-object metrics to {args.csv}")


if __name__ == "__main__":
    main()
