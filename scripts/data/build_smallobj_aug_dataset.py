import argparse
import os
import random
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.sobj_utils import xywh_to_xyxy, bbox_area, iou_xyxy


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


def save_labels(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, x, y, w, h in labels:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def select_small_objects(labels, area_ratio_thresh=0.01):
    small = []
    for cls_id, x, y, w, h in labels:
        if (w * h) < area_ratio_thresh:
            small.append((cls_id, x, y, w, h))
    return small


def xywh_norm_to_xyxy_px(label, width, height):
    cls_id, x, y, w, h = label
    x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)
    return (cls_id, x1 * width, y1 * height, x2 * width, y2 * height)


def xyxy_px_to_xywh_norm(cls_id, x1, y1, x2, y2, width, height):
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0.0 or h <= 0.0:
        return None
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    return (cls_id, cx, cy, w / width, h / height)


def random_crop_around_obj(img, labels, obj_label, min_scale=0.5, max_scale=0.9):
    height, width = img.shape[:2]
    cls_id, x, y, w, h = obj_label
    obj_cx = x * width
    obj_cy = y * height
    obj_w = w * width
    obj_h = h * height

    crop_w = random.uniform(min_scale, max_scale) * width
    crop_h = random.uniform(min_scale, max_scale) * height

    crop_w = max(crop_w, obj_w * 2.0, 32.0)
    crop_h = max(crop_h, obj_h * 2.0, 32.0)
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)

    jitter_x = random.uniform(-0.1, 0.1) * crop_w
    jitter_y = random.uniform(-0.1, 0.1) * crop_h

    x1 = int(round(obj_cx - crop_w / 2 + jitter_x))
    y1 = int(round(obj_cy - crop_h / 2 + jitter_y))
    x1 = max(0, min(x1, width - int(crop_w)))
    y1 = max(0, min(y1, height - int(crop_h)))
    x2 = int(round(x1 + crop_w))
    y2 = int(round(y1 + crop_h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    new_labels = []
    crop_w = x2 - x1
    crop_h = y2 - y1
    scale_x = width / float(crop_w)
    scale_y = height / float(crop_h)

    for lab in labels:
        cls2, bx1, by1, bx2, by2 = xywh_norm_to_xyxy_px(lab, width, height)
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        nx1 = max(bx1, x1) - x1
        ny1 = max(by1, y1) - y1
        nx2 = min(bx2, x2) - x1
        ny2 = min(by2, y2) - y1
        nx1 *= scale_x
        nx2 *= scale_x
        ny1 *= scale_y
        ny2 *= scale_y
        nlab = xyxy_px_to_xywh_norm(cls2, nx1, ny1, nx2, ny2, width, height)
        if nlab is not None:
            new_labels.append(nlab)

    if not new_labels:
        return None

    return resized, new_labels


def _pixel_boxes(labels, width, height):
    boxes = []
    for lab in labels:
        cls_id, x1, y1, x2, y2 = xywh_norm_to_xyxy_px(lab, width, height)
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def copy_paste_local(img, labels, obj_label, max_offset_ratio=0.2, iou_thresh=0.1):
    height, width = img.shape[:2]
    cls_id, x1, y1, x2, y2 = xywh_norm_to_xyxy_px(obj_label, width, height)
    x1i, y1i, x2i, y2i = map(int, [round(x1), round(y1), round(x2), round(y2)])
    if x2i <= x1i or y2i <= y1i:
        return None

    patch = img[y1i:y2i, x1i:x2i]
    if patch.size == 0:
        return None

    dx = random.uniform(-max_offset_ratio, max_offset_ratio) * width
    dy = random.uniform(-max_offset_ratio, max_offset_ratio) * height

    new_x1 = int(round(x1i + dx))
    new_y1 = int(round(y1i + dy))
    new_x2 = new_x1 + (x2i - x1i)
    new_y2 = new_y1 + (y2i - y1i)

    if new_x1 < 0 or new_y1 < 0 or new_x2 > width or new_y2 > height:
        return None

    new_box = (new_x1, new_y1, new_x2, new_y2)

    for _, bx1, by1, bx2, by2 in _pixel_boxes(labels, width, height):
        if iou_xyxy(new_box, (bx1, by1, bx2, by2)) >= iou_thresh:
            return None

    out = img.copy()
    out[new_y1:new_y2, new_x1:new_x2] = patch

    new_label = xyxy_px_to_xywh_norm(cls_id, new_x1, new_y1, new_x2, new_y2, width, height)
    if new_label is None:
        return None

    new_labels = labels + [new_label]
    return out, new_labels


def process_image(img_path, label_path, out_img_dir, out_lbl_dir, args):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    labels = load_labels(label_path)

    if args.keep_original:
        out_img = out_img_dir / img_path.name
        out_lbl = out_lbl_dir / label_path.name
        shutil.copy2(img_path, out_img)
        save_labels(out_lbl, labels)

    small_objs = select_small_objects(labels, args.area_ratio_thresh)
    if not small_objs:
        return 1

    created = 0
    for aug_idx in range(args.aug_per_image):
        base_img = img
        base_labels = labels

        use_crop = random.random() < args.crop_prob
        if use_crop:
            obj = random.choice(small_objs)
            crop_result = random_crop_around_obj(base_img, base_labels, obj, args.crop_min_scale, args.crop_max_scale)
            if crop_result is not None:
                base_img, base_labels = crop_result

        paste_attempts = 0
        while paste_attempts < args.max_paste:
            if random.random() > args.paste_prob:
                break
            small_after = select_small_objects(base_labels, args.area_ratio_thresh)
            if not small_after:
                break
            obj = random.choice(small_after)
            paste_result = copy_paste_local(base_img, base_labels, obj, args.paste_offset_ratio, args.iou_thresh)
            if paste_result is None:
                paste_attempts += 1
                continue
            base_img, base_labels = paste_result
            paste_attempts += 1

        out_name = f"{img_path.stem}_sobjaug_{aug_idx}{img_path.suffix}"
        out_img = out_img_dir / out_name
        out_lbl = out_lbl_dir / f"{img_path.stem}_sobjaug_{aug_idx}.txt"
        cv2.imwrite(str(out_img), base_img)
        save_labels(out_lbl, base_labels)
        created += 1

    return created


def build_dataset(src, dst, args):
    src = Path(src)
    dst = Path(dst)
    img_dir = src / "images"
    lbl_dir = src / "labels"

    out_img_dir = dst / "images"
    out_lbl_dir = dst / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if args.limit:
        img_files = img_files[: args.limit]

    total = 0
    for img_path in img_files:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        total += process_image(img_path, label_path, out_img_dir, out_lbl_dir, args)

    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Build small-object augmented dataset (target-aware crop + constrained copy-paste).")
    parser.add_argument("--src", required=True, help="Source dataset folder containing images/ and labels/")
    parser.add_argument("--dst", required=True, help="Destination dataset folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--area-ratio-thresh", type=float, default=0.01, help="Small-object area ratio threshold")
    parser.add_argument("--aug-per-image", type=int, default=1, help="Augmented variants per image with small objects")
    parser.add_argument("--keep-original", action="store_true", help="Copy original image/label to dst")
    parser.add_argument("--crop-prob", type=float, default=0.8, help="Probability to apply target-aware crop")
    parser.add_argument("--crop-min-scale", type=float, default=0.5, help="Min crop scale vs original")
    parser.add_argument("--crop-max-scale", type=float, default=0.9, help="Max crop scale vs original")
    parser.add_argument("--paste-prob", type=float, default=0.7, help="Probability to apply copy-paste")
    parser.add_argument("--max-paste", type=int, default=2, help="Max copy-paste attempts per image")
    parser.add_argument("--paste-offset-ratio", type=float, default=0.2, help="Max local offset ratio for paste")
    parser.add_argument("--iou-thresh", type=float, default=0.1, help="IoU threshold for paste placement")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images for quick test")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    total = build_dataset(args.src, args.dst, args)
    print(f"Done. Processed images: {total}")


if __name__ == "__main__":
    main()
