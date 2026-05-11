import argparse
import random
import shutil
from pathlib import Path

import cv2

from sliced_yolo_eval import generate_slices


def load_yolo_labels(label_path):
    labels = []
    if not label_path.is_file():
        return labels
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = parts
            labels.append((int(cls_id), float(x), float(y), float(w), float(h)))
    return labels


def save_yolo_labels(label_path, labels):
    with label_path.open("w", encoding="utf-8") as handle:
        for cls_id, x, y, w, h in labels:
            handle.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def label_to_xyxy_px(label, width, height):
    cls_id, x, y, w, h = label
    x1 = (x - w / 2.0) * width
    y1 = (y - h / 2.0) * height
    x2 = (x + w / 2.0) * width
    y2 = (y + h / 2.0) * height
    return cls_id, x1, y1, x2, y2


def clip_labels_to_tile(labels, image_width, image_height, tile, min_visible=0.3, center_required=True):
    tx1, ty1, tx2, ty2 = tile
    tile_width = tx2 - tx1
    tile_height = ty2 - ty1
    clipped = []

    for label in labels:
        cls_id, x1, y1, x2, y2 = label_to_xyxy_px(label, image_width, image_height)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        box_area = box_width * box_height
        if box_area <= 0.0:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_inside = tx1 <= cx <= tx2 and ty1 <= cy <= ty2

        nx1 = max(x1, tx1)
        ny1 = max(y1, ty1)
        nx2 = min(x2, tx2)
        ny2 = min(y2, ty2)
        visible_area = max(0.0, nx2 - nx1) * max(0.0, ny2 - ny1)
        if visible_area <= 0.0:
            continue
        if center_required and not center_inside:
            continue
        if visible_area / box_area < min_visible:
            continue

        nx1 -= tx1
        nx2 -= tx1
        ny1 -= ty1
        ny2 -= ty1
        nw = max(0.0, nx2 - nx1)
        nh = max(0.0, ny2 - ny1)
        if nw <= 1.0 or nh <= 1.0:
            continue

        clipped.append(
            (
                cls_id,
                (nx1 + nx2) / 2.0 / tile_width,
                (ny1 + ny2) / 2.0 / tile_height,
                nw / tile_width,
                nh / tile_height,
            )
        )

    return clipped


def copy_original(image_path, label_path, out_img_dir, out_lbl_dir):
    shutil.copy2(image_path, out_img_dir / image_path.name)
    if label_path.is_file():
        shutil.copy2(label_path, out_lbl_dir / label_path.name)
    else:
        (out_lbl_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")


def process_image(image_path, label_path, out_img_dir, out_lbl_dir, args):
    image = cv2.imread(str(image_path))
    if image is None:
        return 0
    height, width = image.shape[:2]
    labels = load_yolo_labels(label_path)
    created = 0

    if args.keep_original:
        copy_original(image_path, label_path, out_img_dir, out_lbl_dir)
        created += 1

    for idx, tile in enumerate(generate_slices(width, height, args.tile_size, args.overlap)):
        tile_labels = clip_labels_to_tile(
            labels,
            width,
            height,
            tile,
            min_visible=args.min_visible,
            center_required=not args.allow_offcenter,
        )
        if not tile_labels and random.random() > args.empty_ratio:
            continue

        x1, y1, x2, y2 = tile
        tile_img = image[y1:y2, x1:x2]
        if tile_img.size == 0:
            continue

        out_name = f"{image_path.stem}_tile{args.tile_size}_{idx:03d}{image_path.suffix}"
        out_img = out_img_dir / out_name
        out_lbl = out_lbl_dir / f"{Path(out_name).stem}.txt"
        cv2.imwrite(str(out_img), tile_img)
        save_yolo_labels(out_lbl, tile_labels)
        created += 1

    return created


def build_dataset(args):
    random.seed(args.seed)
    src = Path(args.src)
    dst = Path(args.dst)
    img_dir = src / "images"
    lbl_dir = src / "labels"
    out_img_dir = dst / "images"
    out_lbl_dir = dst / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        image_paths = image_paths[: args.limit]

    total = 0
    for idx, image_path in enumerate(image_paths, start=1):
        label_path = lbl_dir / f"{image_path.stem}.txt"
        total += process_image(image_path, label_path, out_img_dir, out_lbl_dir, args)
        if idx <= 5 or idx == len(image_paths) or idx % args.progress_every == 0:
            print(f"[{idx}/{len(image_paths)}] {image_path.name}: total_outputs={total}")
    print(f"Done. Wrote {total} image/label pairs to {dst}")
    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Build a YOLO tile dataset for small-object training.")
    parser.add_argument("--src", required=True, help="Source split folder containing images/ and labels/")
    parser.add_argument("--dst", required=True, help="Destination split folder")
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-visible", type=float, default=0.3)
    parser.add_argument("--empty-ratio", type=float, default=0.1)
    parser.add_argument("--keep-original", action="store_true")
    parser.add_argument("--allow-offcenter", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
