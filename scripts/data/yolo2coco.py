import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm

def yolo_to_coco(names, src_img_dir, src_label_dir, out_json_path, single_class=False):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    if single_class:
        coco_format['categories'].append({"id": 0, "name": "object"})
    else:
        for idx, name in names.items():
            coco_format['categories'].append({"id": idx, "name": name})
            
    img_files = list(Path(src_img_dir).glob("*.jpg"))
    annotation_id = 1
    
    for img_id, img_path in enumerate(tqdm(img_files, desc=f"Converting {Path(out_json_path).name}")):
        label_path = os.path.join(src_label_dir, img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape
        
        coco_format['images'].append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": img_path.name
        })
        
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id not in names:
                    continue
                
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # YOLO center to COCO top-left
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                width = bw * w
                height = bh * h
                
                # Bound check
                x1 = max(0, x1)
                y1 = max(0, y1)
                width = min(width, w - x1)
                height = min(height, h - y1)
                
                area = width * height
                
                cat_id = 0 if single_class else cls_id
                
                coco_format['annotations'].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, width, height],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
                
    with open(out_json_path, "w") as f:
        json.dump(coco_format, f)

if __name__ == "__main__":
    names = {
        0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 
        4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 
        8: 'bus', 9: 'motor'
    }
    
    base_dir = "datasets/Visdrone"
    
    splits = [
        ("train", "VisDrone2019-DET-train"),
        ("val", "VisDrone2019-DET-val"),
        ("test", "VisDrone2019-DET-test-dev")
    ]
    
    for split_name, split_folder in splits:
        img_dir = os.path.join(base_dir, split_folder, "images")
        lbl_dir = os.path.join(base_dir, split_folder, "labels")
        
        ann_dir = os.path.join(base_dir, split_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        
        out_multi = os.path.join(ann_dir, f"{split_name}_coco_multi.json")
        out_single = os.path.join(ann_dir, f"{split_name}_coco_single.json")
        
        if os.path.exists(out_multi) and os.path.exists(out_single):
            print(f"Skipping {split_name} conversion, files already exist.")
            continue
            
        yolo_to_coco(names, img_dir, lbl_dir, out_multi, single_class=False)
        yolo_to_coco(names, img_dir, lbl_dir, out_single, single_class=True)
        
    print("Done converting YOLO to COCO.")
