import os
import cv2
from pathlib import Path
from tqdm import tqdm

def process_dataset(split, output_dir, src_img_dir, src_label_dir, class_names):
    out_split_dir = os.path.join(output_dir, split)
    
    # Check if already processed (e.g. at least one class directory has files)
    if os.path.exists(out_split_dir):
        total_files = sum([len(list(Path(os.path.join(out_split_dir, d)).glob("*.jpg"))) for d in class_names.values() if os.path.exists(os.path.join(out_split_dir, d))])
        if total_files > 100:  # Arbitrary threshold to ensure it's not empty
            print(f"Skipping {split} crop generation, files already exist.")
            return

    for class_name in class_names.values():
        os.makedirs(os.path.join(out_split_dir, class_name), exist_ok=True)
    
    img_files = list(Path(src_img_dir).glob("*.jpg"))
    
    for img_path in tqdm(img_files, desc=f"Processing {split}"):
        label_path = os.path.join(src_label_dir, img_path.stem + ".txt")
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id not in class_names:
                    continue
                
                cx, cy, bw, bh = map(float, parts[1:5])
                
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                # Boundary check
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue # Ignore extremely small crops
                    
                crop_img = img[y1:y2, x1:x2]
                out_filename = f"{img_path.stem}_{idx}.jpg"
                out_path = os.path.join(out_split_dir, class_names[cls_id], out_filename)
                
                cv2.imwrite(out_path, crop_img)

if __name__ == "__main__":
    names = {
        0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 
        4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 
        8: 'bus', 9: 'motor'
    }
    
    base_dir = "datasets/Visdrone"
    out_dir = "datasets/Visdrone_Cls"
    
    splits = [
        ("train", "VisDrone2019-DET-train"),
        ("val", "VisDrone2019-DET-val"),
        ("test", "VisDrone2019-DET-test-dev")
    ]
    
    for split_name, split_folder in splits:
        img_dir = os.path.join(base_dir, split_folder, "images")
        lbl_dir = os.path.join(base_dir, split_folder, "labels")
        process_dataset(split_name, out_dir, img_dir, lbl_dir, names)
    
    print("Done generating classification dataset.")
