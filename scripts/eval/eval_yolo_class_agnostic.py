import os
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from tqdm import tqdm

def eval_class_agnostic(model_path, data_dir, test_json_path, split="VisDrone2019-DET-test-dev"):
    print(f"Loading COCO Ground Truth from {test_json_path}...")
    coco_gt = COCO(test_json_path)
    
    print(f"Loading YOLO Model {model_path}...")
    model = YOLO(model_path)
    
    img_dir = os.path.join(data_dir, split, "images")
    img_files = list(Path(img_dir).glob("*.jpg"))
    
    coco_dt = []
    
    for img_id in tqdm(coco_gt.getImgIds(), desc="YOLO Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        
        results = model(img_path, verbose=False)[0]
        boxes = results.boxes.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            
            w = x2 - x1
            h = y2 - y1
            
            coco_dt.append({
                "image_id": img_id,
                "category_id": 0,  # Force to object class
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": conf
            })
            
    # Save results to temp json
    dt_path = "temp_yolo_dt.json"
    with open(dt_path, "w") as f:
        json.dump(coco_dt, f)
        
    print("Evaluating Class-Agnostic Performance via pycocotools...")
    coco_dt_obj = coco_gt.loadRes(dt_path)
    
    coco_eval = COCOeval(coco_gt, coco_dt_obj, "bbox")
    coco_eval.params.catIds = [0]
    # Set max detections to standard 100, 300, 1000 for AR (matching PF-RPN usually computes AR@100, AR@300)
    coco_eval.params.maxDets = [100, 300, 1000]
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    os.remove(dt_path)

if __name__ == "__main__":
    model_path = "VisDrone_Experiments/yolo11n_baseline/weights/best.pt"
    # If the user has a specific model, change it here, for instance 'yolo11n.pt'
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"  # Fallback
        
    base_dir = "datasets/Visdrone"
    test_json = os.path.join(base_dir, "VisDrone2019-DET-test-dev", "annotations", "test_coco_single.json")
    
    eval_class_agnostic(model_path, base_dir, test_json)
