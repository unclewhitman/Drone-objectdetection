import os
import json
import cv2
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO

# MMDetection imports
try:
    from mmdet.apis import init_detector, inference_detector
except ImportError:
    print("Warning: mmdet not installed in this environment. Run this in pf-rpn environment.")

def evaluate_cascade(pf_cfg, pf_ckpt, yolo_cls_path, data_dir, test_json_path, split="VisDrone2019-DET-test-dev"):
    print("Initializing PF-RPN...")
    pf_model = init_detector(pf_cfg, pf_ckpt, device='cuda:0')
    
    print("Initializing YOLOv11 Classification Head...")
    yolo_model = YOLO(yolo_cls_path)
    
    print(f"Loading test annotations {test_json_path}...")
    coco_gt = COCO(test_json_path)
    
    img_dir = os.path.join(data_dir, split, "images")
    
    # In VisDrone_Cls: class names mapped to VisDrone index
    # Note: YOLO classification returns predicted class name (e.g. '0', '1' or 'pedestrian', depending on training mapping)
    visdrone_names = {
        0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 
        4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 
        8: 'bus', 9: 'motor'
    }
    # Create reverse lookup if yolo model predicts string names
    name_to_id = {v: k for k, v in visdrone_names.items()}
    
    coco_dt = []
    
    img_ids = coco_gt.getImgIds()
    for img_id in tqdm(img_ids, desc="Cascade Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 1. PF-RPN extracts Object Proposals
        result = inference_detector(pf_model, img)
        pred_instances = result.pred_instances
        
        # Filter by PF-RPN threshold (e.g. 0.05) to reduce classification overhead
        mask = pred_instances.scores > 0.05
        bboxes = pred_instances.bboxes[mask].cpu().numpy()
        objectness_scores = pred_instances.scores[mask].cpu().numpy()
        
        # Take Top K if there are too many (e.g. top 100)
        top_k = min(100, len(bboxes))
        top_indices = np.argsort(objectness_scores)[::-1][:top_k]
        
        bboxes = bboxes[top_indices]
        objectness_scores = objectness_scores[top_indices]
        
        h_img, w_img, _ = img.shape
        
        # 2. YOLO Classify
        for box, obj_score in zip(bboxes, objectness_scores):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2:
                continue
                
            crop = img[y1:y2, x1:x2]
            
            # Predict
            cls_result = yolo_model(crop, verbose=False)[0]
            
            # Top-1 class
            top1_idx = cls_result.probs.top1
            top1_conf = float(cls_result.probs.top1conf)
            top1_name = cls_result.names[top1_idx]
            
            # Resolve class ID
            cls_id = None
            if str(top1_name).isdigit():
                cls_id = int(top1_name)
            elif top1_name in name_to_id:
                cls_id = name_to_id[top1_name]
                
            if cls_id is None:
                continue
                
            final_conf = obj_score * top1_conf # Joint probability!
            
            coco_dt.append({
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(final_conf)
            })
            
    # Save results
    dt_path = "temp_cascade_dt.json"
    with open(dt_path, "w") as f:
        json.dump(coco_dt, f)
        
    print("Evaluating Cascade Output (Multi-class)...")
    coco_dt_obj = coco_gt.loadRes(dt_path)
    coco_eval = COCOeval(coco_gt, coco_dt_obj, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    os.remove(dt_path)

if __name__ == "__main__":
    pf_root = Path("third_party") / "PF-RPN-main"
    pf_cfg = str(pf_root / "configs" / "pf-rpn" / "visdrone.py")
    # User interrupted training, let's use the pre-trained universal PF-RPN weights which has strong zero-shot capability!
    pf_ckpt = str(pf_root / "checkpoints" / "pf_rpn_swinb_5p_coco_imagenet.pth")
    base_exp_dir = Path("VisDrone_Experiments")
    cls_exps = list(base_exp_dir.glob("yolo11n_cls_baseline*"))
    if not cls_exps:
        yolo_cls_path = "VisDrone_Experiments/yolo11n_cls_baseline/weights/best.pt"
    else:
        # Sort by modification time to get the latest completed training run
        latest_exp = max(cls_exps, key=os.path.getmtime)
        yolo_cls_path = str(latest_exp / "weights" / "best.pt")
        
    print(f"Using YOLO weights from: {yolo_cls_path}")
    
    base_dir = "datasets/Visdrone"
    # Note: evaluate multi-class needs multi-class GT
    test_json = os.path.join(base_dir, "VisDrone2019-DET-test-dev", "annotations", "test_coco_multi.json")
    
    # WARNING: This script needs both mmdet and ultralytics in the SAME environment, 
    # OR you have to run PF-RPN offline, dump proposals, and then run YOLO in a separate script.
    # The current approach assumes both can be installed in `pf-rpn` conda env via `pip install ultralytics`
    try:
        evaluate_cascade(pf_cfg, pf_ckpt, yolo_cls_path, base_dir, test_json)
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Tip: You might need to `pip install ultralytics tqdm pycocotools` within your pf-rpn mmcv environment.")
