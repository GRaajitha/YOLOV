import pycocotools.coco
from tools.cocoeval_custom import COCOeval
import numpy as np
import io
import contextlib
import json
from yolox.evaluators.coco_evaluator import per_class_AR_table, per_class_AP_table
import tempfile
import os

# # Define file paths
# gt_file = "/shared/vision/experiments/vision-detector/2kinp_v7_w_val_2024-12-14_01-26-18/val_gt_coco_fmt.json"
# dt_file = "/shared/vision/experiments/vision-detector/2kinp_v7_w_val_2024-12-14_01-26-18/val_inference_results.json"

# # Load data
# gt_data = json.load(open(gt_file, "r"))
# dt_data = json.load(open(dt_file, "r"))

# # Add iscrowd and area fields to ground truth annotations
# for ann in gt_data["annotations"]:
#     if 'iscrowd' not in ann:
#         ann['iscrowd'] = 0
#     if 'area' not in ann:
#         # Calculate area from bbox [x, y, width, height]
#         bbox = ann['bbox']
#         ann['area'] = bbox[2] * bbox[3]  # width * height

# # Define categories sorted by ID
# categories = [
#     {'supercategory': 'none', 'id': 0, 'name': 'Airborne'},
#     {'supercategory': 'none', 'id': 1, 'name': 'Zip'},
#     {'supercategory': 'none', 'id': 2, 'name': 'Glider'},
#     {'supercategory': 'none', 'id': 3, 'name': 'Balloon'},
#     {'supercategory': 'none', 'id': 4, 'name': 'Paraglider'},
#     {'supercategory': 'none', 'id': 5, 'name': 'Bird'},
#     {'supercategory': 'none', 'id': 6, 'name': 'Flock'},
#     {'supercategory': 'none', 'id': 7, 'name': 'Airplane'},
#     {'supercategory': 'none', 'id': 8, 'name': 'Ultralight'},
#     {'supercategory': 'none', 'id': 9, 'name': 'Helicopter'},
#     {'supercategory': 'none', 'id': 10, 'name': 'Unknown'},
#     {'supercategory': 'none', 'id': 11, 'name': 'HangGlider'},
#     {'supercategory': 'none', 'id': 12, 'name': 'CommercialAirliner'},
#     {'supercategory': 'none', 'id': 13, 'name': 'Drone'},
#     {'supercategory': 'none', 'id': 14, 'name': 'Artificial'},
#     {'supercategory': 'none', 'id': 15, 'name': 'Natural'}
# ]

# # Create COCO format dictionary for ground truth
# gt_coco = {
#     "info": {},
#     "categories": categories,
#     "images": gt_data["images"],
#     "annotations": gt_data["annotations"]
# }

# # Save ground truth with categories to temporary file
# temp_dir = tempfile.gettempdir()
# gt_temp_file = os.path.join(temp_dir, "temp_gt_coco.json")
# with open(gt_temp_file, "w") as f:
#     json.dump(gt_coco, f)

# Initialize COCO objects
cocoGt = pycocotools.coco.COCO("/home/rgummadi/YOLOV/gt_refined.json")
dt_data = json.load(open("/home/rgummadi/YOLOV/refined_pred.json", "r"))
cocoDt = cocoGt.loadRes(dt_data)

# Initialize evaluator and set parameters
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.params.iouThrs = np.array([0.2, 0.5, 0.75])
cocoEval.params.useCats = 1  # Enable category-based evaluation
cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
cocoEval.params.catIds = sorted(cocoGt.getCatIds())

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()

# Generate output
redirect_string = io.StringIO()
info = ""
with contextlib.redirect_stdout(redirect_string):
    cocoEval.summarize()
info += redirect_string.getvalue()

cat_ids = list(cocoGt.cats.keys())
cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
info += "per class AP:\n" + AP_table + "\n"

AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
info += "per class AR:\n" + AR_table + "\n"

print(cocoEval.stats[0], cocoEval.stats[1], info)