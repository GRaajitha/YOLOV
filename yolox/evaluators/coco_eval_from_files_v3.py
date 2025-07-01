import pycocotools.coco
from tools.cocoeval_custom import COCOeval
import numpy as np
import io
import contextlib
import json
from yolox.evaluators.coco_evaluator import log_pr_curve, per_class_AR_table, per_class_AP_table, evaluate_per_attribute_per_class, log_per_attribute_per_class_metrics
import tempfile
import os
import matplotlib.pyplot as plt
import wandb

def plot_pr_curve(cocoEval, iou_threshold, output_filename):
    """Plots the Precision-Recall curve averaged over categories.

    Args:
        cocoEval (COCOeval): The evaluated COCOeval object.
        iou_threshold (float): The IoU threshold to use for the curve.
        output_filename (str): The path to save the plot image.
    """
    try:
        # Find the index for the given IoU threshold
        iou_idx = np.where(np.isclose(cocoEval.params.iouThrs, iou_threshold))[0][0]
    except IndexError:
        print(f"Warning: IoU threshold {iou_threshold} not found in params.iouThrs.")
        print(f"Available thresholds: {cocoEval.params.iouThrs}")
        return

    # Default indices for area range ('all') and max detections (100)
    area_idx = 0  # Index for area = 'all'
    max_dets_idx = cocoEval.params.maxDets.index(100) if 100 in cocoEval.params.maxDets else -1
    if max_dets_idx == -1:
        print(f"Warning: Max detections 100 not found in params.maxDets: {cocoEval.params.maxDets}. Using last index.")
        max_dets_idx = len(cocoEval.params.maxDets) - 1

    # Extract precision values across all recall thresholds (R) and categories (K) Shape: [R, K]
    precisions_all_categories = cocoEval.eval['precision'][iou_idx, :, :, area_idx, max_dets_idx]

    # Average precision across categories (axis=1), treating -1 as NaN
    mean_precision = np.nanmean(np.where(precisions_all_categories == -1, np.nan, precisions_all_categories), axis=1)

    # Get recall thresholds
    recall = cocoEval.params.recThrs

    # Filter out NaN values from averaging and corresponding recall values
    valid_mask = ~np.isnan(mean_precision)
    final_precision = mean_precision[valid_mask]
    final_recall = recall[valid_mask]

    # Check if there's data to plot
    if final_recall.size == 0 or final_precision.size == 0:
        print(f"Warning: No valid data points found to plot the PR curve for IoU={iou_threshold}.")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(final_recall, final_precision, 'b-', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (IoU={iou_threshold}, Averaged Over Categories)', fontsize=14)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(output_filename)
        print(f"Precision-Recall curve saved to {output_filename}")
        plt.close()


# Define file paths
gt_file = "/shared/vision/dataset/metadata/v7_8_cls/coco_vid/trimmed1000_64-500seq_test_coco_vid_06_06.json"
dt_file = "/shared/users/raajitha/YOLOVexperiments/test_yolox_nano_v7_8_cls_1080x1920_20ep_2025-06-16_trimmed100_fixedlen_02_27_test_split_video_sequences.json/refined_pred.json"
wandb_name = "test_w_attributes_yolox_nano_v7_8_cls_1080x1920_20ep_2025-05-22_trimmed1000_64-500seq_test_coco_vid_06_06"

# Load data
gt_data = json.load(open(gt_file, "r"))
dt_data = json.load(open(dt_file, "r"))

# Add iscrowd and area fields to ground truth annotations
for ann in gt_data["annotations"]:
    if 'iscrowd' not in ann:
        ann['iscrowd'] = 0
    # if ann["category_id"] in [0, 6, 10, 11, 12, 13, 14, 15]:
    #     ann['ignore'] = 1
    if 'area' not in ann:
        # Calculate area from bbox [x, y, width, height]
        bbox = ann['bbox']
        ann['area'] = bbox[2] * bbox[3]  # width * height
        # if ann['area'] < 100:
        #     ann['ignore'] = 1

# Define categories sorted by ID
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

categories=[{'supercategory': 'none', 'id': 0, 'name': 'Airplane'},
            {'supercategory': 'none', 'id': 1, 'name': 'Paraglider'},
            {'supercategory': 'none', 'id': 2, 'name': 'Helicopter'},
            {'supercategory': 'none', 'id': 3, 'name': 'Zip'},
            {'supercategory': 'none', 'id': 4, 'name': 'Ultralight'},
            {'supercategory': 'none', 'id': 5, 'name': 'Glider'},
            {'supercategory': 'none', 'id': 6, 'name': 'Bird'},
            {'supercategory': 'none', 'id': 7, 'name': 'Balloon'}]

# Create COCO format dictionary for ground truth
gt_coco = {
    "info": {},
    "categories": categories,
    "images": gt_data["images"],
    "annotations": gt_data["annotations"]
}

# Save ground truth with categories to temporary file
temp_dir = tempfile.gettempdir()
gt_temp_file = os.path.join(temp_dir, "temp_gt_coco.json")
with open(gt_temp_file, "w") as f:
    json.dump(gt_coco, f)

# Initialize COCO objects
cocoGt = pycocotools.coco.COCO(gt_temp_file)
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
    cocoEval.summarize(compute_confidence_matrix=True)
info += redirect_string.getvalue()

wandb.init(project="YOLOV-tools", name=wandb_name)
cat_ids = list(cocoGt.cats.keys())
cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
AP_table, per_class_AP = per_class_AP_table(cocoEval, class_names=cat_names)
wandb.log({f"val/mAP_{name}":value/100 for name, value in per_class_AP.items()})
info += "per class AP:\n" + AP_table + "\n"
AR_table, per_class_AR = per_class_AR_table(cocoEval, class_names=cat_names)
wandb.log({f"val/mAR_{name}":value/100 for name, value in per_class_AR.items()})
info += "per class AR:\n" + AR_table + "\n"

log_pr_curve(cocoEval, iou_threshold=0.5)

print(cocoEval.stats[0], cocoEval.stats[1], info, cocoEval.conf_matrix)

ap50_95, ap50 =cocoEval.stats[0], cocoEval.stats[1]
wandb.log({
    "val/COCOAP50": ap50,
    "val/COCOAP50_95": ap50_95,
})
attribute_names = ["horizon", "size_cat", "occlusion", "clipping", "primary_terrain", "secondary_terrain", "terrain_modifier", "low_visibility", "annotated_weather", "cloud_coverage", "intruder_lateral_view", "intruder_vertical_view", "image_quality"]

attr_results = evaluate_per_attribute_per_class(cocoGt, cocoDt, cat_names, attribute_names=attribute_names)
log_per_attribute_per_class_metrics(attr_results, iouThrs=[0.5])
