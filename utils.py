from shapely.geometry import box
from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
from PIL import Image
import pandas as pd
from supervision.detection.utils import box_iou_batch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import List, Tuple, Union


def compute_iou(boxes_true: np.ndarray, boxes_detection: np.ndarray,IOU_threshold = 0.5) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes using the `shapely` library.
    
    Args:
        boxes_true (np.ndarray): Ground-truth bounding boxes of shape `(N, 4)`.
        boxes_detection (np.ndarray): Detected bounding boxes of shape `(M, 4)`.
    
    Returns:
        np.ndarray: IoU matrix of shape `(N, M)`, where each entry (i, j) is the IoU
        between the i-th ground-truth box and the j-th detected box.
    """
    true_polygons = [box(*b) for b in boxes_true]
    detection_polygons = [box(*b) for b in boxes_detection]

    iou_matrix = np.zeros((len(true_polygons), len(detection_polygons)))

    for i, true_box in enumerate(true_polygons):
        for j, det_box in enumerate(detection_polygons):
            inter_area = true_box.intersection(det_box).area
            union_area = true_box.union(det_box).area
            iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0

    # True Positives: Detections that match at least one ground-truth box
    tp = np.sum(np.any(iou_matrix >= IOU_threshold, axis=0))  

    # False Positives: Detections that do not match any ground-truth box
    fp = np.sum(np.all(iou_matrix < IOU_threshold, axis=0))  

    # False Negatives: Ground-truth boxes that do not match any detection
    fn = np.sum(np.all(iou_matrix < IOU_threshold, axis=1))  

    return iou_matrix , (tp,fp,fn)


def Compute_AP(model,test_images: List[str],test_labels: List[str],r_levels: str = "Pascal_VOC11",method: str = "Interpolation") -> float:
    """
    Computes the Average Precision (AP) for a given object detection model.

    Parameters:
    model: The object detection model to evaluate.
    test_images (List[str]): List of paths to test images.
    test_labels (List[str]): List of paths to test label files (YOLO format).
    r_levels (str, optional): Recall levels method, either 'Pascal_VOC11' or 'COCO_101'. Default is 'Pascal_VOC11'.
    method (str, optional): Method to compute AP, either 'Interpolation' or 'Area_Under_Curve'. Default is 'Interpolation'.

    Returns:
    float: The computed Average Precision (AP) value.
    """
    AP_df = pd.DataFrame(columns=["IOU", "Confidence", "TP", "FP"])
    total_gt_box_count = 0
    test_img_dir = "data/test/images"
    test_label_dir = "data/test/labels"
    
    for img_path, label_path in zip(test_images, test_labels):
        img = Image.open(os.path.join(test_img_dir, img_path))
        data = pd.read_csv(os.path.join(test_label_dir, label_path), header=None, sep=" ")
        total_gt_box_count += len(data)
        gt_boxes = []
        
        for _, row in data.iterrows():
            xc, yc, w, h = row[1:].values
            x1 = (xc - w / 2) * 416
            y1 = (yc - h / 2) * 416
            x2 = (xc + w / 2) * 416
            y2 = (yc + h / 2) * 416
            gt_boxes.append([x1, y1, x2, y2])

        result = model(img, verbose=False)
        pred_boxes = result[0].boxes.xyxy.cpu().numpy()
        iou_matrix,(tp,fp,fn) = compute_iou(np.array(gt_boxes), pred_boxes)
        max_ious = np.max(iou_matrix, axis=0) if iou_matrix.size else np.array([])
        conf = result[0].boxes.conf.cpu().numpy()

        for i, pt_box in enumerate(pred_boxes): 
            AP_df = pd.concat([
                AP_df,
                pd.DataFrame({
                    "IOU": max_ious[i],
                    "Confidence": conf[i],
                    "TP": int(max_ious[i] > 0.5) ,
                    "FP": int(max_ious[i] <= 0.5) 
                }, index=[0])
            ], ignore_index=True)

    AP_df = AP_df.sort_values(by="Confidence", ascending=False, ignore_index=True)
    AP_df["cum_TP"] = AP_df["TP"].cumsum()
    AP_df["cum_FP"] = AP_df["FP"].cumsum()
    AP_df["Precision"] = AP_df["cum_TP"] / (AP_df["cum_TP"] + AP_df["cum_FP"])
    AP_df["Recall"] = AP_df["cum_TP"] / total_gt_box_count

    if r_levels == "Pascal_VOC11":
        recall_levels = np.linspace(0, 1, 11)
    elif r_levels == "COCO_101":
        recall_levels = np.linspace(0, 1, 101)
    else:
        raise ValueError("Invalid recall levels. Choose either 'Pascal_VOC11' or 'COCO_101'.")
    
    if method == "Interpolation":
        precisions_at_recall = []
        for r in recall_levels:
            valid_precisions = AP_df.loc[AP_df["Recall"] >= r, "Precision"]
            max_precision = valid_precisions.max() if not valid_precisions.empty else 0
            precisions_at_recall.append(max_precision)
        AP = np.mean(precisions_at_recall)

    elif method == "Area_Under_Curve":
        area_under_curve = 0
        for i in range(len(recall_levels) - 1):
            recall_start, recall_end = recall_levels[i], recall_levels[i + 1]
            
            # Find the max precision in this recall range
            valid_precisions = AP_df.loc[
                (AP_df["Recall"] >= recall_start) & (AP_df["Recall"] < recall_end),
                "Precision"
            ]
            max_precision = valid_precisions.max() if not valid_precisions.empty else 0
            
            # Multiply by the recall step size to approximate area under curve
            area_under_curve += max_precision * (recall_end - recall_start)
        
        AP = area_under_curve
    else:
        raise ValueError("Invalid method. Choose either 'Interpolation' or 'Area_Under_Curve'.")
    
    return AP , AP_df


def generate_boxes(num_boxes, image_size, box_size=20):
    """
    Generate `num_boxes` random bounding boxes of size `box_size`x`box_size`
    within an image of size `image_size`.
    """
    boxes = []
    for _ in range(num_boxes):
        x = np.random.randint(0, image_size[1] - box_size)
        y = np.random.randint(0, image_size[0] - box_size)
        boxes.append([x, y, x + box_size, y + box_size])  # (x_min, y_min, x_max, y_max)
    return np.array(boxes)


def Compute_AP_withboxes(gt_boxes,pt_boxes,conf,method: str = "Interpolation",r_levels: str = "Pascal_VOC11") -> float:
    """
    Computes the Average Precision (AP) for a given object detection model.
    """

    iou_matrix = box_iou_batch(gt_boxes, pt_boxes)
    max_ious = np.max(iou_matrix, axis=0) if iou_matrix.size else np.array([])

    AP_df = pd.DataFrame({
        "IOU": max_ious,
        "Confidence": conf,
        "TP": max_ious > 0.5,
        "FP": max_ious <= 0.5
    })

    AP_df = AP_df.sort_values(by="Confidence", ascending=False, ignore_index=True)
    AP_df["cum_TP"] = AP_df["TP"].cumsum()
    AP_df["cum_FP"] = AP_df["FP"].cumsum()
    AP_df["Precision"] = AP_df["cum_TP"] / (AP_df["cum_TP"] + AP_df["cum_FP"])
    AP_df["Recall"] = AP_df["cum_TP"] / len(gt_boxes)

    if r_levels == "Pascal_VOC11":
        recall_levels = np.linspace(0, 1, 11)
    elif r_levels == "COCO_101":
        recall_levels = np.linspace(0, 1, 101)
    else:
        raise ValueError("Invalid recall levels. Choose either 'Pascal_VOC11' or 'COCO_101'.")
    
    if method == "Interpolation":
        precisions_at_recall = []
        for r in recall_levels:
            valid_precisions = AP_df.loc[AP_df["Recall"] >= r, "Precision"]
            max_precision = valid_precisions.max() if not valid_precisions.empty else 0
            precisions_at_recall.append(max_precision)
        AP = np.mean(precisions_at_recall)

    elif method == "Area_Under_Curve":  
        area_under_curve = 0
        for i in range(len(recall_levels) - 1):
            recall_start, recall_end = recall_levels[i], recall_levels[i + 1]
            
            # Find the max precision in this recall range
            valid_precisions = AP_df.loc[
                (AP_df["Recall"] >= recall_start) & (AP_df["Recall"] < recall_end),
                "Precision"
            ]
            max_precision = valid_precisions.max() if not valid_precisions.empty else 0
            
            # Multiply by the recall step size to approximate area under curve
            area_under_curve += max_precision * (recall_end - recall_start)
        
        AP = area_under_curve
    else:
        raise ValueError("Invalid method. Choose either 'Interpolation' or 'Area_Under_Curve'.")
    
    return AP , AP_df