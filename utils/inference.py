import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import yaml
from collections import defaultdict

from nets import nn
from utils import util

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and evaluate metrics on dataset")
    parser.add_argument('--weights', type=str, default='/home/ssl41/cs21b048_37_dl/YOLOv8-pt/report_fl_server_results/soda10m_iid_server_fedavg_20250513_223438/global_model_round_30.pt', help='model weights path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--dataset-path', type=str, default='/home/ssl41/cs21b048_37_dl/datasets/soda10m', 
                        help='dataset path')
    parser.add_argument('--save-dir', type=str, default='./output2', help='directory to save results')
    parser.add_argument('--save-img', action='store_true', help='save images with detections')
    parser.add_argument('--eval-split', type=str, default='val', choices=['train', 'val', 'test'], 
                        help='dataset split to evaluate')
    return parser.parse_args()

def load_model(weights_path, device, num_classes=None):
    print(f"Loading model from {weights_path}...")
    
    # Load the checkpoint first to determine model parameters
    ckpt = torch.load(weights_path, map_location=device)
    
    # If the checkpoint includes args, we can use that to determine model size
    model_size = 'n'  # Default
    if 'args' in ckpt and hasattr(ckpt['args'], 'model_size'):
        model_size = ckpt['args'].model_size
    
    # Initialize model based on size
    if model_size == 'n':
        model = nn.yolo_v8_n(num_classes)
    elif model_size == 's':
        model = nn.yolo_v8_s(num_classes)
    elif model_size == 'm':
        model = nn.yolo_v8_m(num_classes)
    elif model_size == 'l':
        model = nn.yolo_v8_l(num_classes)
    elif model_size == 'x':
        model = nn.yolo_v8_x(num_classes)
    else:
        print(f"Unknown model size: {model_size}, defaulting to nano")
        model = nn.yolo_v8_n(num_classes)
    
    # Load the state dict
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.float().eval()
    return model

def get_class_names(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    return names

def resize_image(image, target_size):
    height, width = image.shape[:2]
    scale = min(target_size / height, target_size / width)
    
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_height, :new_width, :] = resized
    
    return canvas, scale, (height, width, new_height, new_width)

def draw_detections(image, detections, class_names, color=(0, 255, 0)):
    img_copy = image.copy()
    for detection in detections:
        x1, y1, x2, y2, conf, cls_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        cls_name = class_names.get(cls_id, f"Class {cls_id}")
        label = f"{cls_name}: {conf:.2f}"
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img_copy

def run_inference(model, image_path, img_size, conf_thres, iou_thres):
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error loading image: {image_path}")
        return None, None, None
    
    img, scale, shape_info = resize_image(img_orig, img_size)
    orig_h, orig_w, new_h, new_w = shape_info
    
    img_tensor = torch.from_numpy(img).float().cuda()
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # to [C, H, W] and normalize
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension [1, C, H, W]
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    detections = util.non_max_suppression(outputs, conf_thres, iou_thres)
    detections = detections[0]  # Get first image in batch
    
    if detections is not None and len(detections):
        detections = detections.cpu().numpy()
        
        detections[:, 0] /= scale
        detections[:, 1] /= scale
        detections[:, 2] /= scale 
        detections[:, 3] /= scale
        
        detections[:, 0] = np.clip(detections[:, 0], 0, orig_w)
        detections[:, 1] = np.clip(detections[:, 1], 0, orig_h)
        detections[:, 2] = np.clip(detections[:, 2], 0, orig_w)
        detections[:, 3] = np.clip(detections[:, 3], 0, orig_h)
    
    return img_orig, detections, shape_info

def parse_label_file(label_path, img_width, img_height):
    """Parse YOLO format label file and convert to absolute coordinates"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:  # class, x_center, y_center, width, height
                cls_id = int(data[0])
                x_center = float(data[1]) * img_width
                y_center = float(data[2]) * img_height
                width = float(data[3]) * img_width
                height = float(data[4]) * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                labels.append([cls_id, x1, y1, x2, y2])
    
    return np.array(labels)

def evaluate_detections(detections, ground_truth, num_classes, iou_thresholds):
    """Evaluate detections against ground truth to calculate precision, recall, mAP"""
    stats = []
    
    class_metrics = [[] for _ in range(num_classes)]
    
    if detections is not None and len(detections) > 0:
        confidence = detections[:, 4]
        pred_cls = detections[:, 5].astype(int)
        pred_boxes = detections[:, :4]  
        
        if len(ground_truth) > 0:
            gt_cls = ground_truth[:, 0].astype(int)
            gt_boxes = ground_truth[:, 1:5]
            
            correct = np.zeros((len(detections), len(iou_thresholds)), dtype=bool)
            
            class_correct = [np.zeros((len(detections), len(iou_thresholds)), dtype=bool) 
                            for _ in range(num_classes)]
            
            for i, gt_box in enumerate(gt_boxes):
                gt_c = gt_cls[i]
                
                iou = box_iou_numpy(gt_box[np.newaxis, :], pred_boxes)

                for j, iou_threshold in enumerate(iou_thresholds):
                    matches = (iou >= iou_threshold) & (pred_cls == gt_c)
                    if matches.any():
                        max_iou_idx = np.argmax(iou.flatten())
                        if matches.flatten()[max_iou_idx]:
                            correct[max_iou_idx, j] = True
                            
                            class_correct[gt_c][max_iou_idx, j] = True

            stats.append((correct, confidence, pred_cls, gt_cls))
            for cls in range(num_classes):
                cls_mask = pred_cls == cls
                if not cls_mask.any():
                    if cls in gt_cls:
                        class_metrics[cls].append((
                            np.zeros((0, len(iou_thresholds)), dtype=bool),
                            np.array([]),
                            np.array([]),
                            gt_cls[gt_cls == cls]
                        ))
                    continue
                
                cls_correct_detections = class_correct[cls][cls_mask]
                cls_confidence = confidence[cls_mask]
                cls_pred = pred_cls[cls_mask]
                
                cls_gt = gt_cls[gt_cls == cls] if cls in gt_cls else np.array([])    
                class_metrics[cls].append((cls_correct_detections, cls_confidence, cls_pred, cls_gt))
        
        else:
            stats.append((np.zeros((len(detections), len(iou_thresholds)), dtype=bool), 
                         confidence, pred_cls, np.array([])))
            
            for cls in range(num_classes):
                cls_mask = pred_cls == cls
                if not cls_mask.any():
                    continue
                
                class_metrics[cls].append((
                    np.zeros((np.sum(cls_mask), len(iou_thresholds)), dtype=bool),
                    confidence[cls_mask],
                    pred_cls[cls_mask],
                    np.array([])
                ))
    
    elif len(ground_truth) > 0:
        # No detections but ground truths exist - all false negatives
        # Empty arrays for correct, conf, pred_cls
        stats.append((np.zeros((0, len(iou_thresholds)), dtype=bool), 
                     np.array([]), np.array([]), ground_truth[:, 0]))
        
        for cls in range(num_classes):
            cls_gt = ground_truth[ground_truth[:, 0] == cls, 0]
            if len(cls_gt) > 0:
                class_metrics[cls].append((
                    np.zeros((0, len(iou_thresholds)), dtype=bool),
                    np.array([]),
                    np.array([]),
                    cls_gt
                ))
    
    return stats, class_metrics

def box_iou_numpy(box1, box2):
    """Calculate IoU between box1 and box2 (numpy implementation)"""
    # Box areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Intersection coordinates
    xx1 = np.maximum(box1[:, None, 0], box2[:, 0])  # max of x1s
    yy1 = np.maximum(box1[:, None, 1], box2[:, 1])  # max of y1s
    xx2 = np.minimum(box1[:, None, 2], box2[:, 2])  # min of x2s
    yy2 = np.minimum(box1[:, None, 3], box2[:, 3])  # min of y2s
    
    # Intersection area
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    intersection = w * h
    
    # Union area
    union = area1[:, None] + area2 - intersection
    
    # IoU
    iou = intersection / (union + 1e-16)
    
    return iou

def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap = ap + p / 11.0
    return ap

def calculate_metrics(all_stats, all_class_metrics, num_classes, iou_thresholds):
    """Calculate mAP and class-wise precision from accumulated statistics"""
    # Process overall metrics
    overall_metrics = {}
    
    # Concatenate stats from all images
    if all_stats:
        # List of (correct, confidence, pred_cls, gt_cls) tuples
        all_correct = np.concatenate([x[0] for x in all_stats if x[0].size > 0], axis=0) if any(x[0].size > 0 for x in all_stats) else np.zeros((0, len(iou_thresholds)), dtype=bool)
        all_conf = np.concatenate([x[1] for x in all_stats if x[1].size > 0], axis=0) if any(x[1].size > 0 for x in all_stats) else np.array([])
        all_pred_cls = np.concatenate([x[2] for x in all_stats if x[2].size > 0], axis=0) if any(x[2].size > 0 for x in all_stats) else np.array([], dtype=int)
        all_gt_cls = np.concatenate([x[3] for x in all_stats if x[3].size > 0], axis=0) if any(x[3].size > 0 for x in all_stats) else np.array([], dtype=int)
        
        if len(all_conf) > 0:
            # Sort by confidence
            sorted_ind = np.argsort(-all_conf)
            all_correct = all_correct[sorted_ind]
            all_conf = all_conf[sorted_ind]
            all_pred_cls = all_pred_cls[sorted_ind]
            
            # Compute TP and FP for each IoU threshold
            overall_metrics['mAP'] = []
            overall_metrics['precision'] = []
            overall_metrics['recall'] = []
            
            for i, iou_thresh in enumerate(iou_thresholds):
                tp = np.cumsum(all_correct[:, i])
                fp = np.cumsum(~all_correct[:, i])
                
                # Precision and recall
                precision = tp / (tp + fp + 1e-16)
                recall = tp / (len(all_gt_cls) + 1e-16)
                
                # Average precision
                ap = compute_ap(recall, precision)
                
                overall_metrics['mAP'].append(ap)
                overall_metrics['precision'].append(precision[-1] if len(precision) > 0 else 0)
                overall_metrics['recall'].append(recall[-1] if len(recall) > 0 else 0)
            
            # Calculate mAP50 (index 0) and mAP50-95 (mean of all)
            overall_metrics['mAP50'] = overall_metrics['mAP'][0]
            overall_metrics['mAP50-95'] = np.mean(overall_metrics['mAP'])
        else:
            # No detections
            overall_metrics['mAP50'] = 0
            overall_metrics['mAP50-95'] = 0
            overall_metrics['precision'] = [0]
            overall_metrics['recall'] = [0]
            overall_metrics['mAP'] = [0] * len(iou_thresholds)
    else:
        # No images evaluated
        overall_metrics['mAP50'] = 0
        overall_metrics['mAP50-95'] = 0
        overall_metrics['precision'] = [0]
        overall_metrics['recall'] = [0]
        overall_metrics['mAP'] = [0] * len(iou_thresholds)
    
    # Calculate class-wise precision (at IoU=0.5)
    class_precision = []
    
    for cls in range(num_classes):
        cls_metrics = all_class_metrics[cls]
        if not cls_metrics:
            class_precision.append(0.0)
            continue
        
        # Concatenate class metrics from all images
        cls_correct = np.concatenate([x[0] for x in cls_metrics if x[0].size > 0], axis=0) if any(x[0].size > 0 for x in cls_metrics) else np.zeros((0, len(iou_thresholds)), dtype=bool)
        cls_conf = np.concatenate([x[1] for x in cls_metrics if x[1].size > 0], axis=0) if any(x[1].size > 0 for x in cls_metrics) else np.array([])
        cls_pred = np.concatenate([x[2] for x in cls_metrics if x[2].size > 0], axis=0) if any(x[2].size > 0 for x in cls_metrics) else np.array([])
        cls_gt = np.concatenate([x[3] for x in cls_metrics if x[3].size > 0], axis=0) if any(x[3].size > 0 for x in cls_metrics) else np.array([])
        
        if len(cls_conf) > 0:
            # Sort by confidence
            sorted_ind = np.argsort(-cls_conf)
            cls_correct = cls_correct[sorted_ind]
            
            # Calculate TP and FP for IoU=0.5 (index 0)
            tp = np.cumsum(cls_correct[:, 0])
            fp = np.cumsum(~cls_correct[:, 0])
            
            # Calculate precision
            precision = tp / (tp + fp + 1e-16)
            
            # Final precision is the last value
            cls_precision = precision[-1] if len(precision) > 0 else 0.0
        else:
            cls_precision = 0.0
        
        class_precision.append(cls_precision)
    
    return overall_metrics, class_precision

def get_label_path_from_image_path(image_path, dataset_path):
    """Convert image path to corresponding label path"""
    # Extract relative path from dataset_path
    images_dir = os.path.join(dataset_path, 'images')
    rel_path = os.path.relpath(image_path, images_dir)
    
    # Replace extension with .txt
    base_path, _ = os.path.splitext(rel_path)
    
    # Construct label path
    label_path = os.path.join(dataset_path, 'labels', base_path + '.txt')
    
    return label_path

def main():
    args = parse_args()
    
    # Create save directory if needed
    if args.save_img and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Save metrics directory
    metrics_dir = os.path.join(args.save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load class names
    data_yaml_path = os.path.join(args.dataset_path, 'data.yaml')
    class_names = get_class_names(data_yaml_path)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {', '.join(class_names.values())}")
    
    # Load model
    model = load_model(args.weights, device, num_classes)
    model = model.to(device)
    
    # Find images for evaluation
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    images_dir = os.path.join(args.dataset_path, 'images')
    eval_dir = os.path.join(images_dir, args.eval_split)
    
    if os.path.exists(eval_dir):
        for root, _, files in os.walk(eval_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        print(f"Warning: {eval_dir} not found, checking all images directory")
        for root, _, files in os.walk(images_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images for evaluation in {args.eval_split} split")
    
    # Evaluate with different IoU thresholds
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, 0.6, ..., 0.95]
    
    # Initialize metrics storage
    all_stats = []
    all_class_metrics = [[] for _ in range(num_classes)]
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Running inference and evaluation"):
        # Run inference
        original_image, detections, shape_info = run_inference(
            model, image_path, args.img_size, args.conf_thres, args.iou_thres
        )
        
        if original_image is None:
            continue
        
        # Get corresponding label file
        label_path = get_label_path_from_image_path(image_path, args.dataset_path)
        
        # Get image dimensions for label conversion
        orig_h, orig_w = shape_info[0], shape_info[1]
        
        # Parse label file to get ground truth
        ground_truth = parse_label_file(label_path, orig_w, orig_h)
        
        # Save visualization if requested
        if args.save_img and detections is not None and len(detections) > 0:
            result_image = draw_detections(original_image, detections, class_names)
            output_path = os.path.join(args.save_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, result_image)
        
        # Evaluate detections against ground truth
        stats, class_metrics = evaluate_detections(detections, ground_truth, num_classes, iou_thresholds)
        
        # Accumulate metrics
        all_stats.extend(stats)
        for cls in range(num_classes):
            all_class_metrics[cls].extend(class_metrics[cls])
    
    # Calculate final metrics
    overall_metrics, class_precision = calculate_metrics(all_stats, all_class_metrics, num_classes, iou_thresholds)
    
    # Print overall metrics
    print("\n===== Evaluation Results =====")
    print(f"mAP@0.5: {overall_metrics['mAP50']:.4f}")
    print(f"mAP@0.5-0.95: {overall_metrics['mAP50-95']:.4f}")
    
    # Print class-wise precision
    print("\nClass-wise Precision (IoU=0.5):")
    for i, precision in enumerate(class_precision):
        print(f"  {class_names[i]}: {precision:.4f}")
    
    # Save metrics to files
    metrics_file = os.path.join(metrics_dir, f'metrics_{args.eval_split}.txt')
    with open(metrics_file, 'w') as f:
        f.write("===== Evaluation Results =====\n")
        f.write(f"mAP@0.5: {overall_metrics['mAP50']:.4f}\n")
        f.write(f"mAP@0.5-0.95: {overall_metrics['mAP50-95']:.4f}\n\n")
        
        f.write("Class-wise Precision (IoU=0.5):\n")
        for i, precision in enumerate(class_precision):
            f.write(f"  {class_names[i]}: {precision:.4f}\n")
    
    detailed_metrics = {
        'mAP50': float(overall_metrics['mAP50']),
        'mAP50-95': float(overall_metrics['mAP50-95']),
        'class_precision': {class_names[i]: float(precision) for i, precision in enumerate(class_precision)},
        'iou_thresholds': iou_thresholds.tolist(),
        'mAP_at_IoU': {f'IoU_{iou:.2f}': float(ap) for iou, ap in zip(iou_thresholds, overall_metrics['mAP'])}
    }
    
    yaml_file = os.path.join(metrics_dir, f'detailed_metrics_{args.eval_split}.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(detailed_metrics, f, default_flow_style=False)
    
    print(f"\nMetrics saved to {metrics_file} and {yaml_file}")

if __name__ == '__main__':
    main()