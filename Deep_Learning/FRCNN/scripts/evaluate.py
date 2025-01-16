import numpy as np
import json
from collections import defaultdict

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = intersection_x * intersection_y

    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def calculate_map(detections, ground_truth, iou_thresholds, class_id=None):
    tp_fp = defaultdict(lambda: defaultdict(list))

    for annotation in ground_truth['annotations']:
        if class_id is not None and annotation['category_id'] != class_id:
            continue

        image_id = annotation['image_id']
        image_name = next((img['file_name'] for img in ground_truth['images'] if img['id'] == image_id), None)
        category_id = annotation['category_id']
        gt_bbox = annotation['bbox']

        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]]

        for detection in detections:
            if detection['image_id'] == image_name:
                det_bbox = detection['bbox']
                det_category_id = detection['category_id']
                det_score = detection['score']
                if class_id is not None and det_category_id != class_id:
                    continue    
                det_bbox = [det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3]]
                # print(f'Image:{image_name}')
                # print(f'GT:{gt_bbox}')
                # print(f'PR:{det_bbox}')
                iou = calculate_iou(det_bbox, gt_bbox)

                for iou_threshold in iou_thresholds:
                    if iou >= iou_threshold and det_category_id == category_id:
                        tp_fp[category_id][iou_threshold].append((det_score, True))
                    else:
                        tp_fp[category_id][iou_threshold].append((det_score, False))

    map_values = {}
    for category_id in tp_fp:
        for iou_threshold in tp_fp[category_id]:
            tp_fp_list = tp_fp[category_id][iou_threshold]
            tp_fp_list.sort(key=lambda x: x[0], reverse=True)

            true_positives = 0
            false_positives = 0
            precision = []
            recall = []

            for score, is_tp in tp_fp_list:
                if is_tp:
                    true_positives += 1
                else:
                    false_positives += 1
                precision.append(true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0)
                recall.append(true_positives / len([a for a in ground_truth['annotations'] if a['category_id'] == category_id]) if len([a for a in ground_truth['annotations'] if a['category_id'] == category_id]) != 0 else 0)

            ap = 0
            for i in range(1, len(precision)):
                ap += (recall[i] - recall[i-1]) * precision[i]
            map_values[f'{category_id}_{iou_threshold}'] = ap

    mAP50 = np.mean([map_values[f'{category_id}_0.5'] for category_id in tp_fp if f'{category_id}_0.5' in map_values])
    mAP5095 = np.mean([map_values[f'{category_id}_{iou_threshold}'] for category_id in tp_fp for iou_threshold in iou_thresholds if f'{category_id}_{iou_threshold}' in map_values])

    return mAP50 if not np.isnan(mAP50) else 0, mAP5095 if not np.isnan(mAP5095) else 0

def calculate_precision_recall_f1(detections, ground_truth, class_id=None):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for annotation in ground_truth['annotations']:
        if class_id is not None and annotation['category_id'] != class_id:
            continue

        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_name = next((img['file_name'] for img in ground_truth['images'] if img['id'] == image_id), None)
        gt_bbox = annotation['bbox']

        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]]

        for detection in detections:
            if detection['image_id'] == image_name:
                det_bbox = detection['bbox']
                det_category_id = detection['category_id']
                if class_id is not None and det_category_id != class_id:
                    continue 

                det_bbox = [det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3]]
                iou = calculate_iou(det_bbox, gt_bbox)

                if iou >= 0 and det_category_id == category_id:
                    true_positives += 1
                    # print('True')
                else:
                    false_positives += 1

        if not any(detection['image_id'] == image_id and detection['category_id'] == category_id for detection in detections):
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():
    detections_file_path = '/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/FRCNN/outputs/inference_results_clean/results.json'
    ground_truth_file_path = '/home/omar/Masters/Adv Image and Video Processiong/Project/Dataset/fcrnn_split_data_new/test/test_annotations.json'


    detections = load_json_file(detections_file_path)
    ground_truth = load_json_file(ground_truth_file_path)

    class_id = int(input("Enter the class ID to calculate metrics for (or press Enter for all classes): ") or -1)
    class_id = class_id if class_id != -1 else None

    precision, recall, f1 = calculate_precision_recall_f1(detections, ground_truth, class_id)
    mAP50, mAP5095 = calculate_map(detections, ground_truth, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], class_id)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1:.4f}')
    print(f'mAP@50: {mAP50:.4f}')
    print(f'mAP@50-95: {mAP5095:.4f}')

if __name__ == '__main__':
    main()
