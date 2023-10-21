from skimage import measure
import numpy as np

def cal_overlap(gt_box, pred_box):
    pred_h, pred_w = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
    ix1 = max(gt_box[0], pred_box[0])
    iy1 = max(gt_box[1], pred_box[1])
    ix2 = min(gt_box[2], pred_box[2])
    iy2 = min(gt_box[3], pred_box[3])
    overlap_h, overlap_w = ix2 - ix1, iy2 - iy1
    if overlap_h <= 0 or overlap_w <= 0:
        return 0
    return overlap_h * overlap_w / (pred_h * pred_w)

def get_normalized_bboxes(score_map:np.ndarray, threshold):
    bboxes = []
    binary_mask = np.where(score_map > threshold , 1, 0)
    labeled_image, num_objects = measure.label(binary_mask, connectivity=1, return_num=True)
    regions = measure.regionprops(labeled_image)
    bounding_boxes = [list(region.bbox) for region in regions]
    height, width = score_map.shape
    for box in bounding_boxes:
        box[0] = box[0] / height
        box[1] = box[1] / width
        box[2] = box[2] / height
        box[3] = box[3] / width
        bboxes.append(box)
    return bboxes

def merge_overlapping_boxes(boxes, padding=2e-2):
    merged_boxes = []

    while len(boxes) > 0:
        current_box = boxes[0]
        boxes = boxes[1:]

        merged_box = current_box

        i = 0
        while i < len(boxes):
            box = boxes[i]

            min_x = min(merged_box[0], box[0])
            min_y = min(merged_box[1], box[1])
            max_x = max(merged_box[2], box[2])
            max_y = max(merged_box[3], box[3])

            merged_width = max_x - min_x
            merged_height = max_y - min_y

            if merged_width < (merged_box[2]-merged_box[0]) + (box[2] - box[0]) + padding \
                and merged_height < (merged_box[3]-merged_box[1])  + (box[3]-box[1]) + padding:
                merged_box = [min_x, min_y, max_x, max_y]
                boxes.pop(i)
            else:
                i += 1

        merged_boxes.append(merged_box)

    return merged_boxes

def from_gt_to_box(gt_mask):
    return get_normalized_bboxes(gt_mask, 128)

def from_pred_to_box(pred_mask, threshold=0.2):
    return merge_overlapping_boxes(get_normalized_bboxes(pred_mask, threshold))

def cal_conf(pred_mask, box):
    confs = []
    h, w = pred_mask.shape
    for b in box:
        confs.append(pred_mask[int(b[0]*h):int(b[2]*h), int(b[1]*w):int(b[3]*w)].max())
    return confs

def from_pred_to_positive_box(pred_mask, threshold=0.2, positive_thresh=0.5):
    box = from_pred_to_box(pred_mask, threshold)
    confs = cal_conf(pred_mask, box)
    return [box[i] for i in range(len(box)) if confs[i] >= positive_thresh]


def get_missing_boxes(gt_mask, pred_mask):
    gt_boxes = from_gt_to_box(gt_mask)
    pred_boxes = from_pred_to_positive_box(pred_mask)
    missing_boxes = []
    for gt_box in gt_boxes:
        matched = False
        for pred_box in pred_boxes:
            if cal_overlap(gt_box, pred_box) > 0:
                matched = True
                break
        if matched is False:
            missing_boxes.append(gt_box)
    return missing_boxes

def convert_saved_box(box, height, width):
    x1, y1, x2, y2 = box
    return int(x1*height), int(y1*width), int(x2*height), int(y2*width)