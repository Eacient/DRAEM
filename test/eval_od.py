import numpy as np
import glob
import cv2
import os
from skimage import measure
from PIL import Image
import pandas
import torch
from torchvision.utils import make_grid


def visualize_gray(input_array, max_value=10000):
    input_array[input_array > max_value] = max_value
    return (input_array / max_value * 255).astype(np.uint8)

def convert_saved_box(box, height, width):
    x1, y1, x2, y2 = box
    return int(x1*height), int(y1*width), int(x2*height), int(y2*width)

def calculate_iou(gt_box, pred_box):
    ix1 = max(gt_box[0], pred_box[0])
    iy1 = max(gt_box[1], pred_box[1])
    ix2 = min(gt_box[2], pred_box[2])
    iy2 = min(gt_box[3], pred_box[3])

    overlap_h, overlap_w = ix2 - ix1, iy2 - iy1
    if overlap_h < 0 or overlap_w < 0:
        return 0
    gt_h, gt_w = gt_box[2] - gt_box[0],  gt_box[3] - gt_box[1]
    pred_h, pred_w = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]

    return overlap_h * overlap_w / (gt_h*gt_w + pred_h*pred_w - overlap_h*overlap_w)

# 计算相交面积相对于预测框的比例
def calculate_overlap(gt_box, pred_box):
    pred_h, pred_w = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
    ix1 = max(gt_box[0], pred_box[0])
    iy1 = max(gt_box[1], pred_box[1])
    ix2 = min(gt_box[2], pred_box[2])
    iy2 = min(gt_box[3], pred_box[3])
    overlap_h, overlap_w = ix2 - ix1, iy2 - iy1
    if overlap_h <= 0 or overlap_w <= 0:
        return 0
    return overlap_h * overlap_w / (pred_h * pred_w)


# 得到每个bbox的置信值, 有最大值和平均值两种计算模式, 如果为mean, threshold与二值化时保持一致
def calculate_confidence(score_map:np.ndarray, bboxes, mode='max', threshold=0.2):
    confidences = []
    if mode == 'max':
        for bbox in bboxes:
            min_x, min_y, max_x, max_y = convert_saved_box(bbox, *score_map.shape)
            confidences.append(np.max(score_map[min_x:max_x, min_y:max_y]))
        return confidences
    elif mode == 'mean':
        binary_mask = np.where(score_map > threshold, 1, 0)
        score_map *= binary_mask
        for bbox in bboxes:
            min_x, min_y, max_x, max_y = convert_saved_box(bbox, *score_map.shape)
            confidences.append(np.sum(score_map[min_x:max_x, min_y:max_y])/ 
                               np.sum(binary_mask[min_x:max_x, min_y:max_y]))
        return confidences

# 可选, 将存在重叠的bbox合并, padding允许将两个邻近的bbox合并
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

# 从score_map获得bbox, 并根据图片长宽缩放到[0, 1]空间
def get_normalized_bboxes(score_map:np.ndarray, threshold):
    print(score_map.shape)
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

# 读取标签文件夹, 转化成读取更快的dict{name: []bboxes}
def convert_gt_to_dict(gt_dir, output_path):
    gt_dict = {}
    for name in os.listdir(gt_dir):
        p = os.path.join(gt_dir, name)
        score_map = np.array(Image.open(p))
        gt_dict[os.path.splitext(name)[0]] = get_normalized_bboxes(score_map, 128)
    np.save(output_path, gt_dict)
    return gt_dict

"""读取预测文件夹转化成dict
{
    name: {
        'bbox': []bboxes
        'confidence': []confidences
    }
}, 
其中预测文件夹中格式应该为{图像名}.npy, 为图像预测的规范化到[0,1)空间的score_map, 
"""
def convert_pred_to_dict(scoremap_dir, tmp_path, threshold=0.2, confidence_mode='max'):
    pred_dict = {}
    for name in os.listdir(scoremap_dir):
        p = os.path.join(scoremap_dir, name)
        score_map = np.load(p)
        bboxes= merge_overlapping_boxes(get_normalized_bboxes(score_map, threshold))
        confidences = calculate_confidence(score_map, bboxes, mode=confidence_mode)
        # confidences = calculate_confidence(score_map, bboxes, mode='mean', threshold)
        pred_dict[os.path.splitext(name)[0]] = {
            'bbox': bboxes,
            'confidence': confidences,
        }
    np.save(tmp_path, pred_dict)
    return pred_dict

# 计算tp fn fp tn jaccard or overlap
def calculate_metric(gt_boxes, positive_boxes, tp_mode, tp_thresh):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    if len(gt_boxes) == 0:
        false_positives = len(positive_boxes)
        true_negatives =  1 if false_positives == 0 else 0
        return true_positives, false_positives, false_negatives, true_negatives

    matched = set()
    for true_box in gt_boxes:
        if tp_mode == 'jaccard':
            tp_values = [calculate_iou(true_box, pred_box) for pred_box in positive_boxes]
        elif tp_mode == 'overlap':
            tp_values = [calculate_overlap(true_box, pred_box) for pred_box in positive_boxes]
        max_tp = max(tp_values) if tp_values else 0
        if max_tp > tp_thresh:
            true_positives += 1
            indexes = [i for i in range(len(tp_values)) if tp_values[i] > tp_thresh]
            for index in indexes:
                if index not in matched:
                    matched.add(index)
        else:
            false_negatives += 1
    
    false_positives = len(positive_boxes) - len(matched)
    
    return true_positives, false_positives, false_negatives, true_negatives

# 计算并保存dict{name: [tp, fp, fn, tn]}
def calculate_metric_dict(gt_dict, pred_dict, positive_thresh, tp_mode, tp_thresh, output_path):
    metric_dict = {}
    ok_count = 0
    tot_tp, tot_fp, tot_fn, tot_tn = 0, 0, 0, 0
    for name, pred in pred_dict.items():
        pred_boxes = pred['bbox']
        try:
            # print(tot_tn, name)
            gt_boxes = gt_dict[name]
        except:
            # print(tot_tn)
            print(f'calculate metric using blank gt for {name}')
            ok_count += 1
            gt_boxes = []
        pred_confidences = pred['confidence']
        positive_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if pred_confidences[i] > positive_thresh]
        tp, fp, fn, tn = calculate_metric(gt_boxes, positive_boxes, tp_mode, tp_thresh)
        metric_dict[name] = [tp, fp, fn, tn]
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_tn += tn
    metric_dict['total'] = [tot_tp, tot_fp, tot_fn, tot_tn]
    np.save(output_path, metric_dict)
    print(ok_count)
    # assert ok_count == 50
    return metric_dict

def eval_od(root_dir, pred_dir, bin_thresh=0.2, confidence_mode='max', pos_thresh=0.2, tp_mode='overlap',  tp_thresh=0.1):

    gt_dir = os.path.join(root_dir, 'mask')
    tmp_dir = os.path.join(pred_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # converting gt_masks
    gt_dict_path = os.path.join(tmp_dir, 'gt_dict.npy')
    if not os.path.exists(gt_dict_path):
        print(f'converting gt_dict to {gt_dict_path}......')
        gt_dict = convert_gt_to_dict(gt_dir, gt_dict_path)
    else:
        print(f'loading gt_dict from {gt_dict_path}......')
        gt_dict = np.load(gt_dict_path, allow_pickle=True).item()

    # convert score_maps
    pred_dict_path = os.path.join(tmp_dir, f'pred_dict_{bin_thresh:.2f}_{confidence_mode}.npy')
    scoremap_dir = os.path.join(pred_dir, 'score_maps')
    if not os.path.exists(pred_dict_path):
        print(f'converting pred_dict to {pred_dict_path}......')
        pred_dict = convert_pred_to_dict(scoremap_dir, pred_dict_path, bin_thresh, confidence_mode)
    else:
        print(f'loading pred_dict to {pred_dict_path}......')
        pred_dict = np.load(pred_dict_path, allow_pickle=True).item()

    # calculate metric
    metric_dict_path = os.path.join(tmp_dir, 
        f'metric_dict_{bin_thresh:.2f}_{confidence_mode}_{pos_thresh:.2f}'
        f'_{tp_mode}_{tp_thresh:.2f}.npy')
    if not os.path.exists(metric_dict_path):
        print(f'computing metrics and writing to {metric_dict_path}......')
        metric_dict = calculate_metric_dict(gt_dict, pred_dict, 
                        pos_thresh, tp_mode, tp_thresh, metric_dict_path)
    else:
        print(f'loading metrics from {metric_dict_path}......')
        metric_dict = np.load(metric_dict_path, allow_pickle=True).item()
    # tot_tp, tot_fp, tot_fn, tot_tn = metric_dict['total']
    print('tot_tp={0}, tot_fp={1}, tot_fn={2}, tot_tn={3}'.format(*(metric_dict['total'])))
    return gt_dict, pred_dict, metric_dict

# 可视化预测框
def visualize(root_dir, pred_dir, gt_dict, pred_dict, metric_dict, positive_thresh, vis_max, compose_output=True):
    output_dir = os.path.join(pred_dir, 'visualize')
    print(f'writing visualize results to {output_dir}......')
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(root_dir+'/ok/*.npy') + glob.glob(root_dir+'/ng/*.npy')
    compose_list = []
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        score_map = np.load(os.path.join(pred_dir, 'score_maps', base_name+'.npy'))
        heatmap = cv2.applyColorMap((score_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        img_gray = visualize_gray(np.load(img_path), vis_max)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlaid_image = img_rgb * 0.7 + heatmap * 0.3

        try:
            gt_boxes = gt_dict[base_name]
        except:
            gt_boxes = []
        pred_boxes = pred_dict[base_name]
        metrics = metric_dict[base_name]
        for box, confidence in zip(pred_boxes['bbox'], pred_boxes['confidence']):
            x1, y1, x2, y2 = convert_saved_box(box, *score_map.shape)
            confidence_text = f"{confidence:.2f}"
            if confidence > positive_thresh:
                cv2.putText(overlaid_image, confidence_text, (y1-25, x1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.rectangle(overlaid_image, (y1, x1), (y2, x2), (0, 255, 0), 1)
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = convert_saved_box(gt_box, *score_map.shape)
            cv2.rectangle(overlaid_image, (y1, x1), (y2, x2), (255, 0, 0), 1)
        tp, fp, fn, tn = metrics
        text_lines = [f'pred:{len(pred_boxes["bbox"])}; gt:{len(gt_boxes)}', f'threshold:{positive_thresh:.2f}', f'TP: {tp}', f'FP: {fp}', f'FN: {fn}', f'TN: {tn}']
        for i, line in enumerate(text_lines):
            cv2.putText(overlaid_image, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        output_path = os.path.join(output_dir, base_name+'.png')
        cv2.imwrite(output_path, overlaid_image)
        if compose_output:
            compose_list.append(torch.tensor(overlaid_image).permute(2, 0, 1))
    if compose_output:
        tensorlist = torch.stack(compose_list) / 255
        print(tensorlist.shape)
        img_grid = make_grid(tensorlist, nrow=10)
        cv2.imwrite(os.path.join(pred_dir, 'compose.png'), (img_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', action='store', type=str, required=True)
    parser.add_argument('--pred_dir', action='store', type=str, required=True)
    parser.add_argument('--bin_thresh', action='store', type=float, required=True)
    parser.add_argument('--confidence_mode', action='store', type=str, required=True)
    parser.add_argument('--pos_thresh', action='store', type=float, required=True)
    parser.add_argument('--tp_mode', action='store', type=str, required=True)
    parser.add_argument('--tp_thresh', action='store', type=float, required=True)
    parser.add_argument('--vis_max', action='store', type=int, required=True)

    args = parser.parse_args()

    gt_dict, pred_dict, metric_dict = eval_od(args.root_dir, args.pred_dir, args.bin_thresh, 
                                              args.confidence_mode, args.pos_thresh, args.tp_mode, args.tp_thresh)
    visualize(args.root_dir, args.pred_dir, gt_dict, pred_dict, metric_dict, args.tp_thresh, args.vis_max)