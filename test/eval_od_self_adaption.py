# 自适应的阈值指的是在一个合理的bin_thresh的基础上，自适应地调节提高pos_thresh，直到没有任何一个假阳性
import os
import glob
import cv2
import torch
from torchvision.utils import make_grid
import numpy as np
from eval_od import convert_gt_to_dict, convert_pred_to_dict, calculate_metric, visualize_gray, convert_saved_box, calculate_overlap, calculate_iou

def binary_search_pos_thresh(pred_boxes, pred_confidences, gt_boxes, tp_mode, tp_thresh, search_gap=0.01):
    tp_func = calculate_iou if tp_mode =='iou' else calculate_overlap
    search_threshs = [search_gap * i for i in range(int(1 / search_gap))]
    l = 0
    r = len(search_threshs)
    steps = 0
    while l < r:
        mid = (l + r) // 2
        pos_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if pred_confidences[i] > search_threshs[mid]]
        fp = 0
        for pos_box in pos_boxes:
            matched=False
            for gt_box in gt_boxes:
                if tp_func(pos_box, gt_box) > tp_thresh:
                    matched = True
                    break
            if not matched:
                fp += 1
                break
        if fp > 0:
            l = mid + 1
        else:
            r = mid
        steps += 1
        if steps > 10:
            print('bs error')
            break
    try:
        return search_threshs[l]
    except:
        print('out of range')
        return search_threshs[l-1]

def calculate_metric_dict_self_adaption(gt_dict, pred_dict, tp_mode, tp_thresh, output_path):
    metric_dict = {}
    ok_count = 0
    tot_tp, tot_fp, tot_fn, tot_tn = 0, 0, 0, 0
    for name, pred in pred_dict.items():
        pred_boxes = pred['bbox']
        try:
            gt_boxes = gt_dict[name]
        except:
            # print(tot_tn)
            print(f'calculate metric using blank gt for {name}')
            ok_count += 1
            gt_boxes = []
        pred_confidences = pred['confidence']
        pos_thresh = binary_search_pos_thresh(pred_boxes, pred_confidences, gt_boxes, tp_mode, tp_thresh)
        pos_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if pred_confidences[i] > pos_thresh]
        tp, fp, fn, tn = calculate_metric(gt_boxes, pos_boxes, tp_mode, tp_thresh)
        metric_dict[name] = [tp, fp, fn, tn, pos_thresh]
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_tn += tn
    metric_dict['total'] = [tot_tp, tot_fp, tot_fn, tot_tn]
    np.save(output_path, metric_dict)
    print(ok_count)
    # assert ok_count == 50
    return metric_dict

def eval_od_self_adaption(root_dir, pred_dir, bin_thresh=0.2, confidence_mode='max', tp_mode='overlap',  tp_thresh=0.1):

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

    # calculate metric in a self adaption way
    metric_dict_path = os.path.join(tmp_dir, 
        f'metric_dict_{bin_thresh:.2f}_{confidence_mode}_self-adapt'
        f'_{tp_mode}_{tp_thresh:.2f}.npy')
    if not os.path.exists(metric_dict_path):
        print(f'computing metrics and writing to {metric_dict_path}......')
        metric_dict = calculate_metric_dict_self_adaption(gt_dict, pred_dict, tp_mode, tp_thresh, metric_dict_path)
    else:
        print(f'loading metrics from {metric_dict_path}......')
        metric_dict = np.load(metric_dict_path, allow_pickle=True).item()
    # tot_tp, tot_fp, tot_fn, tot_tn = metric_dict['total']
    print('tot_tp={0}, tot_fp={1}, tot_fn={2}, tot_tn={3}'.format(*(metric_dict['total'])))
    return gt_dict, pred_dict, metric_dict

def visualize_self_adaption(root_dir, pred_dir, gt_dict, pred_dict, metric_dict, vis_max, compose_output=True):
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
        tp, fp, fn, tn, positive_thresh = metrics
        for box, confidence in zip(pred_boxes['bbox'], pred_boxes['confidence']):
            x1, y1, x2, y2 = convert_saved_box(box, *score_map.shape)
            confidence_text = f"{confidence:.2f}"
            if confidence > positive_thresh:
                cv2.putText(overlaid_image, confidence_text, (y1-25, x1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.rectangle(overlaid_image, (y1, x1), (y2, x2), (0, 255, 0), 1)
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = convert_saved_box(gt_box, *score_map.shape)
            cv2.rectangle(overlaid_image, (y1, x1), (y2, x2), (255, 0, 0), 1)
        text_lines = [f'pred:{len(pred_boxes["bbox"])}; gt:{len(gt_boxes)}', f'pos_thresh:{positive_thresh:.2f}', f'TP: {tp}', f'FP: {fp}', f'FN: {fn}', f'TN: {tn}']
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
    parser.add_argument('--tp_mode', action='store', type=str, required=True)
    parser.add_argument('--tp_thresh', action='store', type=float, required=True)
    parser.add_argument('--vis_max', action='store', type=int, required=True)

    args = parser.parse_args()

    gt_dict, pred_dict, metric_dict = eval_od_self_adaption(args.root_dir, args.pred_dir, args.bin_thresh, 
                                              args.confidence_mode, args.tp_mode, args.tp_thresh)
    visualize_self_adaption(args.root_dir, args.pred_dir, gt_dict, pred_dict, metric_dict, args.vis_max)