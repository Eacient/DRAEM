import numpy as np
from skimage import measure

class Detection():
    def __init__(self, prob,  x1, y1, x2, y2, index=None, label=None,):
        self.label = label
        self.prob = prob
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

def post_process(score_map, bin_thresh=0.5, pos_thresh=0.3):
    # 二值化
    binary_mask = score_map > bin_thresh

    # 使用形态学操作将二值掩码转换为实体边界框
    labeled_image, num_objects = measure.label(binary_mask, connectivity=2, return_num=True)
    regions = measure.regionprops(labeled_image)
    bounding_boxes = [list(region.bbox) for region in regions]

    # 计算每个bounding box的置信度，等于scoremap*mask之后，在框内区域的有效平均值
    confidences = []
    for i, region in enumerate(regions):
        mask = labeled_image == i + 1
        masked_scores = score_map * mask
        masked_scores_sum = np.sum(masked_scores)
        masked_area = np.sum(mask)
        if masked_area > 0:
            confidence = masked_scores_sum / masked_area
            confidences.append(confidence)
        else:
            confidences.append(0.0)

    # # 根据置信度和pos_thresh选出阳性框
    positive_confidences = [confidence for confidence in confidences if confidence > pos_thresh]
    positive_boxes = [box for box, confidence in zip(bounding_boxes, confidences) if confidence > pos_thresh]
    return positive_confidences, positive_boxes

# 用法示例
score_map = np.random.rand(100, 100)  # 示例随机分数图
positive_boxes = post_process(score_map, bin_thresh=0.5, pos_thresh=0.3)
print("Positive Bounding Boxes:", positive_boxes)