import glob
import os
import torch
import cv2
import numpy as np
from eval_utils import convert_saved_box
import random

from DRAEM.base.data_loader import MVTecDRAEMTrainDataset

def get_start_pos(entity_box, box):
    h1, w1, h2, w2 = box
    h_min, w_min, h_max, w_max = entity_box
    h_pos = random.randint(h_min, h_max-(h2-h1))
    w_pos = random.randint(w_min, w_max-(w2-w1))
    return (h_pos, w_pos)

def get_area_mean(image, box):
    h_min = min(0, box[0] - 3)
    w_min = min(0, box[1] - 3)
    h_max = max(image.shape[0], box[2]+3)
    w_max = max(image.shape[1], box[3]+3)
    return image[h_min:h_max, w_min:w_max].mean()

class MVTecDRAEMTrainDatasetMixUp(MVTecDRAEMTrainDataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        super().__init__(root_dir, anomaly_source_path, resize_shape)
        self.ok_paths = sorted(glob.glob(root_dir+"/train/good/*.jpg"))
        self.ng_paths = sorted(glob.glob(root_dir+"/train/impurity/*.jpg"))
        self.gt_root = os.path.join(root_dir, 'ground_truth', 'impurity')
        self.missing_objects = list(np.load(os.path.join(root_dir, 'missing.npy'), allow_pickle=True).item().items())
        self.entity_boxes = np.load(os.path.join(root_dir, 'entity.npy'), allow_pickle=True).item()
        self.len_mixup = len(self.missing_objects)
        self.len_ok = len(self.ok_paths)
        self.len_ng = len(self.ng_paths)

    def __len__(self):
        return self.len_ng
    
    def transform_image_sup(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def transform_image_mixup(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # # 随机选择异常来源并读取
        # mask_id = torch.randint(0, self.len_mixup, (1,)).item()
        # anomaly_name, anomaly_boxes = self.missing_objects[mask_id]
        # anomaly_path = os.path.join(self.root_dir, 'train', 'impurity', anomaly_name+'.jpg')
        # anomaly_path = anomaly_path if os.path.exists(anomaly_path) else os.path.join(self.root_dir, 'val', 'impurity', anomaly_name+'.jpg')
        # anomaly_image = cv2.imread(anomaly_path, cv2.IMREAD_COLOR)

        # resize and normalize
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0])) / 255.0
        # anomaly_image = cv2.resize(anomaly_image, dsize=(self.resize_shape[1], self.resize_shape[0])) / 255.0
        image = image.astype(np.float32)
        # anomaly_image = anomaly_image.astype(np.float32)

        entity_box = self.entity_boxes[os.path.splitext(os.path.basename(image_path))[0]]
        entity_box = convert_saved_box(entity_box, *image.shape[:2])

        mixup_anomaly_mask = np.zeros(image.shape[:2])
        image_mixup = np.copy(image)
        # mixup异常到图像
        for box in [[0,0,2,4], [0,0,4,2], [0,0,1,2], [0,0,2,1]]:
            # 随机挑选映射位置
            # box = convert_saved_box(box, *mixup_anomaly_mask.shape)
            randStartPos = get_start_pos(entity_box, box)
            h1, w1, h2, w2 = box
            h1_t = randStartPos[0]
            w1_t = randStartPos[1]
            h2_t = randStartPos[0] + (h2-h1)
            w2_t = randStartPos[1] + (w2-w1)
            # mixup
            # image_mixup[h1_t:h2_t, w1_t:w2_t] = 0.5*image_mixup[h1:h2, w1:w2] + 0.5*anomaly_image[h1:h2, w1:w2]
            # image_mixup[h1_t:h2_t, w1_t:w2_t] = anomaly_image[h1:h2, w1:w2]
            image_mixup[h1_t:h2_t, w1_t:w2_t] = get_area_mean(image, (h1_t, w1_t, h2_t, w2_t)) * 0.6
            # 更新异常mask
            mixup_anomaly_mask[h1_t:h2_t, w1_t:w2_t] = 1

        # pseudo aument
        augmented_image, augment_anomaly_mask, _ = self.augment_image(image_mixup, anomaly_source_path)
        anomaly_mask = augment_anomaly_mask + mixup_anomaly_mask.reshape(*mixup_anomaly_mask.shape, 1)
        anomaly_mask[anomaly_mask > 1] = 1

        # transpose
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, True



    def __getitem__(self, idx):
        ok_idx = torch.randint(0, self.len_ok, (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image_mixup(self.ok_paths[ok_idx],
                                                                       self.anomaly_source_paths[anomaly_source_idx])
        ng_idx = idx
        ng_path = self.ng_paths[ng_idx]
        gt_path = os.path.join(self.gt_root, os.path.basename(ng_path).replace('.jpg', '.png'))
        ng_image, gt_mask = self.transform_image_sup(ng_path, gt_path)
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'ng_image': ng_image, 'gt_mask': gt_mask, 'idx': idx}

        return sample