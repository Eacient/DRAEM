import glob
import os
import torch
import cv2
import numpy as np

from DRAEM.base.data_loader import MVTecDRAEMTrainDataset

class MVTecDRAEMTrainDatasetSup(MVTecDRAEMTrainDataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        super().__init__(root_dir, anomaly_source_path, resize_shape)
        self.ok_paths = sorted(glob.glob(root_dir+"/train/good/*.jpg"))
        self.ng_paths = sorted(glob.glob(root_dir+"/train/impurity/*.jpg"))
        self.gt_root = os.path.join(root_dir, 'ground_truth', 'impurity')
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

    def __getitem__(self, idx):
        ok_idx = torch.randint(0, self.len_ok, (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.ok_paths[ok_idx],
                                                                       self.anomaly_source_paths[anomaly_source_idx])
        ng_idx = idx
        ng_path = self.ng_paths[ng_idx]
        gt_path = os.path.join(self.gt_root, os.path.basename(ng_path).replace('.jpg', '.png'))
        ng_image, gt_mask = self.transform_image_sup(ng_path, gt_path)
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'ng_image': ng_image, 'gt_mask': gt_mask, 'idx': idx}

        return sample