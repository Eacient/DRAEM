# mode: single, ms(multi-scale), sw(sliding window)
# single: 0-65000 -> float32
# ms: 0-511, 0-1023,    0-2047,     0-4095,     0-8191,     0-16383,    0-32767,    0-65535
# sw: 0-511, 256-1023,  512-2047,   1024-4095,  2048-8191,  4096-16383, 8192-32767, 16384-65535

import glob
import os
import torch
import cv2
import numpy as np
import random

if __name__ == "__main__":
    import path, sys
    file_path = path.Path(__file__).abspath()
    sys.path.append(file_path.parent.parent.parent)

from DRAEM.base.perlin import rand_perlin_2d_np
from DRAEM.base.data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset

def ratio_addup(arr1, arr2, max_val = 65000):
    r1 = (max_val - arr1) / max_val
    r2 = (max_val - arr2) / max_val
    return max_val * (1-r1) * (1-r2)

def channeled_norm(image, mode='single'):
    # image [h, w], 0-65000
    assert mode in ['single', 'ms', 'sw']
    if mode == 'single':
        image = image / 65535
        image = image.astype(np.float32)
    elif mode == 'ms':
        ths = [511, 1023, 2047, 4095, 8191, 16383, 32767, 65535]
        channels = [(np.where(image >= th, th, 0) + (image < th) * image) / th for th in ths]
        image = np.concatenate(channels, axis=0).astype(np.float32)
    else:
        lows = [0, 256, 512, 1024, 2048, 4096, 8192, 16384]
        highs = [511, 1023, 2047, 4095, 8191, 16383, 32767, 65535]
        channels = [(np.where(image >= thr, thr-thl, 0) + (image < thr) * (image > thl) * (image - thl)) / (thr - thl)
                    for (thl, thr) in zip(lows, highs)]
        image = np.concatenate(channels, axis=0).astype(np.float32)
    return image

def mean_std_norm(image, mean, std):
    return ((image - mean) / std).astype(np.float32)


class MVTecDRAEMTestDatasetGray(MVTecDRAEMTestDataset):
    def __init__(self, root_dir, mode='single', mean=0, std=1, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.npy"))
        self.resize_shape = resize_shape
        self.mode = mode
        self.mean = np.array(mean).reshape(-1,1,1)
        self.std = np.array(std).reshape(-1,1,1)
        
    def __len__(self):
        return len(self.images)
    
    def transform_image(self, image_path, mask_path):
        image = np.load(image_path) #[h, w] 0-65000
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255 #[h, w] 0-1
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        mask = np.expand_dims(mask, 0) #[1, h, w] 0-1
        image = np.expand_dims(image, 0) #[1, h, w] 0-1
        image = channeled_norm(image, mode=self.mode) #[?, h, w]
        image = mean_std_norm(image, self.mean, self.std)
        return image, mask


class MVTecDRAEMTrainDatasetGray(MVTecDRAEMTrainDataset):

    def __init__(self, root_dir, anomaly_source_path, mode='single', mean=0, std=1, resize_shape=None):
        super().__init__(root_dir, anomaly_source_path, resize_shape)
        self.ok_paths = sorted(glob.glob(root_dir+"/train/good/*.npy"))
        self.ng_paths = sorted(glob.glob(root_dir+"/train/impurity/*.npy"))
        self.recon_paths = sorted(glob.glob(root_dir+"/train/recon/*.npy"))
        self.gt_root = os.path.join(root_dir, 'ground_truth', 'impurity')
        self.len_ok = len(self.ok_paths)
        self.len_ng = len(self.ng_paths)
        self.len_ano = len(self.anomaly_source_paths)
        self.mode = mode
        self.mean = np.array(mean).reshape(-1,1,1)
        self.std = np.array(std).reshape(-1,1,1)

    def __len__(self):
        return self.len_ng + self.len_ok
    
    def generate_perlin_thr(self, shape):
        perlin_scale = 6
        min_perlin_scale = 0
        
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        return perlin_thr

    def perlin_augment_image_gray(self, augmented_image, mask, anomaly_source_path, ratio=1):
        has_anomaly=0.0
        if random.random() < ratio:
            # anomaly_source 从rgb转灰度，然后重缩放到范围0-16000
            aug = self.randAugmenter()
            anomaly_source_img = cv2.imread(anomaly_source_path)
            anomaly_source_img = cv2.resize(anomaly_source_img, augmented_image.shape[-2:])
            anomaly_img_augmented = aug(image=anomaly_source_img)

            anomaly_img_gray = cv2.cvtColor(anomaly_img_augmented, cv2.COLOR_BGR2GRAY) / 255 * 16000 #[h, w] 1-16000
            perlin_thr = self.generate_perlin_thr(augmented_image.shape[-2:]) #[h, w]
            anomaly_img_gray[perlin_thr == 0] = 65000 #[h, w]
        
            anomaly_img_gray = np.expand_dims(anomaly_img_gray, 0) #[1, h, w]
            augmented_image = ratio_addup(augmented_image, anomaly_img_gray) #[1, h, w]
            mask = mask + perlin_thr
            mask[mask > 1] = 1
            has_anomaly=1.0
        return augmented_image, mask, np.array(has_anomaly, dtype=np.float32)

    def transform_image_gray(self, image_path, recon_path, mask_path, anomaly_source_path, mode): #该函数保持65000取值范围
        image = np.expand_dims(np.load(image_path), 0) #[1, h, w] 0-65000
        augmented_image = image

        if recon_path is not None:
            image = np.expand_dims(np.load(recon_path), 0) #[1, h, w] 0-65000
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, 0) / 255 #[1, h, w] 0-1
        else:
            mask = np.zeros_like(image) #[1, h, w]

        if anomaly_source_path is not None:
            augmented_image, anomaly_mask, has_anomaly = self.perlin_augment_image_gray(augmented_image, mask, anomaly_source_path)
        else:
            anomaly_mask = mask
            has_anomaly = np.array(1.0, dtype=np.float32) if np.any(anomaly_mask > 0) else np.array(0.0, dtype=np.float32)

        image = channeled_norm(image, mode=mode) #[?, h, w]
        augmented_image = channeled_norm(augmented_image, mode=mode) #[?, h, w]
        anomaly_mask = anomaly_mask.astype(np.float32) #[1, h, w]
        return image, augmented_image, anomaly_mask, has_anomaly

    def calculate_mean_std(self):
        image, _, _, _ = self.transform_image_gray(self.ok_paths[0], None, None, None, self.mode)
        total_sum = np.zeros(image.shape[0])
        total_count = 0
        total_var = np.zeros(image.shape[0])
        for p in self.ok_paths:
            image, _, _, _ = self.transform_image_gray(p, None, None, None, self.mode)
            total_sum += np.sum(image, (1, 2))
            total_count += image.shape[1] * image.shape[2]
        mean = total_sum / total_count
        print(f'mean={mean}')
        for p in self.ok_paths:
            image, _, _, _ = self.transform_image_gray(p, None, None, None, self.mode)
            total_var += np.sum((image - mean.reshape(-1,1,1))**2, (1, 2))
        var = total_var / total_count
        std = np.sqrt(var)
        print(f'mean={mean} std={std}')

    def __getitem__(self, idx):
        if idx < self.len_ok:
            image_path = self.ok_paths[idx]
            recon_path, mask_path = None, None
        else:
            image_path = self.ng_paths[idx-self.len_ok]
            recon_path = self.recon_paths[idx-self.len_ok]
            image_name = os.path.basename(image_path)
            mask_path = os.path.join(self.gt_root, image_name.replace('.npy', '.png'))

        anomaly_source_idx = random.randint(0, self.len_ano-1)
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image_gray(image_path, 
                                                recon_path, mask_path,
                                                self.anomaly_source_paths[anomaly_source_idx],
                                                mode=self.mode)
        image = mean_std_norm(image, self.mean, self.std)
        augmented_image = mean_std_norm(augmented_image, self.mean, self.std)

        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample
    
if __name__ == "__main__":
    # dataset = MVTecDRAEMTrainDatasetGray('/home/caoxiatian/data/xray/mvtec/fish_bubbles1k', '/home/caoxiatian/data/dtd/images', mode='single')
    # dataset.calculate_mean_std()
    # mean=[0.50729982] std=[0.41779612]

    # dataset = MVTecDRAEMTrainDatasetGray('/home/caoxiatian/data/xray/mvtec/fish_bubbles1k', '/home/caoxiatian/data/dtd/images', mode='ms')
    # dataset.calculate_mean_std()
    # mean = [0.99310124 0.9728329  0.94237832 0.90898088 0.83903737 0.74311205 0.60096486 0.50729982]
    # std = [0.05002454 0.1179261  0.18425917 0.24282377 0.29716158 0.34985626 0.38834061 0.41779612]

    dataset = MVTecDRAEMTrainDatasetGray('/home/caoxiatian/data/xray/mvtec/fish_bubbles1k', '/home/caoxiatian/data/dtd/images', mode='sw')
    dataset.calculate_mean_std()
    # mean=[0.99310124 0.96397182 0.92547368 0.88770151 0.80459293 0.68782326 0.52160943 0.42869957]
    # std=[0.05002454 0.15590053 0.2362465  0.2952776  0.34971105 0.41146914 0.44979469 0.47254149]
    pass