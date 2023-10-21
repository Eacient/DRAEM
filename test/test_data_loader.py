import os
import cv2
import glob
import numpy as np
import torch
from DRAEM.deploy.channeled_norm import channeled_norm
from torch.utils.data import Dataset

if __name__ == "__main__":
    import path, sys
    file_path = path.Path(__file__).abspath()
    sys.path.append(file_path.parent.parent.parent)
from DRAEM.train_gray.gray_data_loader import mean_std_norm

class DRAEMTestDatasetGray(Dataset):
    def __init__(self, root_dir, mode='single', mean=0, std=1, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/ok/*.npy")+glob.glob(root_dir+"/ng/*.npy"))
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        base_file_name = file_name.split(".")[0]
        if base_dir == 'ok':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../mask/')
            mask_file_name = base_file_name +".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'name': base_file_name, 'idx': idx}

        return sample
    
if __name__ == "__main__":
    dataset = DRAEMTestDatasetGray('/home/caoxiatian/data/xray/bubbles_test', mode='single', mean=0.50729982, std=0.41779612)
    dataset[0]
    # mean = [0.99310124 0.9728329  0.94237832 0.90898088 0.83903737 0.74311205 0.60096486 0.50729982]
    # std = [0.05002454 0.1179261  0.18425917 0.24282377 0.29716158 0.34985626 0.38834061 0.41779612]
    dataset = DRAEMTestDatasetGray('/home/caoxiatian/data/xray/bubbles_test', mode='ms', 
                                   mean = [0.99310124, 0.9728329,  0.94237832, 0.90898088, 0.83903737, 0.74311205, 0.60096486, 0.50729982], 
                                   std = [0.05002454, 0.1179261,  0.18425917, 0.24282377, 0.29716158, 0.34985626, 0.38834061, 0.41779612])
    dataset[0]
    # mean=[0.99310124 0.96397182 0.92547368 0.88770151 0.80459293 0.68782326 0.52160943 0.42869957]
    # std=[0.05002454 0.15590053 0.2362465  0.2952776  0.34971105 0.41146914 0.44979469 0.47254149]
    dataset = DRAEMTestDatasetGray('/home/caoxiatian/data/xray/bubbles_test', mode='sw', 
                                   mean=[0.99310124, 0.96397182, 0.92547368, 0.88770151, 0.80459293, 0.68782326, 0.52160943, 0.42869957], 
                                   std=[0.05002454, 0.15590053, 0.2362465,  0.2952776,  0.34971105, 0.41146914, 0.44979469, 0.47254149])
    dataset[0]