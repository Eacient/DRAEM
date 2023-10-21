from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from DRAEM.base.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import glob
import os
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from eval_utils import get_missing_boxes


class TestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        # self.images = glob.glob(os.path.join(root_dir, 'train', 'impurity')+'/*.jpg') + glob.glob(os.path.join(root_dir, 'val', 'impurity')+'/*.jpg')
        # self.gt_dir = os.path.join(root_dir, 'ground_truth', 'impurity')
        self.images = glob.glob(os.path.join(root_dir, 'images')+'/*.jpg')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        _, file_name = os.path.split(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.resize_shape)
        image = image.astype(np.float32) / 255
        image = np.transpose(image, (2, 0, 1))
        # mask_path = os.path.join(self.gt_dir, file_name.replace('.jpg', '.png'))
        # mask = Image.open(mask_path).resize(self.resize_shape)
        # mask = np.array(mask, dtype=np.float32)
        # mask = mask.reshape(1, *mask.shape)
        # has_anomaly = np.array([1], dtype=np.float32)

        # sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx, 'name': os.path.splitext(file_name)[0]}
        sample = {'image': image, 'idx': idx, 'name': os.path.splitext(file_name)[0]}

        return sample


def test(obj_name, data_path, checkpoint_path, base_model_name):
    prediction_dir = 'test_outputs/'
    os.makedirs(prediction_dir, exist_ok=True)
    img_dim = 256
    run_name = base_model_name+"_"+obj_name+'_'

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cpu'))
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cpu'))
    model_seg.cuda()
    model_seg.eval()

    dataset = TestDataset(data_path, resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

    missing_dict = {}
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):

            prediction_name = os.path.join(prediction_dir, sample_batched['name'][0])+'.npy' 
            # if (os.path.exists(prediction_name)):
            #     continue
            gray_batch = sample_batched["image"].cuda()


            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
            np.save(prediction_name, out_mask_cv)
            print(f'{prediction_name} saved')

            # true_mask = sample_batched["mask"][0,0]
    #         missing_boxes = get_missing_boxes(true_mask, out_mask_cv)
    #         if len(missing_boxes):
    #             print(missing_boxes)
    #             missing_dict[sample_batched['name'][0]] = missing_boxes
    #         # return
    # print(len(missing_dict))
    # np.save('missing.npy', missing_dict, allow_pickle=True)
    # print('missing.npy saved')

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, default=0)
    parser.add_argument('--base_model_name', action='store', type=str, default='DRAEM_test_0.0001_200_bs2')
    parser.add_argument('--data_path', action='store', type=str, default='/home/zhougaowei/datasets/xray/test/milk/')
    parser.add_argument('--checkpoint_path', action='store', type=str, default='/checkpoint/DRAEM_test_0.0001_200_bs1_cans_sup_')

    args = parser.parse_args()

    obj_name = 'cans_sup'

    with torch.cuda.device(args.gpu_id):
        test(obj_name,args.data_path, args.checkpoint_path, args.base_model_name)