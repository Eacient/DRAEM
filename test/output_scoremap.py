import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import path, sys
file_path = path.Path(__file__).abspath()
sys.path.append(file_path.parent.parent.parent)

from test_data_loader import DRAEMTestDatasetGray
from model_pack import DRAEMPack

def output_scoremap(test_root_dir, mode, mean, std, recon_path, seg_path, pack_path, mod_path, output_dir='./test_output'):
    assert mode in ['single', 'ms', 'sw']
    dataset = DRAEMTestDatasetGray(test_root_dir, mode, mean, std)
    in_channels = dataset[0]['image'].shape[0]

    if mod_path is not None:
        model = torch.load(mod_path, map_location='cpu')
    else:
        model = DRAEMPack(in_channels)
        if pack_path is not None:
            model.load_pack_checkpoint(pack_path)
        else:
            model.load_checkpoint(recon_path, seg_path)
    model = model.cuda()
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)
    
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for sample in tqdm(dataloader):
            prediction_name = os.path.join(output_dir, sample['name'][0])+'.npy' 
            input = sample["image"].cuda()
            output = model(input)
            np.save(prediction_name, output[0, 1, :, :].cpu().numpy())
            print(f'{prediction_name} saved')
    
if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, default=0)
    parser.add_argument('--test_root_dir', action='store', type=str, required=True)
    parser.add_argument('--mode', action='store', type=str, required=True)
    parser.add_argument('--mean', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--std', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--recon_path', action='store', type=str, required=True)
    parser.add_argument('--seg_path', action='store', type=str, required=True)
    parser.add_argument('--pred_dir', action='store', type=str, required=True)
    parser.add_argument('--pack_path', action='store', type=str, required=False)
    parser.add_argument('--mod_path', action='store', type=str, required=False)

    args = parser.parse_args()
    with torch.cuda.device(args.gpu_id):
        output_scoremap(args.test_root_dir, args.mode, args.mean, args.std, args.recon_path, args.seg_path, args.pack_path, args.mod_path, os.path.join(args.pred_dir, 'score_maps'))