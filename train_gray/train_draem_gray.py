import torch
from torch.utils.data import DataLoader
from torch import optim

import path, sys
file_path = path.Path(__file__).abspath()
print(file_path)
sys.path.append(file_path.parent.parent.parent)

from gray_data_loader import MVTecDRAEMTrainDatasetGray
from DRAEM.base.tensorboard_visualizer import TensorboardVisualizer
from DRAEM.base.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from DRAEM.base.loss import FocalLoss, SSIM
import os
from tqdm import tqdm
import itertools

def decode_shift_window(img, mean, std, min_value, max_value):
    img = img * std + mean
    img = img * (max_value - min_value) + min_value
    max_mask = img >= max_value - 1
    return img, max_mask

def gray_visualize(img_tensor, mode, mean, std, max_value):
    if mode == 'single':
        img_tensor = img_tensor[:, 0]
        mean = mean[0]
        std = std[0]
        img_tensor = img_tensor * std + mean
        img_tensor = img_tensor * 65535
    elif mode == 'ms':
        img_tensor = img_tensor[:, -1]#[bs, h, w]
        mean = mean[-1]
        std = std[-1]
        img_tensor = img_tensor * std + mean
        img_tensor = img_tensor * 65535
    elif mode == 'sw':
        img0, max_mask0 = decode_shift_window(img_tensor[:, 0], std[0], mean[0], 0, 511)
        img1, max_mask1 = decode_shift_window(img_tensor[:, 2], std[2], mean[2], 512, 2047)
        img2, max_mask2 = decode_shift_window(img_tensor[:, 4], std[4], mean[4], 2048, 4095)
        img_tensor = img0 * (~max_mask0) + img1 * max_mask0 * (~max_mask1) + img2 * max_mask1 * (~max_mask2) + 65000 * max_mask2


    img_tensor[img_tensor > max_value] = max_value
    img_tensor = img_tensor / max_value
    return img_tensor.unsqueeze(1) #[bs, 1, h, w]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_name, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.iters)+'_bs'+str(args.bs)+"_"+obj_name+'_'

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    dataset = MVTecDRAEMTrainDatasetGray(args.root_dir, args.anomaly_source_path, args.mode, mean=args.mean, std=args.std)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16)
    in_channels = dataset[0]['image'].shape[0]
    

    model = ReconstructiveSubNetwork(in_channels=in_channels, out_channels=in_channels)
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=2*in_channels, out_channels=2)
    model_seg.apply(weights_init)
    model_path = os.path.join(args.checkpoint_path, run_name+".pckl")
    model_seg_path = os.path.join(args.checkpoint_path, run_name+"_seg.pckl")
    optimizer_path = os.path.join(args.checkpoint_path, run_name+'_opt.ckpt')
    scheduler_path = os.path.join(args.checkpoint_path, run_name+'_sch.ckpt')
    model_bak_path = model_path+'.bak'
    model_seg_bak_path = model_seg_path+'.bak'
    optimizer_bak_path = optimizer_path+'.bak'
    scheduelr_bak_path = scheduler_path+'.bak'
    
    if args.resume != 0:
        model_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
        model_seg_state_dict = torch.load(model_seg_path, map_location='cpu')
        model_seg.load_state_dict(model_seg_state_dict)
        os.rename(model_path, model_bak_path)
        os.rename(model_seg_path, model_seg_bak_path)

    model.cuda()
    model_seg.cuda()

    optimizer = torch.optim.Adam([
                                  {"params": model.parameters(), "lr": args.lr},
                                  {"params": model_seg.parameters(), "lr": args.lr}])

    epochs = int(args.iters / args.epoch_size)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[epochs*0.8, epochs*0.9],gamma=0.2, last_epoch=-1)
    
    if args.resume != 0:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu'))
        os.rename(optimizer_path, optimizer_bak_path)
        os.rename(scheduler_path, scheduelr_bak_path)
        

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    print(len(dataloader))
    dataloader_iter = iter(itertools.cycle(dataloader))
    print(f'start training from iter {args.resume}')
    for n_iter in tqdm(range(args.resume, args.iters)):
        sample_batched = next(dataloader_iter)
    
        gray_batch = sample_batched["image"].cuda()
        aug_gray_batch = sample_batched["augmented_image"].cuda()
        anomaly_mask = sample_batched["anomaly_mask"].cuda()

        gray_rec = model(aug_gray_batch)
        joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        l2_loss = loss_l2(gray_rec,gray_batch)
        ssim_loss = loss_ssim(gray_rec, gray_batch)

        segment_loss = loss_focal(out_mask_sm, anomaly_mask)
        loss = l2_loss + ssim_loss + segment_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if args.visualize and (n_iter+1) % 200 == 0:
            visualizer.plot_loss(l2_loss, n_iter+1, loss_name='l2_loss')
            visualizer.plot_loss(ssim_loss, n_iter+1, loss_name='ssim_loss')
            visualizer.plot_loss(segment_loss, n_iter+1, loss_name='segment_loss')
        if args.visualize and (n_iter+1) % 400 == 0:
            t_mask = out_mask_sm[:, 1:, :, :]
            visualizer.visualize_image_batch(gray_visualize(aug_gray_batch, args.mode, args.mean, args.std, args.vis_max), n_iter+1, image_name='batch_augmented')
            visualizer.visualize_image_batch(gray_visualize(gray_batch, args.mode, args.mean, args.std, args.vis_max), n_iter+1, image_name='batch_recon_target')
            visualizer.visualize_image_batch(gray_visualize(gray_rec, args.mode, args.mean, args.std, args.vis_max), n_iter+1, image_name='batch_recon_out')
            visualizer.visualize_image_batch(anomaly_mask, n_iter+1, image_name='mask_target')
            visualizer.visualize_image_batch(t_mask, n_iter+1, image_name='mask_out')

        if (n_iter+1) % args.epoch_size == 0:
            scheduler.step()

        if (n_iter+1) % 1000 == 0:
            torch.save(model.state_dict(), model_path)
            torch.save(model_seg.state_dict(), model_seg_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            torch.save(scheduler.state_dict(), scheduler_path)
            checkpoint = {
                'model': model.state_dict(),
                'model_seg': model_seg.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint,f'{args.checkpoint_path}/{n_iter+1:04d}.ckpt')


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epoch_size', action='store', type=int, required=True)
    parser.add_argument('--iters', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--root_dir', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--mode', action='store', type=str, required=True)
    parser.add_argument('--mean', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--std', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--vis_max', action='store', type=int, required=True)
    parser.add_argument('--resume', action='store', type=int, default=0)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device('bubbles1k', args)

