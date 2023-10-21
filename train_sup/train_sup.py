import torch
from DRAEM.base.data_loader import MVTecDRAEMTrainDataset
from sup_data_loader import MVTecDRAEMTrainDatasetSup
from torch.utils.data import DataLoader
from torch import optim
from DRAEM.base.tensorboard_visualizer import TensorboardVisualizer
from DRAEM.base.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from DRAEM.base.loss import FocalLoss, SSIM
import os
from tqdm import tqdm

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

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDatasetSup(os.path.join(args.data_path,obj_name), args.anomaly_source_path, resize_shape=[512, 512])

        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)

        n_iter = 0
        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(tqdm(dataloader)):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                ng_batch = sample_batched["ng_image"].cuda()
                gt_mask = sample_batched["gt_mask"].cuda()


                composed_batch = torch.cat([aug_gray_batch, ng_batch], dim=0)
                composed_rec = model(composed_batch)
                joined_in = torch.cat([composed_rec, composed_batch], dim=1)

                # gray_rec = model(aug_gray_batch)
                # joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                gray_rec = composed_rec[:args.bs]
                l2_loss = loss_l2(gray_rec ,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                segment_loss = loss_focal(out_mask_sm, torch.cat([anomaly_mask, gt_mask], dim=0))

                loss = l2_loss + ssim_loss + segment_loss

                loss = loss / args.accumulation_iters
                loss.backward()

                if n_iter % args.accumulation_iters == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                if args.visualize and n_iter % 400 == 0:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(ng_batch, n_iter, image_name='batch_ng')
                    visualizer.visualize_image_batch(composed_rec[args.bs:], n_iter, image_name='batch_ng_rec')
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                    visualizer.visualize_image_batch(torch.cat([anomaly_mask, gt_mask], dim=0), n_iter, image_name='mask_out')


                n_iter +=1

            scheduler.step()
            print(f'loss={loss:.3f}, l2_loss={l2_loss:.3f}, ssim_loss={ssim_loss:.3f}, segment_loss={segment_loss:.3f}')
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=0)
    parser.add_argument('--bs', action='store', type=int, default=1)
    parser.add_argument('--accumulation_iters', action='store', type=int, default=4)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, default='200')
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, default='/root/autodl-tmp/datasets/xray/mvtec')
    parser.add_argument('--anomaly_source_path', action='store', type=str, default='/root/autodl-tmp/datasets/dtd/images')
    parser.add_argument('--checkpoint_path', action='store', type=str, default='./checkpoint')
    parser.add_argument('--log_path', action='store', type=str, default='./checkpoint')
    parser.add_argument('--visualize', action='store_true', default='True')

    args = parser.parse_args()

    obj_batch = [['cans_sup']]

    if int(args.obj_id) == -1:
        obj_list = ['cans_sup']
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

