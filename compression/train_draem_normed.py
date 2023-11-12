import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import itertools
import os



import path, sys
file_path = path.Path(__file__).abspath()
print(file_path)
sys.path.append(file_path.parent.parent.parent)

from DRAEM.compression.to_onnx import DRAEMPack
from DRAEM.base.tensorboard_visualizer import TensorboardVisualizer
from DRAEM.train_gray.gray_data_loader import MVTecDRAEMTrainDatasetGray
from DRAEM.base.loss import FocalLoss, SSIM

def convert_cuda_to_cpu(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_cuda_to_cpu(value)
    # elif isinstance(data, list):
    #     print('list here')
    #     for i, item in enumerate(data):
    #         data[i] = convert_cuda_to_cpu(item)
    elif isinstance(data, torch.Tensor):
        data = data.cpu()
    return data

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

    img_tensor[img_tensor > max_value] = max_value
    img_tensor = img_tensor / max_value
    return img_tensor.unsqueeze(1) #[bs, 1, h, w]

def training_step(batch, model, *largs, **kwargs):
    gray_batch = batch["image"].cuda()
    aug_gray_batch = batch["augmented_image"].cuda()
    anomaly_mask = batch["anomaly_mask"].cuda()
    gray_rec, out_mask = model(aug_gray_batch);
    out_mask_sm = torch.softmax(out_mask, dim=1)
    l2_loss = loss_l2(gray_rec,gray_batch)
    ssim_loss = loss_ssim(gray_rec, gray_batch)
    segment_loss = loss_focal(out_mask_sm, anomaly_mask)
    loss = l2_loss + ssim_loss + segment_loss
    if args.visualize and (kwargs['n_iter']+1) % 50 == 0:
        kwargs['visualizer'].plot_loss(l2_loss, kwargs['n_iter']+1, loss_name='l2_loss')
        kwargs['visualizer'].plot_loss(ssim_loss, kwargs['n_iter']+1, loss_name='ssim_loss')
        kwargs['visualizer'].plot_loss(segment_loss, kwargs['n_iter']+1, loss_name='segment_loss')
    if args.visualize and (kwargs['n_iter']+1) % 400 == 0:
        t_mask = out_mask_sm[:, 1:, :, :]
        kwargs['visualizer'].visualize_image_batch(gray_visualize(aug_gray_batch, args.mode, args.mean, args.std, args.vis_max), kwargs['n_iter']+1, image_name='batch_augmented')
        kwargs['visualizer'].visualize_image_batch(gray_visualize(gray_batch, args.mode, args.mean, args.std, args.vis_max), kwargs['n_iter']+1, image_name='batch_recon_target')
        kwargs['visualizer'].visualize_image_batch(gray_visualize(gray_rec, args.mode, args.mean, args.std, args.vis_max), kwargs['n_iter']+1, image_name='batch_recon_out')
        kwargs['visualizer'].visualize_image_batch(anomaly_mask, kwargs['n_iter']+1, image_name='mask_target')
        kwargs['visualizer'].visualize_image_batch(t_mask, kwargs['n_iter']+1, image_name='mask_out')
    return loss


def training_model(model, optimizer, training_step, lr_scheduler, max_steps, max_epochs, *largs, **kwargs):
    try:
        retrain = kwargs['retrain']
        run_name = 'DRAEM_retrain_'+str(args.retrain_lr)+'_'+str(args.retrain_iters)+'_bs'+str(args.bs)
        max_steps = args.retrain_iters
    except:
        run_name = 'DRAEM_compress_'+str(args.compress_lr)+'_'+str(args.compress_iters)+'_bs'+str(args.bs)
        max_steps = args.compress_iters
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    # init dataloader
    dataset = MVTecDRAEMTrainDatasetGray(args.root_dir, args.anomaly_source_path, args.mode, mean=args.mean, std=args.std)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16)

    
    print(max_steps)
    dataloader_iter = iter(itertools.cycle(dataloader))
    for n_iter in tqdm(range(0, max_steps)):

        sample_batched = next(dataloader_iter)
        loss = training_step(sample_batched, model, visualizer=visualizer, n_iter=n_iter)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if (n_iter+1) % args.epoch_size == 0:
            lr_scheduler.step()
        

if __name__ == "__main__":

    import argparse
    from nni.compression.pruning import SlimPruner
    from nni.compression.speedup import ModelSpeedup


    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--compress_lr', action='store', type=float, required=True)
    parser.add_argument('--retrain_lr', action='store', type=float, required=True)
    parser.add_argument('--epoch_size', action='store', type=int, required=True)
    parser.add_argument('--compress_iters', action='store', type=int, required=True)
    parser.add_argument('--retrain_iters', action='store', type=int, required=True)
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
    parser.add_argument('--pack_path', action='store', type=str, required=True)
    parser.add_argument('--regular_scale', action='store', type=float, required=False)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):

        model = DRAEMPack(in_channels=8)
        model.load_pack_checkpoint(args.pack_path)
        model = model.cuda()
        print('origin model paramater number: ', sum([param.numel() for param in model.parameters()]))

        import nni
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=args.compress_lr)
        epochs = args.compress_iters // args.epoch_size
        scheduler = nni.trace(optim.lr_scheduler.MultiStepLR)(traced_optimizer, milestones=[epochs*0.8, epochs*0.9], gamma=0.2, last_epoch=-1)
        from nni.compression import TorchEvaluator
        evaluator = TorchEvaluator(training_func=training_model, optimizers=traced_optimizer, training_step=training_step, lr_schedulers=scheduler)
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()
        
        config_list = [{
            'op_types': ['Conv2d'],
            'sparse_ratio': 0.5,
            'exclude_op_names': [
                'recon.decoder.fin_out.0',
                'seg.decoder_segment.fin_out.0',
            ]
        }]
        pruner = SlimPruner(model, config_list, evaluator, training_steps=args.compress_iters, regular_scale=args.regular_scale)
        _, masks = pruner.compress()
        masks = convert_cuda_to_cpu(masks)
        torch.save(model.cpu(), os.path.join(args.checkpoint_path, 'normed.mod'))
        pruner.unwrap_model()
        model = ModelSpeedup(model.cpu(), torch.rand(1, 8, 128, 128), masks).speedup_model()
        print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
        torch.save(model.cpu(), os.path.join(args.checkpoint_path, 'compressed.mod'))

        model = model.cuda()
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.retrain_lr}])
        epochs = args.retrain_iters // args.epoch_size
        scheduler = optim.lr_scheduler.MultiStepLR(traced_optimizer,[epochs*0.8, epochs*0.9],gamma=0.2, last_epoch=-1)
        training_model(model, optimizer, training_step, lr_scheduler=scheduler, max_steps=args.retrain_iters, max_epochs=None, retrain=True)
        torch.save(model.cpu(), os.path.join(args.checkpoint_path, 'retrain.mod'))