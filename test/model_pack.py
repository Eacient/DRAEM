import torch
from DRAEM.base.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DRAEMPack(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.recon = ReconstructiveSubNetwork(in_channels=in_channels, out_channels=in_channels)
        self.seg = DiscriminativeSubNetwork(in_channels=2*in_channels, out_channels=2)
        self.recon.apply(weights_init)
        self.seg.apply(weights_init)

    def load_pack_checkpoint(self, pack_path):
        pack_dict = torch.load(pack_path, map_location='cpu')
        self.recon.load_state_dict(pack_dict['model'])
        self.seg.load_state_dict(pack_dict['model_seg'])
    
    def load_checkpoint(self, recon_path, seg_path):
        recon_dict = torch.load(recon_path, map_location='cpu')
        self.recon.load_state_dict(recon_dict)
        seg_dict = torch.load(seg_path, map_location='cpu')
        self.seg.load_state_dict(seg_dict)
    
    def forward(self, x):
        rec = self.recon(x)
        joined_in = torch.cat((rec, x), dim=1)
        out_mask = self.seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        return out_mask_sm
    