import torch
import path, sys
cur_path = path.Path(__file__).abspath()
sys.path.append(cur_path.parent.parent.parent)
from DRAEM.base.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

class DRAEMPack(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.recon = ReconstructiveSubNetwork(in_channels=in_channels, out_channels=in_channels)
        self.seg = DiscriminativeSubNetwork(in_channels=2*in_channels, out_channels=2)

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
        if self.training:
            return rec, out_mask_sm
        else:
            return out_mask_sm

if __name__ == "__main__":
    model = torch.load('/home/caoxiatian/DRAEM/checkpoints/retrain_bubbles_50/retrain.mod')
    model.eval()
    model = model.cuda(1)

    dynamic_axes_023 = {
        "input": {0: "batch", 2: "height", 3: "width"}, #[1, 8, h, w]
        "output": {0: "batch", 2: "height", 3: "width"} #[1, 2, h, w]
    }

    dummy_input = torch.randn(1, 8, 640, 640).cuda(1)
    torch.onnx.export(model, dummy_input, '/home/caoxiatian/DRAEM/deploy/compressed_bubbles_50.onnx', 
                      export_params=True, 
                      input_names=["input"], output_names=["output"], 
                      dynamic_axes=dynamic_axes_023, opset_version=11)
    print('onnx export finished!')