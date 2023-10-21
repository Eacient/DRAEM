import path, sys
file_path = path.Path(__file__).abspath()
sys.path.append(file_path.parent.parent.parent)
from DRAEM.test.model_pack import DRAEMPack
import torch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_channel', action='store', type=int, required=True)
    parser.add_argument('--recon_path', action='store', type=str, required=True)
    parser.add_argument('--seg_path', action='store', type=str, required=True)
    parser.add_argument('--export_path', action='store', type=str, required=True)

    args = parser.parse_args()

    recon_path = args.recon_path
    seg_path = args.seg_path
    model = DRAEMPack(args.input_channel)
    model.load_checkpoint(recon_path, seg_path)
    model.eval()

    dynamic_axes_023 = {
        "input": {0: "batch", 2: "height", 3: "width"}, #[1, 8, h, w]
        "output": {0: "batch", 2: "height", 3: "width"} #[1, 2, h, w]
    }

    dummy_input = torch.randn(1, args.input_channel, 640, 640)
    torch.onnx.export(model, dummy_input, args.export_path, 
                      export_params=True, 
                      input_names=["input"], output_names=["output"], 
                      dynamic_axes=dynamic_axes_023, opset_version=11)
    print('onnx export finished!')