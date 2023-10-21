python deploy/to_onnx.py \
    --recon_path /home/caoxiatian/DRAEM/checkpoints/gray_ms1k/DRAEM_test_0.0001_80000_bs1_bubbles1k_.pckl \
    --seg_path /home/caoxiatian/DRAEM/checkpoints/gray_ms1k/DRAEM_test_0.0001_80000_bs1_bubbles1k__seg.pckl \
    --input_channel 8 \
    --export_path /home/caoxiatian/DRAEM/deploy/bubbles.onnx