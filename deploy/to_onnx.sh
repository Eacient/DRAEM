python deploy/to_onnx.py \
    --recon_path /home/caoxiatian/DRAEM/checkpoints/gray_ms1k/DRAEM_test_0.0001_80000_bs1_bubbles1k_.pckl \
    --seg_path /home/caoxiatian/DRAEM/checkpoints/gray_ms1k/DRAEM_test_0.0001_80000_bs1_bubbles1k__seg.pckl \
    --pack_path /home/caoxiatian/DRAEM/checkpoints/gray_cans_bottles1k/100000.ckpt \
    --input_channel 8 \
    --export_path /home/caoxiatian/DRAEM/deploy/cans.onnx