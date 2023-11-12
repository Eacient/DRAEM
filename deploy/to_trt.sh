ONNX_PATH=/home/caoxiatian/DRAEM/deploy/compressed_bubbles_50.onnx
ENGINE_PATH=/home/caoxiatian/DRAEM/deploy/compressed_bubbles_50.engine
/home/caoxiatian/.basis/TensorRT-8.6.1.6/bin/trtexec \
    --skipInference \
    --minShapes=input:1x8x256x256 \
    --optShapes=input:2x8x640x640 \
    --maxShapes=input:4x8x640x640 \
    --onnx=$ONNX_PATH \
    --saveEngine=$ENGINE_PATH \
    --device=3

    # --inputIOFormats \
    # --outputIOFormats \
