import torch 
import onnx 
import tensorrt as trt 
 
onnx_model = onnx.load("bubbles.onnx") 

# create builder and network 
logger = trt.Logger(trt.Logger.ERROR) 
builder = trt.Builder(logger) 
EXPLICIT_BATCH = 1 << (int)( 
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
network = builder.create_network(EXPLICIT_BATCH) 
 
# parse onnx 
parser = trt.OnnxParser(network, logger) 
 
if not parser.parse(onnx_model.SerializeToString()): 
    error_msgs = '' 
    for error in range(parser.num_errors): 
        error_msgs += f'{parser.get_error(error)}\n' 
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
 
config = builder.create_builder_config() 
config.max_workspace_size = 1<<20 
# config.set_memory_pool_limit(pool_size=1<<20)
profile = builder.create_optimization_profile() 
 
profile.set_shape('input', [1,8 ,640 ,640], [1,8,640, 640], [4, 8, 640, 640])
config.add_optimization_profile(profile) 


# create engine 
with torch.cuda.device(torch.device('cuda:0')): 
    engine = builder.build_engine(network, config)
 
with open('bubbles.engine', mode='wb') as f: 
    f.write(bytearray(engine.serialize())) 
    print("generating file done!") 