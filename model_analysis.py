import sys
import numpy as np
from openvino.runtime import Core, get_version, serialize, Node
import openvino as ov

if len(sys.argv) < 2:
    print("Please provide model path")
    exit(-1)
model_path = sys.argv[1]
core = Core()
ov_model = core.read_model(model_path)

batch_size = 2
seq_len = 5
input_ids = np.arange(batch_size*seq_len).reshape([batch_size, seq_len])
attention_mask = np.ones([batch_size, seq_len])

input_names = [output.get_node().get_friendly_name() for output in ov_model.inputs]
original_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": np.array(np.arange(seq_len).reshape([1, seq_len]) * batch_size), "beam_idx": np.zeros([batch_size])}
print("model inputs {0}".format(input_names))

actual_inputs = {}
for name in input_names:
    if name in original_inputs:
        actual_inputs[name] = original_inputs[name]
    else:
        print("get unexpected input `{0}`, please check the model".format(name))
        exit(-1)

compiled_model = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"})

ov_results = compiled_model(actual_inputs)

def check_sdpa(sdpa: Node):
    # simple sdpa pattern check. only counts the memoryoutput nodes.
    sdpa_outputs = sdpa.outputs()
    if len(sdpa_outputs) == 1:
        return False
    output_type = {}
    for output in sdpa_outputs:
        child_nodes = output.get_target_inputs()
        node = next(iter(child_nodes)).get_node()
        rt_info = node.get_rt_info()
        node_type = rt_info['layerType'].astype(str).lower()
        output_type[node_type] = output_type[node_type] + 1 if node_type in output_type else 1

    return False if 'memoryoutput' not in output_type else True
        
        
runtime_model = compiled_model.get_runtime_model()
fuse_rope = False
fuse_sdpa = False
for execution_node in runtime_model.get_ordered_ops():
    rt_info = execution_node.get_rt_info()
    layer_type = rt_info['layerType'].astype(str)
    if layer_type.lower() == 'rope':
        fuse_rope = True
    elif layer_type.lower() == 'scaleddotproductattention':
        fuse_sdpa = check_sdpa(execution_node)
        

print("fuse rope {0}, fuse sdpa {1}".format(fuse_rope, fuse_sdpa))