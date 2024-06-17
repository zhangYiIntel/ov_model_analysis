from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys
import numpy as np
from openvino.runtime import Core, get_version, serialize
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

compiled_model = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"})

input_names = [output.get_node().get_friendly_name() for output in ov_model.inputs]

print("model inputs {0}".format(input_names))

original_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": np.array(np.arange(seq_len).reshape([1, seq_len]) * batch_size), "beam_idx": np.zeros([batch_size])}

actual_inputs = {}
for name in input_names:
    if name in original_inputs:
        actual_inputs[name] = original_inputs[name]
    else:
        print("get unexpected input {0}".format(name))


ov_results = compiled_model(actual_inputs)
        
runtime_model = compiled_model.get_runtime_model()
find_rope = False
for execution_node in runtime_model.get_ordered_ops():
    rt_info = execution_node.get_rt_info()
    layer_type = rt_info['layerType'].astype(str)
    if layer_type.lower() == 'rope':
        find_rope = True
        break
if find_rope:
    print(":)Good Job Find Rope")
else:
    print("Bad Job Rope Not found")