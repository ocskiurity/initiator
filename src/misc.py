import json
import torch
import torch.nn as nn
from pathlib import Path

# convert tensor into list since tensor objects are not json serializable
def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


# convert the weights_biases to json
def state_dict_to_json(state_dict):
    state_dict_serializable = {name: tensor_to_list(param) for name, param in state_dict.items()}
    return json.dumps(state_dict_serializable)

        
def export_to_onnx(model, input_dim, model_checkpoint: Path, onnx_model_path: Path):
    # load weigths
    with open(model_checkpoint) as infile:
        weights = json.load(infile)

    # Convert the weights and biases to PyTorch tensors
    fc1_weight = torch.tensor(weights["fc1.weight"])
    fc1_bias = torch.tensor(weights["fc1.bias"])

    # Load the weights and biases into the model
    with torch.no_grad():
        model.fc1.weight = nn.Parameter(fc1_weight)
        model.fc1.bias = nn.Parameter(fc1_bias)

    dummy_input = torch.randn(1, input_dim)
    model.eval()
    torch.onnx.export(model, 
                    dummy_input, 
                    onnx_model_path, 
                    export_params=True, 
                    opset_version=12)
    