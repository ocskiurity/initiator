import os
import sys
sys.path.append(os.getcwd())
import onnx
import json
import torch
import onnxruntime
import subprocess
import numpy as np
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static, 
    CalibrationDataReader, 
    QuantType, 
    QuantFormat
)
import torch
import torch.nn as nn
from onnx import numpy_helper
from sklearn.metrics import accuracy_score

from src.data import prepare_dataset, CalibrationDataLoader
from src.model import MLP
from src.misc import export_to_onnx
from src.quantize import run_static_quantization, calculate_percolumn_scales_and_zero_points, quantize_data


if __name__ == "__main__":
    BASE = Path.cwd()
    
    model_checkpoint = BASE / f"checkpoint/MLP.json"
    onnx_model_path = BASE / f"checkpoint/MLP.onnx"
    quantized_model_path = BASE / f"checkpoint/MLP_quant.onnx"
    preproc_onnx_model_path = BASE / f"checkpoint/MLP_quant_preproc.onnx"
    
    df_path = BASE / "data/df.csv"
    
    print(f"### 1. Preparing train, calibration and test datasets.")
    data_train, data_calib, data_test, labels_train, labels_calib, labels_test, train_loader = prepare_dataset(df_path)
    
    print(f"### 2. Loading pretrained model and exporting it to onnx.")
    input_dim = data_test.shape[1]
    model = MLP(input_dim=input_dim)
    export_to_onnx(model,
                   input_dim=input_dim,
                   model_checkpoint=model_checkpoint,
                   onnx_model_path=onnx_model_path)
    
    print(f"### 3. Performing static uint8 quantization.")
    # perform static quantizaton
    calib_loader = CalibrationDataLoader(data_calib)
    
    preproc_command = [
        'python', '-m', 'onnxruntime.quantization.preprocess',
        '--input', onnx_model_path, '--output', preproc_onnx_model_path
    ]

    # Run the command
    subprocess.run(preproc_command, capture_output=True, text=True)

    run_static_quantization(preproc_onnx_model_path=preproc_onnx_model_path,
                            calib_loader=calib_loader,
                            quantized_model_out_path=quantized_model_path)
    
    print(f"### 4. Running inference with quantized weights and full-precision data input.")
    # Create an ONNX runtime session for the quantized model
    session = onnxruntime.InferenceSession(quantized_model_path)

    preds_quant = []
    # Adjust input data format if necessary
    for i in range(data_test.shape[0]):
        input_data = data_test[i:i+1]  # Ensuring input data is correctly shaped
        outputs = session.run(None, {'onnx::Gemm_0': input_data.numpy()})
        preds_quant.append(outputs[0].item())

    preds_quant = np.array(preds_quant)

    labels_quant = np.where(preds_quant>0.5, 1, 0).reshape(-1,)
    acc_quant = accuracy_score(labels_test, labels_quant)
    print(f"-> Quantized accuracy (float inputs): {acc_quant:.2f}")

    print(f"### 5. Loading quantized model and extract quantized weights and bias.")
    quant_model = onnx.load(quantized_model_path)
    # for init in quant_model.graph.initializer:
    #     print("Tensor Name:", init.name)
    #     # Convert ONNX tensor to numpy array
    #     weights = numpy_helper.to_array(init)
    #     print("Weights Array:\n", weights)
    #     print("---------------------------------------------------")    
    
    # retrieve uint8 quantized weights
    weights = numpy_helper.to_array(quant_model.graph.initializer[6]).astype(np.uint8).tolist()
    bias = numpy_helper.to_array(quant_model.graph.initializer[7]).astype(np.uint8).item()        

    print(f"### 6. Performing quantization on data input.")
    weight_scale = numpy_helper.to_array(quant_model.graph.initializer[4])
    weight_zero_point = numpy_helper.to_array(quant_model.graph.initializer[5])
    # print(f"Weight scale and zero point: {weight_scale} {weight_zero_point}")
    
    # retrieve scale and zero points to perform data input quantization
    data_scales, data_zero_points = calculate_percolumn_scales_and_zero_points(data_test)
    q_data_test = quantize_data(data_test, data_scales, data_zero_points).astype(np.uint8).tolist()

    print(f"### 7. Saving model and quantized data json files.")
    # prepare model.json and quantizedData.json
    model_dict = {
        "weight": weights,
        "bias": bias
    }
    model_dict_path = BASE / f"data/json/model.json"
    with open(model_dict_path, "w") as fw:
        json.dump(model_dict, fw)
    
    quantized_data_dict = {
        "input": q_data_test
    }
    quantized_data_dict_path = BASE / f"data/json/quantizedData.json"
    with open(quantized_data_dict_path, "w") as fw:
        json.dump(quantized_data_dict, fw)
    
    
    

    