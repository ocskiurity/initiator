import torch
import numpy as np
from pathlib import Path

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat


# calculate per-column scale and zero-points
def calculate_percolumn_scales_and_zero_points(data: torch.Tensor):
    data_np = data.numpy()
    scales = (data_np.max(axis=0) - data_np.min(axis=0)) / (255.0 - 0.0)
    zero_points = 0 - (data_np.min(axis=0) / scales)
    return scales, zero_points

def quantize_data_column(data, col_idx, scales, zero_points):
    q_data = (data[:, col_idx] / scales[col_idx]) + zero_points[col_idx]
    q_data = np.array(q_data, dtype=np.uint8)
    
    return q_data

def quantize_data(data, scales, zero_points):
    q_data = (data / scales) + zero_points
    q_data = np.array(q_data, dtype=np.uint8)
    
    return q_data


def run_static_quantization(preproc_onnx_model_path: Path,
                            calib_loader: CalibrationDataReader,
                            quantized_model_out_path: Path):

    quantize_static(
        model_input=preproc_onnx_model_path,
        model_output=quantized_model_out_path,
        calibration_data_reader=calib_loader,
        quant_format=QuantFormat.QOperator,  # Use QDQ format to quantize all including activations
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
    )
    
    

