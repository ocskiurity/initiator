<p align="center">
    <picture>
        <source srcset="https://github.com/ocskiurity/initiator/assets/14362976/91bebf98-fbe0-4a88-bf49-f6a023086e5e">
        <img width="250" alt="logo"
            src="https://github.com/ocskiurity/initiator/assets/14362976/91bebf98-fbe0-4a88-bf49-f6a023086e5e">
    </picture>
<h1 align="center">
    🗡️ initiator 🗡️
</h1>

A collection of utilities to train and quantize 1-layer MLP weights and input data.

## Getting started
Important: make sure [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) is installed in your system.

Create a conda environment with Python 3.9+ and install the required dependencies:
```bash
conda create --prefix=./venv python=3.9
conda activate ./venv
pip install -r requirements.txt
```

We provide two scripts to bootstrap the training and quantization of both model weights and input data.
1. First, we will train a lightweight 1-layer MLP on the [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/14/breast+cancer) dataset with:
```bash
./scripts/train.sh
```
After training is finished, a checkpoint will be saved at `checkpoint/MLP.json`.

1. Then, we will export the trained model to .onnx, perform static uint8 quantization with calibration and save both quantized model weights/bias and data input with:
```bash
./scripts/quantize.sh
```

After quantization is finished, you will find three new .onnx files in the  `checkpoint` directory:
- MLP.onnx: this is the onnx exported model
- MLP_quant_preproc.onnx: this is the preprocessed onnx model prior to quantization
- MLP_quant.onnx: this is the statically quantized model

Additionally, two .json files with quantized model weights/bias and data input are saved to `data/json/model.json` and `data/json/quantizedData.json`, respectively.
