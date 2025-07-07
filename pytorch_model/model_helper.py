import torch
import subprocess
import torch.nn as nn
import os
from onnx_helper import export_general_model, export_pfb_model
from modules.pfb_fft_module import PFBFFTModule
from modules.pfb_dft_module import PFBDFTModule
from modules.fir_cnn_module import FIRCNNModule
from modules.fft_cnn_module import FFTCNNModule
from modules.dft_cnn_module import DFTCNNModule

# Constants for batch sizes, models, channels, and taps
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
MODELS = ["pfb_fft", "pfb_dft", "fir", "dft", "fft"]
# MODELS = ["pfb_dft", "fir", "dft"]
# BATCH_SIZES = [2048, 4096, 6144, 8192]
# MODELS = ["fft"]
CHANNELS = 256
TAPS = 16


def build_onnx_benchmark_files(dir: str):
    """
    Build ONNX benchmark files for PFB FFT and DFT models with various batch sizes.

    :param dir: Directory to save the ONNX files
    """
    if not dir.endswith("/"):
        dir += "/"

    print(f"Building ONNX benchmark files in directory: {dir}")

    # Export ONNX models for each combination of batch size, channels, taps, and model type
    for b in BATCH_SIZES:
        for model_type in MODELS:
            model_name = generate_model_name(b, CHANNELS, TAPS, model_type)
            file_path = f"{dir}{model_name}.onnx"
            export_onnx_model(b, CHANNELS, TAPS, model_type, file_path)


def build_loadable_benchmark_files(
    onnx_dir: str, nvdla_dir: str, trtexe_location: str, verbose: bool = False
):
    """
    Build NVDLA loadable benchmark files from ONNX models.

    :param onnx_dir: Directory containing ONNX models
    :param nvdla_dir: Directory to save the NVDLA loadable files
    :param trtexe_location: Path to the trtexe binary
    :param verbose: Whether to enable verbose output
    """
    if not nvdla_dir.endswith("/"):
        nvdla_dir += "/"

    print(
        f"Building NVDLA loadable benchmark files in directory: {onnx_dir} to {nvdla_dir}"
    )

    # Iterate through folder and build loadable files for each ONNX model
    for f in os.listdir(onnx_dir):
        if f.endswith(".onnx"):
            onnx_file = os.path.join(onnx_dir, f)
            loadable_location = os.path.join(nvdla_dir, f.replace(".onnx", ".nvdla"))
            print(f"Building loadable for ONNX file: {onnx_file}")
            build_loadable_from_onnx(
                onnx_file, loadable_location, trtexe_location, verbose
            )


def generate_model_name(b: int, c: int, t: int, model_type: str) -> str:
    """
    Generate a model name based on the batch size, number of channels, number of taps, and model type.

    :param b: Batch size
    :param c: Number of channels
    :param t: Number of taps
    :param model_type: Type of model ('pfb_fft', 'pfb_dft', 'fir', 'dft', 'fft')
    :return: Formatted model name
    """

    if model_type == "pfb_fft":
        return f"pfb_model_fft-c{c}-t{t}-b{b}"
    elif model_type == "pfb_dft":
        return f"pfb_model_dft-c{c}-t{t}-b{b}"
    elif model_type == "fir":
        return f"fir_model-c{c}-t{t}-b{b}"
    elif model_type == "dft":
        return f"dft_model-c{c}-b{b}"
    elif model_type == "fft":
        return f"fft_model-c{c}-b{b}"

    raise ValueError("Invalid model type specified, valid options:" + str(MODELS))



def build_loadable_from_onnx(
    onnx_file: str, loadable_location: str, trtexe_location: str, verbose: bool
):
    """
    Build NVDLA loadable from ONNX file using trtexe.

    :param onnx_file: Path to the ONNX file
    :param loadable_location: Location to save the NVDLA loadable
    :param trtexe_location: Path to the trtexe binary
    :param verbose: Whether to enable verbose output
    """

    # trtexe --onnx=loc --verbose --fp16 --saveEngine=loadable.bin --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0
    command = [
        trtexe_location,
        "--onnx=" + onnx_file,
        "--saveEngine=" + loadable_location,
        "--fp16",
        "--inputIOFormats=fp16:chw16",
        "--outputIOFormats=fp16:chw16",
        "--buildDLAStandalone",
        "--useDLACore=0",
    ]
    if verbose:
        command.append("--verbose")

    print(f"Running command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error building NVDLA loadable:")
        print(result.stderr)
    else:
        print("NVDLA loadable built successfully.")
        print(result.stdout)


def export_onnx_model(b: int, c: int, t: int, model_type: str, file: str):
    """
    Export ONNX model for PFB FFT or DFT based on the provided parameters.

    :param b: Batch size
    :param c: Number of channels
    :param t: Number of taps
    :param model_type: Type of model to export
    :param file: The location to save the ONNX file
    """

    # Check if the file name ends with '.onnx'
    if not file.endswith(".onnx"):
        raise ValueError("The file name should end with '.onnx'")
    
    print(f"Exporting {model_type} model with batch size {b}, channels {c}, taps {t} to {file}")

    if model_type == "pfb_fft":
        model = PFBFFTModule(c, t, b)
        export_pfb_model(model, file)
    elif model_type == "pfb_dft":
        model = PFBDFTModule(c, t, b)
        export_pfb_model(model, file)
    elif model_type == "fir":
        model = FIRCNNModule(c, t)
        export_general_model(model, file, shape=(b, c * 2, 1, t))
    elif model_type == "dft":
        model = DFTCNNModule(c)
        export_general_model(model, file, shape=(b, c * 2, 1, 1))
    elif model_type == "fft":
        model = FFTCNNModule(c)
        export_general_model(model, file, shape=(b, c * 2, 1, 1))
    else:
        raise ValueError("Invalid model type specified, valid options:" + str(MODELS))


def export_test_module():

    class ExampleModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            input_channel_slices = list(torch.split(x, 1, dim=1))
            x = torch.cat(input_channel_slices, dim=1)
            return x

    module = ExampleModule()
    example_inputs = torch.rand(2, 4, 1, 1)

    torch.onnx.export(
        module,
        example_inputs,
        "test_module.onnx",
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=15,
    )
