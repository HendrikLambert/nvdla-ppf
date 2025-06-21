import torch
import subprocess
import torch.nn as nn
import os
from onnx_helper import export_model
from modules.pfb_fft_module import PFBFFTModule
from modules.pfb_dft_module import PFBDFTModule

# Constants for batch sizes, models, channels, and taps
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MODELS = ["pfb_fft", "pfb_dft"]
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
            file_path = f"{dir}/{model_name}.onnx"
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
    :param model_type: Type of model ('pfb_fft' or 'pfb_dft')
    :return: Formatted model name
    """

    if model_type == "pfb_fft":
        return f"pfb_model_fft-c{c}-t{t}-b{b}"
    elif model_type == "pfb_dft":
        return f"pfb_model_dft-c{c}-t{t}-b{b}"

    raise ValueError("Invalid model type specified. Use 'pfb_fft' or 'pfb_dft'.")


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
    :param model_type: Type of model to export ('pfb_fft' or 'pfb_dft')
    :param file: The location to save the ONNX file
    """

    # Check if the file name ends with '.onnx'
    if not file.endswith(".onnx"):
        raise ValueError("The file name should end with '.onnx'")

    if model_type == "pfb_fft":
        pfb_model = PFBFFTModule(c, t, b)
    elif model_type == "pfb_dft":
        pfb_model = PFBDFTModule(c, t, b)
    else:
        raise ValueError("Invalid model type specified. Use 'pfb_fft' or 'pfb_dft'.")

    print(
        f"Exporting {model_type} model with batch size {b}, channels {c}, taps {t} to {file}"
    )

    # Export the model
    export_model(pfb_model, file)


def export_pfb_fft():
    P = 8  # Number of channels
    M = 16  # Number of taps
    batch_size = 4

    # Initialize the PFB model with FFT
    # pfb_model = PFBModelDFT(P, M, batch_size)
    pfb_model = PFBFFTModule(P, M, batch_size)

    # Create example input tensor
    example_inputs = torch.zeros(batch_size, P * 2, 1, M)
    example_inputs[:, 0, 0, 0] = 1.0
    example_inputs[:, 1, 0, 0] = 2.0

    # Forward pass through the model
    out = pfb_model(example_inputs)

    print("Output shape:", out.shape)
    print("Output values:", out.squeeze())

    export_model(pfb_model, "pfb_model_fft", onnx_version=18)
    # export_model(pfb_model, "pfb_model_dft.onnx")


def export_pfb_dft():
    P = 16  # Number of channels
    M = 16  # Number of taps
    batch_size = 2

    # Initialize the PFB model with FFT
    # pfb_model = PFBModelDFT(P, M, batch_size)
    pfb_model = PFBDFTModule(P, M, batch_size)

    # Create example input tensor
    example_inputs = torch.zeros(batch_size, P * 2, 1, M)
    example_inputs[:, 0, 0, 0] = 1.0
    example_inputs[:, 1, 0, 0] = 2.0

    # Forward pass through the model
    out = pfb_model(example_inputs)

    print("Output shape:", out.shape)
    print("Output values:", out.squeeze())

    export_model(pfb_model, "pfb_model_dft.onnx", onnx_version=18)
    # export_model(pfb_model, "pfb_model_dft.onnx")


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
