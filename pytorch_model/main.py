import torch
import torch.nn as nn
from onnx_helper import export_model
from modules.pfb_fft_module import PFBFFTModule
from modules.pfb_dft_module import PFBDFTModule
from modules.test_module import TestModule

def export_pfb_fft():
    P = 32  # Number of channels
    M = 16   # Number of taps
    batch_size = 4

    # Initialize the PFB model with FFT
    # pfb_model = PFBModelDFT(P, M, batch_size)
    pfb_model = PFBFFTModule(P, M, batch_size)

    # Create example input tensor
    example_inputs = torch.zeros(batch_size, P*2, 1, M)
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
    M = 16   # Number of taps
    batch_size = 2

    # Initialize the PFB model with FFT
    # pfb_model = PFBModelDFT(P, M, batch_size)
    pfb_model = PFBDFTModule(P, M, batch_size)

    # Create example input tensor
    example_inputs = torch.zeros(batch_size, P*2, 1, M)
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

def main():
    # export_test_module()
    export_pfb_fft()
    
    


if __name__ == "__main__":
    main()