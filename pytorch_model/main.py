import torch
from models.pfb_fft_model import PFBModelFFT
from models.pfb_dft_model import PFBModelDFT
from onnx_helper import export_model


def main():
    P = 256  # Number of channels
    M = 16   # Number of taps
    batch_size = 2

    # Initialize the PFB model with FFT
    pfb_model = PFBModelDFT(P, M, batch_size)

    # Create example input tensor
    example_inputs = torch.zeros(batch_size, P*2, 1, M)
    example_inputs[:, 0, 0, 0] = 1.0
    example_inputs[:, 1, 0, 0] = 2.0

    # Forward pass through the model
    out = pfb_model(example_inputs)

    print("Output shape:", out.shape)
    print("Output values:", out.squeeze())
    
    # export_model(pfb_model, batch_size, "pfb_model_fft.onnx")
    export_model(pfb_model, batch_size, "pfb_model_dft.onnx")


if __name__ == "__main__":
    main()