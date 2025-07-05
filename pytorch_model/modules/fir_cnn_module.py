import torch
import torch.nn as nn
from modules.fir_helper import generate_module_weights


class FIRCNNModule(nn.Module):
    def __init__(self, P: int, M: int, fir_weights: torch.Tensor | None = None):
        """
        Initializes the FIRCNNModule with FIR filter weights.

        This module uses a convolutional layer to apply the FIR filter across the input channels.

        Args:
            P (int): Number of channels.
            M (int): Number of taps for the FIR filter.
            fir_weights (torch.Tensor): FIR filter weights of shape (P*2, 1, 1, M).

        """
        super().__init__()
        
        # Initialize FIR filter weights if not provided
        if fir_weights is None:
            fir_weights = generate_module_weights(P, M, reversed=False)

        # Expected shape of fir_weights is (P * 2, 1, 1, M)
        shape = (P * 2, 1, 1, M)
        if fir_weights.shape != shape:
            raise ValueError(
                f"Expected fir_weights shape {shape}, got {fir_weights.shape}"
            )

        self.cnn = nn.Conv2d(P * 2, P * 2, kernel_size=(1, M), groups=P * 2, bias=False)
        self.cnn.weight = nn.Parameter(fir_weights)

        # Disable learning of weights
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the FIRCNNModule.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, P*2, 1, M).
        Returns:
            torch.Tensor: Output tensor after applying the FIR filter.
        """
        x = self.cnn(x)
        return x
