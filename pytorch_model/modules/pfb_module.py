import torch
import torch.nn as nn
from modules.fir_helper import ref_kaiser_weights

class PFBModule(nn.Module):
    def __init__(self, P: int, M: int, batch_size: int):
        """
        Initializes the PFBModule.
        Args:
            P (int): Number of channels.
            M (int): Number of taps for the FIR filter.
            batch_size (int): Batch size for processing.
            fir_weights (torch.Tensor): Predefined FIR filter weights.

        """

        super().__init__()
        self.P = P
        self.M = M
        self.batch_size = batch_size
        
        # Initialize FIR filter with predefined weights
        self.weights = ref_kaiser_weights(P, M, reversed=False)
        # Line below is commented out because it reshapes the weights in a way that is not needed for the current implementation.
        # self.weights = self.weights.reshape(P, 1, M, 1)  # Reshape to (P, 1, M, 1)
        
        # Duplicate weights for complex channels, both channels have the same FIR filter
        # so we can duplicate the weights for both real and imaginary parts.
        self.weights = [self.weights[i // 2].T for i in range(0, 2 * P)]
        self.weights = torch.stack(self.weights, dim=0)

    def forward(self, x):
        x = self.FIR(x)
        x = self.DFT(x)
        return x
