import torch.nn as nn


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

    def forward(self, x):
        x = self.FIR(x)
        x = self.DFT(x)
        return x
