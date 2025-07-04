import torch
import torch.nn as nn
from modules.dft_helper import create_dft_matrix


class DFTCNNModule(nn.Module):

    def __init__(self, N: int):
        """
        Initializes the DFTCNNModule.

        Args:
            N (int): Number of channels for the DFT.
        """

        super().__init__()
        # Only check if N is a positive integer, as the DFT is defined for all positive integers.
        assert N > 0, "N must be a positive integer"

        self.N = N
        self.in_out_features = 2 * N

        self.cnn = nn.Conv2d(
            self.in_out_features, self.in_out_features, kernel_size=1, bias=False
        )

        # Compute and set the fixed DFT weights
        dft_weights = create_dft_matrix(N).unsqueeze(-1).unsqueeze(-1)
        self.cnn.weight.data = dft_weights

        # Disable learning of weights
        self.cnn.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Performs the DFT.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2*N)
                              representing interleaved real and imaginary parts.
                              E.g., [Re(x0), Im(x0), Re(x1), Im(x1), ..., Re(xn-1), Im(xn-1)]

        Returns:
            torch.Tensor: Output tensor of the same shape as input, representing
                          the DFT coefficients in interleaved real/imaginary format.
                          E.g., [Re(X0), Im(X0), Re(X1), Im(X1), ..., Re(Xn-1), Im(Xn-1)]
        """
        x = self.cnn(x)
        return x
