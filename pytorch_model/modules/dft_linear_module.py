import torch
import torch.nn as nn
from dft_helper import create_dft_matrix


class DFTLinearModule(nn.Module):
    def __init__(self, P: int):
        """
        Initializes the DFTLinearModule.

        Args:
            P (int): Number of channels. Must be a positive integer.
        """

        super().__init__()
        # Only check if P is a positive integer, as the DFT is defined for all positive integers.
        assert P > 0, "P must be a positive integer"

        self.P = P
        self.in_out_features = 2 * P

        # Create the nn.Linear layer. Bias is False because DFT is a linear transformation without offset.
        self.fc = nn.Linear(self.in_out_features, self.in_out_features, bias=False)

        # Compute and set the fixed DFT weights
        self.fc.weight.data = create_dft_matrix(P)

        # Disable learning of weights
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the DFT.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2*P)
                              representing interleaved real and imaginary parts.
                              E.g., [Re(x0), Im(x0), Re(x1), Im(x1), ..., Re(xn-1), Im(xn-1)]

        Returns:
            torch.Tensor: Output tensor of the same shape as input, representing
                          the DFT coefficients in interleaved real/imaginary format.
                          E.g., [Re(X0), Im(X0), Re(X1), Im(X1), ..., Re(Xn-1), Im(Xn-1)]
        """

        x = self.fc(x)

        return x
