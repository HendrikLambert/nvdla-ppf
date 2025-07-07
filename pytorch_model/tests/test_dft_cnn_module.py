import unittest
import torch
from modules.dft_cnn_module import DFTCNNModule


class TestDFTCNNModule(unittest.TestCase):
    """
    Tests for the DFTCNNModule.
    """

    def _get_dft_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the reference DFT output using torch.fft.fft and formats it
        to match the interleaved real/imaginary channel format of the module.

        Args:
            input_tensor (torch.Tensor): Tensor of shape (B, 2*N, 1, 1).

        Returns:
            torch.Tensor: Reference DFT output in the same shape as the input.
        """
        self.assertEqual(
            len(input_tensor.shape), 4, "Input tensor must have 4 dimensions"
        )
        batch_size, num_features, H, W = input_tensor.shape

        # Reshape and combine interleaved channels into a complex tensor
        # Get correct shape (B, 2*N, 1, 1) -> (B, 2*N)
        input_squeezed = input_tensor.squeeze(-1).squeeze(-1)
        # Get real and imaginary parts
        re_part = input_squeezed[:, 0::2]
        im_part = input_squeezed[:, 1::2]

        input_complex = torch.complex(re_part, im_part)

        # Compute the reference DFT
        output_complex = torch.fft.fft(input_complex, dim=1)

        # Split the complex output back into interleaved real/imaginary channels
        # Shape of real,imag parts (B, N)
        out_re = output_complex.real
        out_im = output_complex.imag
        # Stack to (B, N, 2) and flatten to (B, 2*N) to interleave
        output_interleaved = torch.stack((out_re, out_im), dim=2).flatten(start_dim=1)

        # Reshape to match the original shape (B, 2*N, 1, 1)
        return output_interleaved.view(batch_size, num_features, H, W)

    def _run_correctness_test(self, N: int, batch_size: int):
        """
        A generic test function that creates a model for a given N,
        runs a forward pass, and compares it to the reference DFT.
        """
        model = DFTCNNModule(N=N)

        input_tensor = torch.randn(batch_size, 2 * N, 1, 1) # Scale for numerical stability

        output_model = model(input_tensor)
        output_ref = self._get_dft_reference(input_tensor)

        self.assertEqual(output_model.shape, input_tensor.shape)
        self.assertTrue(
            torch.allclose(output_model, output_ref, atol=1e-5, rtol=1e-5), 
            msg=(
                f"DFT(N={N}, B={batch_size}) output values do not match reference.\n"
                f"Max absolute difference: {torch.max(torch.abs(output_model - output_ref)).item():.6g} \n"
                f"Max relative difference: {torch.max(torch.abs((output_model - output_ref) / output_ref)).item():.6g}"
            ),
        )

    def test_dft_correctness_N1(self):
        """Tests the edge case N=1 (DFT should be identity)."""
        self._run_correctness_test(N=1, batch_size=10)

    def test_dft_correctness_N4(self):
        """Tests a small DFT size N=4."""
        self._run_correctness_test(N=4, batch_size=5)

    def test_dft_correctness_N16(self):
        """Tests a medium DFT size N=16 with a larger batch."""
        self._run_correctness_test(N=16, batch_size=100)

    def test_dft_correctness_N17(self):
        """Tests a larger DFT size N=17."""
        self._run_correctness_test(N=17, batch_size=3)
