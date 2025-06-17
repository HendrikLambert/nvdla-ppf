import unittest
import torch
from modules.fft_cnn_module import FFTCNNModule


class TestFFTCNNModule(unittest.TestCase):
    """
    Tests for the FFTCNNModule.
    """

    def _get_fft_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the reference FFT output using torch.fft.fft and formats it
        to match the interleaved real/imaginary channel format of the module.

        Args:
            input_tensor (torch.Tensor): Tensor of shape (B, 2*N, 1, 1).

        Returns:
            torch.Tensor: Reference FFT output in the same shape as the input.
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
        runs a forward pass, and compares it to the reference FFT.
        """
        model = FFTCNNModule(N=N)
        
        input_tensor = torch.randn(batch_size, 2 * N, 1, 1)

        output_model = model(input_tensor)
        output_ref = self._get_fft_reference(input_tensor)

        self.assertEqual(output_model.shape, input_tensor.shape)
        self.assertTrue(
            torch.allclose(output_model, output_ref, atol=1e-5),
            msg=(
                f"FFT(N={N}, B={batch_size}) output values do not match reference.\n"
                f"Max absolute difference: {torch.max(torch.abs(output_model - output_ref)).item():.6g}"
            ),
        )

    def test_fft_correctness_N2(self):
        """Tests the edge case N=2 (FFT should be identity)."""
        self._run_correctness_test(N=2, batch_size=10)

    def test_fft_correctness_N4(self):
        """Tests a small FFT size N=4."""
        self._run_correctness_test(N=4, batch_size=5)

    def test_fft_correctness_N16(self):
        """Tests a medium FFT size N=16 with a larger batch."""
        self._run_correctness_test(N=16, batch_size=20)

    def test_fft_correctness_N64(self):
        """Tests a larger FFT size N=64."""
        self._run_correctness_test(N=64, batch_size=3)

    def test_raises_error_for_non_power_of_two(self):
        """
        Tests that the module's constructor raises a ValueError if N is not a power of 2.
        """
        # Test with N=12, which is not a power of two
        with self.assertRaisesRegex(ValueError, "N must be a positive power of 2"):
            FFTCNNModule(N=12)

        # Test with N=0
        with self.assertRaisesRegex(ValueError, "N must be a positive power of 2"):
            FFTCNNModule(N=0)

        # Test with a negative number
        with self.assertRaisesRegex(ValueError, "N must be a positive power of 2"):
            FFTCNNModule(N=-8)
