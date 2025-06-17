import unittest
import numpy as np
import torch
from modules.dft_helper import create_dft_matrix


class TestDFTHelper(unittest.TestCase):
    """
    Tests for the DFT helper functions.
    """

    def test_create_dft_matrix(self):
        N = 8
        dft_matrix = create_dft_matrix(N)
        expected_shape = (2 * N, 2 * N)
        self.assertEqual(dft_matrix.shape, expected_shape)
        self.assertTrue(
            dft_matrix.dtype == torch.float32, "Matrix should be of type float32"
        )
