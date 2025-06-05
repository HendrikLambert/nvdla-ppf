import unittest
import torch
import torch.nn as nn
from models.dft_cnn_model import DFTModelCNN
import numpy as np


class TestDFTCNNModel(unittest.TestCase):
    def setUp(self):
        self.P = 32  # Number of channels
        self.model = DFTModelCNN(self.P)
        self.input_tensor_b1 = torch.randn(1, self.P * 2, 1, 1)
        self.input_tensor_b20 = torch.randn(20, self.P * 2, 1, 1)

    def test_forward_shape_b1(self):
        output = self.model(self.input_tensor_b1)
        expected_shape = (1, self.P * 2, 1, 1)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_shape_b20(self):
        output = self.model(self.input_tensor_b20)
        expected_shape = (20, self.P * 2, 1, 1)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_values_b1(self):
        output_model = self.model(self.input_tensor_b1)
        output_model = output_model

        # Create reference values
        input_ref = self.input_tensor_b1.squeeze(-1).squeeze(
            -1
        )  # Remove unnecessary dimensions
        # Slice to real and imaginary parts
        input_ref_real = input_ref[0, 0::2]
        input_ref_imag = input_ref[0, 1::2]
        input_complex = input_ref_real + 1j * input_ref_imag

        output_ref_complex = torch.fft.fft(input_complex)
        output_ref = torch.stack(
            (output_ref_complex.real, output_ref_complex.imag), dim=1
        ).flatten()
        output_ref = output_ref.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Check if the output values are close to the reference values
        self.assertTrue(
            torch.allclose(output_model, output_ref, rtol=1e-4),
            msg="Output values do not match reference values within tolerance. " + str(torch.max(torch.abs(output_model - output_ref))),
        )

    def test_forward_values_b20(self):
        output_model = self.model(self.input_tensor_b20)
        output_model = output_model[3].squeeze(0)  # Take only the third batch

        # Create reference values
        input_ref = self.input_tensor_b20.squeeze(-1).squeeze(
            -1
        )  # Remove unnecessary dimensions
        # Slice to real and imaginary parts
        input_ref_real = input_ref[3, 0::2]
        input_ref_imag = input_ref[3, 1::2]
        input_complex = input_ref_real + 1j * input_ref_imag

        output_ref_complex = torch.fft.fft(input_complex)
        output_ref = torch.stack(
            (output_ref_complex.real, output_ref_complex.imag), dim=1
        ).flatten()
        output_ref = output_ref.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Check if the output values are close to the reference values
        self.assertTrue(
            torch.allclose(output_model, output_ref, rtol=1e-4),
            msg="Output values do not match reference values within tolerance. " + str(torch.max(torch.abs(output_model - output_ref))),
        )
