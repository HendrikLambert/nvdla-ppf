import torch

from modules.dft_cnn_module import DFTCNNModule
from modules.fir_cnn_module import FIRCNNModule
from modules.pfb_module import PFBModule
from modules.fir_helper import ref_kaiser_weights


class PFBDFTModule(PFBModule):
    def __init__(self, P, M, batch_size: int):
        """
        Initializes the PFBDFTModule.
        Args:
            P (int): Number of channels.
            M (int): Number of taps for the FIR filter.
            batch_size (int): Batch size for processing.
        """

        super().__init__(P, M, batch_size)

        # Initialize FIR filter with predefined weights
        weights = ref_kaiser_weights(P, M, reversed=False)
        weights = [weights[i // 2] for i in range(0, 2 * P)]
        weights = torch.stack(weights, dim=0)

        self.FIR = FIRCNNModule(P, M, weights)

        self.DFT = DFTCNNModule(P)
