import torch

from models.dft_cnn_model import DFTModelCNN
from models.fir_model import FIRModel
from models.pfb_model import PFBModel
from models.fir_helper import ref_kaiser_weights


class PFBModelDFT(PFBModel):
    def __init__(self, P, M, batch_size: int):
        """
        Initializes the PFBModelDFT.
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

        self.FIR = FIRModel(P, M, weights)

        self.DFT = DFTModelCNN(P)
