from modules.dft_cnn_module import DFTCNNModule
from modules.fir_cnn_module import FIRCNNModule
from modules.pfb_module import PFBModule


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

        self.FIR = FIRCNNModule(P, M, self.weights)

        self.DFT = DFTCNNModule(P)
