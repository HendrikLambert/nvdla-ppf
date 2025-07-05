from modules.fft_cnn_module import FFTCNNModule
from modules.fir_cnn_module import FIRCNNModule
from modules.pfb_module import PFBModule


class PFBFFTModule(PFBModule):
    def __init__(self, P, M, batch_size: int):
        """
        Initializes the PFBFFTModule.
        Args:
            P (int): Number of channels.
            M (int): Number of taps for the FIR filter.
            batch_size (int): Batch size for processing.
        """

        super().__init__(P, M, batch_size)

        self.FIR = FIRCNNModule(P, M, self.weights)

        self.DFT = FFTCNNModule(P)
