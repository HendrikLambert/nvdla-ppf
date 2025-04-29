import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import scipy.signal as signal

import numpy as np

REF_IMPLEMENTATION = "../reference/polyphase-filter-bank-generator/polyphase-filter-bank-generator"
OPTIONS = {
    "NONE": 0,
    "PRINT_WEIGHTS": 1 << 0,
    "REVERSED_WEIGHTS": 1 << 1,
    "TEST_FILTER": 1 << 2,
    "TEST_PPF": 1 << 3
}

class FIRModel(nn.Module):
    def __init__(self, P, M, fir_weights: torch.Tensor):
        """
        
        P: Number of channels
        M: Number of taps
        
        """
        
        super().__init__()
        
        self.cnn = nn.Conv2d(P, P, kernel_size=(1, M), groups=P)
        print(self.cnn.weight.shape)
        self.cnn.weight = nn.Parameter(fir_weights)
        self.cnn.bias = nn.Parameter(torch.from_numpy(np.zeros(self.cnn.weight.shape[0]).astype(np.float32)))
        # print("weight", self.cnn.weight.shape)
        # print("bias", self.cnn.bias.shape)

        # print(self.cnn.weight.shape)
    
    def forward(self, x):
        print("in", x.shape)
        # x = x.view(-1, 256, 16, 1)
        # print("view", x.shape)
        x = self.cnn(x)
        print("cnn", x.shape)
        # x = x.transpose(1, 2)
        # print(x.shape)
        return x


class LocalFFTModel(nn.Module):
    
    def __init__(self, P):
        super().__init__()
        self.P = P
            
    def forward(self, x: torch.Tensor):
        assert x.shape[0] == 2, "Input tensor must have size of 2, doesn't support batch size yet"
        x = x.reshape(2, self.P).T
        
        x = x.contiguous()
        
        x = torch.view_as_complex(x)
        x = torch.fft.fft(x)
        
        return x


class FFTModel(nn.Module):
    def __init__(self, P, M):
        """
        Inspired by
        https://github.com/OliVis/coral/blob/main/scripts/fft_model.py
        
        """
        
        super().__init__()
        
        self.FFT = nn.Conv2d(P, P, kernel_size=(1, M), groups=P)
    
    def forward(self, x):
        x = self.FFT(x)
        return x
    
    
    
class PPFModel(nn.Module):
    def __init__(self, P, M, fir_weights: torch.Tensor):
        """
        
        P: Number of channels
        M: Number of taps
        
        """
        
        super().__init__()
        
        self.FIR = FIRModel(P, M, fir_weights)
    
    def forward(self, x):
        x = self.FIR(x)
        return x
    



def create_kaiser_weights(nrChannels:int, nrTaps:int, beta:float = 9.0695, cutoff:float = None):
    if cutoff is None:
        cutoff = 1.0 / nrChannels
        
    # kaiser = signal.windows.kaiser(nrChannels * nrTaps, beta)
    kaiser = signal.firwin(nrTaps * nrChannels, cutoff=cutoff, window=('kaiser', beta), scale=False)
    # Scale peak to 1.0
    kaiser = kaiser / np.max(kaiser)
    weights = torch.tensor(kaiser, dtype=torch.float32)
    # print("weights", weights.shape)
    weights = weights.reshape((nrTaps, nrChannels)).T
    # print("weights", weights.shape)
    weights = weights.reshape(nrChannels, 1, 1, nrTaps)
    
    return weights

    
def plot_kaiser_weights(P=256, M=16):
    # plt.plot(create_kaiser_weights(P, M).reshape(-1), label="Kaiser Weights")
    weights = create_kaiser_weights(P, M).T.reshape(-1)
    ref_weights = ref_kaiser_weights(P, M, reversed=False).T.reshape(-1)
    plt.plot(weights, label="Kaiser Weights")
    plt.plot(ref_weights, label="Reference Weights")
    plt.plot(create_kaiser_weights(P, M, cutoff=1.0/(P-11)).T.reshape(-1), label="Kaiser Weights adjusted")
    plt.title("Kaiser Window")
    plt.legend()
    plt.show()


def ref_kaiser_weights(nrChannels:int, nrTaps:int, type:str = "KAISER", reversed:bool = False):
    options = OPTIONS["PRINT_WEIGHTS"] | (OPTIONS["REVERSED_WEIGHTS"] if reversed else 0)
    
    res = subprocess.run([REF_IMPLEMENTATION, str(nrChannels), str(nrTaps), type, str(options)], capture_output=True)
    res.check_returncode()
    weights = [float(x) for x in res.stdout.decode().splitlines()]
    weights = torch.tensor(weights, dtype=torch.float32)
    # print("weights", weights.shape)
    weights = weights.reshape((nrTaps, nrChannels)).T
    # print("weights", weights.shape)
    weights = weights.reshape(nrChannels, 1, 1, nrTaps)
    # print("weights", weights.shape)
    
    return weights
    

def test_fir_filter():
    P = 6 # channels out
    M = 4 # taps
    ref_weights = ref_kaiser_weights(P, M, reversed=False)
    
    # print(ref_weights)
    # print(ref_weights[0, :, :, :])
    
    ppf_model = FIRModel(P, M, ref_weights)
    
    example_inputs = torch.zeros(2, P, 1, M)
    example_inputs[:, 0, 0, 3] = 1.0
    
    out = ppf_model(example_inputs)
    # # print(out[0, -1, 0, 0].item())
    print(out)
    

def test_local_ppf():
    P = 256
    M = 16
    ref_weights = ref_kaiser_weights(P, M, reversed=True)
    ppf_model = PPFModel(P, M, ref_weights)
    
    example_inputs = torch.zeros(2, 256, 1, 16)
    example_inputs[0, 0, 0, 0] = 1.0
    # example_inputs[0, 0, 0, 0] = 1.0
    # example_inputs[0, 0, 0, 0] = 2.0
    
    out = ppf_model(example_inputs)
    
    out = LocalFFTModel(P)(out)
    
    print(out)
    

def export_model():
    P = 256
    M = 16
    ref_weights = ref_kaiser_weights(P, M, reversed=False)
    
    ppf_model = PPFModel(P, M, ref_weights)
    
    
    example_inputs = torch.zeros(2, 256, 1, 16) 
    # example_inputs[0, 0, 0, 0] = 1.0
    # example_inputs[0, 0, 0, 1] = 1.0
    example_inputs[0, 10, 0, -1] = 1.0

    out = ppf_model(example_inputs)    
    print(out[0, 10, 0, 0].item())
    

    # torch.onnx.export(ppf_model, example_inputs, "ppf.onnx",
    #               input_names=['input'],
    #               output_names=['output'],
    #             #   dynamic_axes={
    #             #     'input': {0: 'batch_size'},
    #             #     'output': {0: 'batch_size'}
    #             #   }
    #             )

    
def main():
    export_model()
    



if __name__ == "__main__":
    main()