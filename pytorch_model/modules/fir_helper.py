import torch 
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