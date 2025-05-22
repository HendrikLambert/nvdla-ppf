import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import scipy.signal as signal
import onnx 
from onnx import numpy_helper, helper

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
        
        self.cnn = nn.Conv2d(P*2, P*2, kernel_size=(1, M), groups=P*2, bias=False)
        # print(self.cnn.weight.shape)
        self.cnn.weight = nn.Parameter(fir_weights)
        # self.cnn.bias = nn.Parameter(torch.from_numpy(np.zeros(self.cnn.weight.shape[0]).astype(np.float32)))
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
        # x = x.reshape(-1, 1, 1, 256)
        # x = x[0:10, 0:128, 0:1, 0:1]
        
        # x = x[:, 0:1]
        # print("x", x.shape)
        # x2 = x[1, :, :, :]
        # x1 = x1.reshape(2, 1, 128, 1)
        # x2 = x2.reshape(1, 1, 256, 1)
        # x1 = x[:, [0, 1], :, :]
        
        # x1 = torch.select(x, 2, 0)
        
        # x = torch.cat((x, x), dim=1)
        # print("cat", x.shape)
        # print("x1", x1.shape)
        # x = x1
        # x = x.reshape(2, 8, 1, 16)
        # x = x1
        # print("out", x.shape)
        return x


class LocalFFTModel(nn.Module):
    
    def __init__(self, P):
        super().__init__()
        self.P = P
            
    def forward(self, x: torch.Tensor):
        print("LocalFFT in", x.shape)
        x = torch.fft.rfft2(x)
        print("LocalFFT out", x.shape)
        
        return x


class DFTModelCNN(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.P = P
        self.cnn = nn.Conv2d(P*2, P*2, kernel_size=(1, 1), bias=False)
        dft_weights = create_dft_matrix(P).unsqueeze(-1).unsqueeze(-1)
        self.cnn.weight = nn.Parameter(dft_weights)
        self.cnn.bias = nn.Parameter(torch.from_numpy(np.zeros(self.cnn.weight.shape[0]).astype(np.float32)))
        # print(self.cnn.weight.shape)
        # print(dft_weights.shape)
        
        # self.cnn.weight = nn.Parameter(fir_weights)

    def forward(self, x: torch.Tensor):
        print("LocalFFT in", x.shape)
        # x = torch.fft.rfft2(x)
        x = self.cnn(x)
        print("LocalFFT out", x.shape)
        
        return x
    

class DFTModelLinear(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        # Only check if n is a positive integer, as the DFT is defined for all positive integers.        
        assert n > 0, "n must be a positive integer"

        self.n = n
        self.in_out_features = 2 * n

        # Create the nn.Linear layer. Bias is False because DFT is a linear transformation without offset.
        self.fc = nn.Linear(self.in_out_features, self.in_out_features, bias=False)

        # Compute and set the fixed DFT weights
        self.fc.weight.data = create_dft_matrix(n)
        # self.fc.bias.data = torch.zeros(self.in_out_features, dtype=torch.float32)
        
        # Make the weights non-trainable
        self.fc.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the DFT.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2*n) or (2*n),
                              representing interleaved real and imaginary parts.
                              E.g., [Re(x0), Im(x0), Re(x1), Im(x1), ..., Re(xn-1), Im(xn-1)]

        Returns:
            torch.Tensor: Output tensor of the same shape as input, representing
                          the DFT coefficients in interleaved real/imaginary format.
                          E.g., [Re(X0), Im(X0), Re(X1), Im(X1), ..., Re(Xn-1), Im(Xn-1)]
        """
        
        x = self.fc(x)
            
        return x


class LocalPPFModel(nn.Module):
    def __init__(self, P, M, fir_weights: torch.Tensor):
        """
        
        P: Number of channels
        M: Number of taps
        
        """
        
        super().__init__()
        
        self.FIR = FIRModel(P, M, fir_weights)
        
        self.DFT = LocalFFTModel(P)
    
    def forward(self, x):
        x = self.FIR(x)
        x = self.DFT(x)
        return x
    
    
class PPFModel(nn.Module):
    def __init__(self, P, M, batch_size:int, fir_weights: torch.Tensor):
        """
        
        P: Number of channels
        M: Number of taps
        
        """
        
        super().__init__()
        self.P = P
        self.batch_size = batch_size
        
        
        self.FIR = FIRModel(P, M, fir_weights)
        
        self.DFT = DFTModelCNN(P)
    
    def forward(self, x):
        x = self.FIR(x)
        print("FIR out", x.shape)
        # x = x.reshape(self.batch_size, 1, 1, 2*self.P)
        # print("reshape", x.shape)
        x = self.DFT(x)
        print("DFT out", x.shape)
        return x
    

def create_dft_matrix(n: int) -> torch.Tensor:
        """
        Creates the (2n x 2n) real-valued matrix that performs the n-point DFT
        on interleaved real/imaginary inputs.
        """
        matrix = torch.zeros((2*n, 2*n), dtype=torch.float32)

        for k in range(n):  # Index for the output DFT coefficient X_k
            for j in range(n):  # Index for the input sample x_j
                # Twiddle factor W_n^{kj} = cos(2*pi*k*j/n) - i*sin(2*pi*k*j/n)
                angle = 2 * np.pi * k * j / n
                re_w = np.cos(angle)
                im_w = -np.sin(angle) # Note: W_n^{kj} standard definition

                # Contribution of x_j = Re(x_j) + i*Im(x_j) to X_k = Re(X_k) + i*Im(X_k)
                # X_k = sum_j (Re(x_j) + i*Im(x_j)) * (re_w + i*im_w)
                #     = sum_j [ (Re(x_j)*re_w - Im(x_j)*im_w) + i*(Re(x_j)*im_w + Im(x_j)*re_w) ]

                # Row for Re(X_k) is 2*k
                # Col for Re(x_j) is 2*j
                # Col for Im(x_j) is 2*j + 1
                matrix[2*k, 2*j]   = re_w   # Re(x_j) * Re(W)
                matrix[2*k, 2*j+1] = -im_w  # -Im(x_j) * Im(W) (since Im(W) for X_k is positive: Im(x_j)*(-Im(W_twiddle)))
                                                # or easier: Re(x_j)*re_w - Im(x_j)*im_w.
                                                # The weight for Im(x_j) is -im_w.

                # Row for Im(X_k) is 2*k + 1
                matrix[2*k+1, 2*j]   = im_w   # Re(x_j) * Im(W)
                matrix[2*k+1, 2*j+1] = re_w   # Im(x_j) * Re(W)
        
        return matrix


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
    batch_size = 2
    ref_weights = ref_kaiser_weights(P, M, reversed=True)
    ppf_model = PPFModel(P, M, batch_size, ref_weights)

    example_inputs = torch.zeros(batch_size, 256, 1, 16)
    example_inputs[0, 0, 0, 0] = 1.0
    # example_inputs[0, 0, 0, 0] = 1.0
    # example_inputs[0, 0, 0, 0] = 2.0
    
    out = ppf_model(example_inputs)
    
    out = LocalFFTModel(P)(out)
    
    print(out)
    

def test_dft():
    batch_size = 1
    P = 16
    linearDFT = DFTModelLinear(P)
    cnnDFT = DFTModelCNN(P)

    # [Re(x0), Im(x0), Re(x1), Im(x1), ..., Re(xn-1), Im(xn-1)]
    example_input = torch.zeros(batch_size, 2*P)
    # example_input[0, 0] = 1.0
    example_input[0, 1] = 1.0
    example_input[0, 2] = 2.0
    example_input[1, 1] = 1.0
    example_input[1, 2] = 2.0


    # Turn to torch complex 
    real_parts = example_input[:, 0::2]
    imag_parts = example_input[:, 1::2]
    complex_input = torch.complex(real_parts, imag_parts)
    
    #### LINEAR DFT
    # Turn into input the linearDFT expects
    transformed_input = example_input.unsqueeze(-1).unsqueeze(-1)
    transformed_input = transformed_input.reshape(batch_size, 1, 1, 2*P)
    linear_out = linearDFT(transformed_input)
    
    ### CNN DFT
    # Turn into input the cnnDFT expects
    transformed_input = example_input.unsqueeze(-1).unsqueeze(-1)
    cnn_out = cnnDFT(transformed_input)
    cnn_out = cnn_out.reshape(batch_size, 1, 1, 2*P)
    
    
    
    ##### REF
    reference_out = torch.fft.fft(complex_input)
    
    # Turn model to complex values
    out_real = linear_out[:, 0::2]
    out_imag = linear_out[:, 1::2]
    complex_out = torch.complex(out_real, out_imag)
    
    
    print(linear_out)
    print(cnn_out)
    print(reference_out)
    print(linear_out.shape)
    print(cnn_out.shape)
    print(reference_out.shape)
    
    
    


def export_model():
    batch_size = 1
    P = 256
    M = 16
    # ref_weights = torch.zeros(P*2, 1, 1, M)
    ref_weights = ref_kaiser_weights(P, M, reversed=False)
    # ref_weights2 = ref_weights.unsqueeze(1)
    
    # Weights twice for Re and Im
    weights = [ref_weights[i//2] for i in range (0, 2*P)]
    weights = torch.stack(weights, dim=0)

    # lppf_model = LocalPPFModel(P, M, ref_weights)
    ppf_model = PPFModel(P, M, batch_size, weights)

    example_inputs = torch.zeros(batch_size, 2*P, 1, M) 
    example_inputs[0, 0, 0, 0] = 1.0
    example_inputs[0, 1, 0, 0] = 2.0
    # example_inputs[0, 0, 0, 1] = 1.0

    # # lout = lppf_model(example_inputs)    
    out = ppf_model(example_inputs)    
    # # print(out[0, 0, 0, 0].item())
    # # print(lout)
    print(out)
    
    
    # # print
    ONNX_FILE = "ppf.onnx"
    torch.onnx.export(ppf_model, example_inputs, ONNX_FILE,
                  input_names=['input'],
                  output_names=['output'],
                  do_constant_folding=True,
                  opset_version=15,
                #   dynamic_axes={
                #     'input': {0: 'batch_size'},
                #     'output': {0: 'batch_size'}
                #   }
                )
    
    # Post processing
    # post_process_onnx(ONNX_FILE)

def extract_constant_value(node):
    """Extract the value from a Constant node."""
    for attr in node.attribute:
        if attr.name == 'value':
            return numpy_helper.to_array(attr.t)
    raise ValueError(f"No 'value' attribute found in Constant node: {node.name}")

def convert_slice_to_opset1(model):
    graph = model.graph

    # Collect initializers and Constant nodes as name → value
    const_values = {}
    node_map = {n.output[0]: n for n in graph.node if n.op_type == 'Constant'}
    for output_name, node in node_map.items():
        const_values[output_name] = extract_constant_value(node)

    new_nodes = []
    used_const_inputs = set()

    for node in graph.node:
        if node.op_type == 'Slice' and len(node.input) >= 3:
            data_input = node.input[0]
            starts_input = node.input[1]
            ends_input = node.input[2]
            axes_input = node.input[3] if len(node.input) > 3 else None 
            steps = node.input[4] if len(node.input) > 4 else None

            if all(i in const_values for i in [starts_input, ends_input]) and \
               (axes_input is None or axes_input in const_values):
                starts = const_values[starts_input].tolist()
                ends = const_values[ends_input].tolist()
                axes = const_values[axes_input].tolist() if axes_input else list(range(len(starts)))

                # Create a Slice (v1) node using attributes
                new_slice = helper.make_node(
                    'Slice',
                    inputs=[data_input],
                    outputs=node.output,
                    starts=starts,
                    ends=ends,
                    axes=axes
                )
                new_nodes.append(new_slice)
                used_const_inputs.update([starts_input, ends_input])
                if axes_input:
                    used_const_inputs.add(axes_input)
                if steps:
                    used_const_inputs.add(steps)
            else:
                raise ValueError(f"Cannot convert Slice node {node.name} — slicing inputs not constant.")
        else:
            new_nodes.append(node)

    # Remove used Constant nodes
    new_nodes = [n for n in new_nodes if not (n.op_type == 'Constant' and n.output[0] in used_const_inputs)]
    # new_nodes = [n for n in new_nodes if not (n.op_type == 'Constant')]

    # Replace graph nodes
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # Replace opset with v1
    model.ClearField("opset_import")
    model.opset_import.extend([onnx.helper.make_opsetid("", 13)])
    print("opset", model.opset_import)
        
    return model
   
def post_process_onnx(onnx_file: str):
    model = onnx.load(onnx_file)
    
    # Convert Slice nodes from v10 to v1
    model = convert_slice_to_opset1(model)
                
                
    onnx.save(model, onnx_file)           
    

    
def main():
    export_model()
    # post_process_onnx("ppf.onnx")
    # test_dft()
    



if __name__ == "__main__":
    main()