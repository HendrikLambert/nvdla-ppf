import torch 
import torch.nn as nn

import numpy as np


class PPFModel(nn.Module):
    def __init__(self, P=256, M=16):
        """
        
        P: Number of channels
        M: Number of taps
        
        """
        
        super().__init__()
        
        self.cnn = nn.Conv2d(P, P, kernel_size=(1, M), groups=P)
        # self.cnn.weight = nn.Parameter(torch.from_numpy(np.zeros(self.cnn.weight.shape).astype(np.float32) + 1))
        self.cnn.bias = nn.Parameter(torch.from_numpy(np.zeros(self.cnn.weight.shape[0]).astype(np.float32)))
        # print()
        print("weight", self.cnn.weight.shape)
        print("bias", self.cnn.bias.shape)

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




def main():
    ppf_model = PPFModel()
    example_inputs = torch.ones(1, 256, 1, 16)
    # example_inputs = torch.ones(4*16, 16, 16, 16)

    out = ppf_model(example_inputs)
    # print(out)
    print()

    # torch.onnx.export(ppf_model, example_inputs, "ppf.onnx",
    #               input_names=['input'],
    #               output_names=['output'],
    #             #   dynamic_axes={
    #             #     'input': {0: 'batch_size'},
    #             #     'output': {0: 'batch_size'}
    #             #   }
    #             )
    
    



if __name__ == "__main__":
    main()