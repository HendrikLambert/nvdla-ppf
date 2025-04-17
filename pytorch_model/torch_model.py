import torch 
import torch.nn as nn

import numpy as np


class PPFModel(nn.Module):
    def __init__(self, P=256, M=16):
        """
        PPFModel is a PyTorch implementation of the PPF model.
        """
        
        super().__init__()
        
        self.cnn = nn.Conv2d(P, P, kernel_size=(1, M), groups=P)
        self.cnn.weight = nn.Parameter(torch.from_numpy(np.zeros(self.cnn.weight.shape).astype(np.float32) + 1))
        self.cnn.bias = nn.Parameter(torch.from_numpy(np.arange(self.cnn.weight.shape[0]).astype(np.float32)))
        # print()
        print(self.cnn.weight.shape)
        print(self.cnn.bias.shape)

        # print(self.cnn.weight.shape)
    
    def forward(self, x):
        x = self.cnn(x)
        return x






def main():
    ppf_model = PPFModel()
    example_inputs = torch.ones(2, 256, 1, 16)

    out = ppf_model(example_inputs)
    # print(out)

    torch.onnx.export(ppf_model, example_inputs, "ppf.onnx",
                  input_names=['input'],
                  output_names=['output'],
                #   dynamic_axes={
                #     'input': {0: 'batch_size'},
                #     'output': {0: 'batch_size'}
                #   }
                )
    
    print(out.shape)



if __name__ == "__main__":
    main()