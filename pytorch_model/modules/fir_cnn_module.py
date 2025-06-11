import torch
import torch.nn as nn

class FIRCNNModule(nn.Module):
    def __init__(self, P, M, fir_weights: torch.Tensor):
        """
        
        P: Number of channels
        M: Number of taps
        
        """
        super().__init__()

        self.cnn = nn.Conv2d(P*2, P*2, kernel_size=(1, M), groups=P*2, bias=False)
        # print(self.cnn.weight.shape)
        self.cnn.weight = nn.Parameter(fir_weights)

        # Disable learning of weights
        for param in self.cnn.parameters():
            param.requires_grad = False
        
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