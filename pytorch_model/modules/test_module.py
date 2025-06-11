import torch
import torch.nn as nn

from modules.pfb_module import PFBModule


class TestModule(PFBModule):
    def __init__(self, batch_size):
        """
        This test module should work with batch size greater than 1, but doesnt work on NVDLA.
        """

        super().__init__(1, 1, batch_size)

    def forward(self, x):
        # x = torch.split(x, 1, dim=1)
        # print(len(x))
        input_channel_slices = list(torch.split(x, 1, dim=1))

        x = torch.cat(input_channel_slices, dim=1)
        return x
