import torch

class ChiaBlock(torch.nn.Module):
    def __init__(self, module, axis=-1):
        super().__init__()
        self.module = module
        self.axis = axis

    def forward(self, x):
        x = torch.stack([self.module(x_i) for x_i in x.transpose(1, 0)], self.axis)
        x = torch.mean(x, self.axis)
        # x = torch.logsumexp(x, self.axis) / x.size(self.axis)
        return x