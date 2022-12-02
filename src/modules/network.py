"""
Inspiration was taken from https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
"""

from collections import OrderedDict
import torch.nn as nn


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class BaseProjection(nn.Module):
    def __init__(
            self, 
            in_features,
            hidden_features,
            out_features,
            dropout
        ):
        super().__init__()
        projecton_layers = [
            # From original to hidden features
            ("fc1", nn.Linear(in_features, hidden_features, bias=False)),
            ("bn1", nn.BatchNorm1d(hidden_features)),
            ("drop1", nn.Dropout(dropout)),
            # Non-linearity
            ("relu1", nn.ReLU()),
            # From hidden to output features
            ("fc2", nn.Linear(hidden_features, out_features, bias=False)),
            ("bn2", BatchNorm1dNoBias(out_features))
        ]
        self.projection = nn.Sequential(OrderedDict(projecton_layers))


class SingleProjection(BaseProjection):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__(in_features, hidden_features, out_features, dropout)
        
    
    def forward(self, item):
        h_0 = item
        return self.projection(h_0)


class DualProjection(BaseProjection):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__(in_features, hidden_features, out_features, dropout)
        
    
    def forward(self, item):
        h_0, h_1 = item
        return self.projection(h_0), self.projection(h_1)

class OriginalFeatureSpace(BaseProjection):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__(in_features, hidden_features, out_features, dropout)

        decompress_layers = [
            # Non-linearity
            ("relu2", nn.ReLU()),
            # From original to hidden features
            ("fc3", nn.Linear(out_features, hidden_features, bias=False)),
            ("bn3", nn.BatchNorm1d(hidden_features)),
            ("drop2", nn.Dropout(dropout)),
            # Non-linearity
            ("relu3", nn.ReLU()),
            # From hidden to output features
            ("fc4", nn.Linear(hidden_features, in_features, bias=False)),
            # ("bn4", BatchNorm1dNoBias(in_features))
        ]
        self.decompression = nn.Sequential(OrderedDict(decompress_layers))
    
    def forward(self, item):
        h_0 = item[0]
        return h_0, self.decompression(self.projection(h_0))