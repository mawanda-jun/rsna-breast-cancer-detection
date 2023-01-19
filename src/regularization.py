import torch
import torch.nn as nn

class L2Regularization(nn.Module):
    def __init__(self, l2_lambda, layers) -> None:
        super().__init__()
        self.l2_lambda = l2_lambda
        self.layers = layers
    
    def forward(self, model:nn.Module):
        l2_reg = 0.
        for name, W in model.named_parameters():
            if "weight" in name:
                for layer in self.layers:
                    if layer in name:
                        l2_reg += W.pow(2.0).sum()         
        # l2_reg = torch.stack([
        #         W.pow(2.0).sum() 
        #     for name, W in model.named_parameters() 
        #         if "weight" in name
        #     ]).sum()

        # l2_reg = 0
        # models = [model.projection, model.fc, model.cancer]
        # for model in models:
        #     for name, W in model.named_parameters():
        #         if "weight" in name:
        #             l2_reg += W.pow(2.0).sum()
        return self.l2_lambda * l2_reg