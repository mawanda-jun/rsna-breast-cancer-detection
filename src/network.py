import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d, ResNet18_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
from torchvision.models.efficientnet import efficientnet_b4, efficientnet_b2, EfficientNet_B4_Weights, EfficientNet_B2_Weights

class ChiaBlock(torch.nn.Module):
    def __init__(self, module, stacked_axis=-1):
        """

        Args:
            module (_type_): _description_
            stacked_axis (int, optional): Axis where images from same patient are stacked.
        """
        super().__init__()
        self.module = module
        self.stacked_axis = stacked_axis

    def forward(self, x: torch.Tensor):
        # Invert batch axis with stacked axis
        x = torch.stack([self.module(x_i) for x_i in x.transpose(self.stacked_axis, 0)], self.stacked_axis)
        x = torch.mean(x, self.stacked_axis)
        # x = torch.logsumexp(x, self.axis) / x.size(self.axis)
        return x

class SimChiaBlock(torch.nn.Module):
    def __init__(self, module, stacked_axis=-1):
        """

        Args:
            module (_type_): _description_
            stacked_axis (int, optional): Axis where images from same patient are stacked.
        """
        super().__init__()
        self.module = module
        self.stacked_axis = stacked_axis
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, x: torch.Tensor):
        # Invert batch axis with stacked axis
        x = torch.stack([self.module(x_i) for x_i in x.transpose(self.stacked_axis, 0)], self.stacked_axis)
        x = torch.mean(x, self.stacked_axis)
        # x = torch.logsumexp(x, self.axis) / x.size(self.axis)
        return x

class ChiaResNet(nn.Module):
    def __init__(self, backbone='resnext101_32x8d', n_classes=1, hidden_dim=1024, freeze_weights=False, dropout=0.):
        super().__init__()

        if backbone == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnext101_32x8d":
            model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        elif backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Network type not implemented yet!")

        # Define encoder
        self.enc = nn.Sequential(*list(model.children())[:-2])
        # Find output dimensions
        bottleneck = list(self.enc.children())[-1][-1]
        try:
            num_features = bottleneck.conv3.out_channels
        except AttributeError:
            num_features = bottleneck.conv2.out_channels

        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        
        self.network = nn.Sequential(
            ChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(1), nn.Mish()), 1),
            # AdaptiveConcatPool2d(),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            # Mish(),
            nn.Flatten(),
            # nn.BatchNorm1d(nc),
            nn.Linear(num_features, hidden_dim),
            nn.Dropout(dropout),
            nn.Mish(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.network(x)


class ChiaEfficientNet(nn.Module):
    def __init__(self, backbone='b4', n_classes=1, hidden_dim=1024, freeze_weights=False, dropout=0.):
        super().__init__()

        if backbone == "b2":
            model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif backbone == "b4":
            model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Network type not implemented yet!")

        # Define encoder
        self.enc = nn.Sequential(*list(model.children())[:-2])
        # Find output dimensions
        num_features = self.enc[-1][-1][0].out_channels
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        
        self.network = nn.Sequential(
            ChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(1), nn.Mish()), 1),
            # AdaptiveConcatPool2d(),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            # Mish(),
            nn.Flatten(),
            # nn.BatchNorm1d(nc),
            nn.Linear(num_features, hidden_dim),
            nn.Dropout(dropout),
            nn.Mish(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.network(x)

if "__main__" in __name__:
    model = ChiaEfficientNet()