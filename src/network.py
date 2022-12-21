import torch
import torch.nn as nn
from functools import partial
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d, ResNet18_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b4, efficientnet_b2, EfficientNet_B4_Weights, EfficientNet_B2_Weights, EfficientNet_B0_Weights
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights


def eff_selection(model_type):
    if model_type == 'b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_type == "b2":
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif model_type == "b4":
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    elif model_type == 'v2_s':
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    elif model_type == 'v2_m':
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    elif model_type == 'v2_l':
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Network type not implemented yet!")
    return model

class Windowing(nn.Module):
    pass




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
    def __init__(self, module, act=None, stacked_axis=-1):
        """

        Args:
            module (_type_): _description_
            stacked_axis (int, optional): Axis where images from same patient are stacked.
        """
        super().__init__()
        self.module = module
        self.stacked_axis = stacked_axis

        criterion = nn.CosineEmbeddingLoss()
        self.criterion = partial(criterion, **{'target': torch.tensor(1)})
        self.act_fun = lambda x: x
        if act == "sigmoid": 
            self.criterion = nn.BCEWithLogitsLoss()
        elif act == "softmax":
            self.act_fun = nn.Softmax(dim=0)
            self.criterion = nn.CrossEntropyLoss()
        elif act == 'relu':
            self.act_fun = nn.ReLU()
        elif act == 'mish':
            self.act_fun = nn.Mish()
        else:
            raise NotImplementedError(f"{act} is not implemented yet!")

    def forward(self, x: torch.Tensor):
        # Invert batch axis with stacked axis
        x = torch.stack([self.module(x_i) for x_i in x.transpose(self.stacked_axis, 0)], self.stacked_axis)
        x_0 = self.act_fun(torch.flatten(x[:, 0, ...]))
        x_1 = self.act_fun(torch.flatten(x[:, 1, ...]))
        x_2 = self.act_fun(torch.flatten(x[:, 2, ...]))

        # Calculated loss
        loss_01 = self.criterion(x_0, x_1)
        loss_12 = self.criterion(x_1, x_2)
        loss_02 = self.criterion(x_0, x_2)
        loss = (loss_01 + loss_12 + loss_02) / 3
        # loss = loss_01

        x = torch.mean(x, self.stacked_axis)
        # x = torch.logsumexp(x, self.axis) / x.size(self.axis)
        return x, loss

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
        return self.network(x), torch.zeros(1).to(x.device)


class ChiaEfficientNet(nn.Module):
    def __init__(self, backbone='b4', n_classes=1, hidden_dim=1024, freeze_weights=False, dropout=0.):
        super().__init__()
        model = eff_selection(backbone)
        # Define encoder
        enc = nn.Sequential(*list(model.children())[:-2])
        # Find output dimensions
        num_features = enc[-1][-1][0].out_channels
        if freeze_weights:
            for param in enc.parameters():
                param.requires_grad_(False)
        
        self.network = nn.Sequential(
            ChiaBlock(nn.Sequential(enc, nn.AdaptiveMaxPool2d(1), nn.Mish()), 1),
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

    def forward(self, x:torch.Tensor):
        return self.network(x), None

class SimChiaEfficientNet(nn.Module):
    def __init__(self, backbone='b4', n_classes=1, hidden_dim=1024, freeze_weights=False, dropout=0., act=None):
        super().__init__()
        model = eff_selection(backbone)
        # Define encoder
        self.enc = nn.Sequential(*list(model.children())[:-2])
        # Find output dimensions
        num_features = self.enc[-1][-1][0].out_channels
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        
        self.backbone = SimChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(1), nn.Mish()), act, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(nc),
            nn.Linear(num_features, hidden_dim),
            nn.Dropout(dropout),
            nn.Mish(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        features, loss = self.backbone(x)
        return self.head(features), loss


if "__main__" in __name__:
    input = torch.zeros((2, 4, 3, 512, 512))
    model = SimChiaEfficientNet()
    model.__repr__
    logits, simloss = model(input)
