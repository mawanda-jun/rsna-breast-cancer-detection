import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d, ResNet18_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b4, efficientnet_b2, EfficientNet_B4_Weights, EfficientNet_B2_Weights, EfficientNet_B0_Weights
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
from itertools import combinations


def eff_selection(model_type):
    if model_type == 'b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_type == "b2":
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif model_type == "b4":
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    elif model_type == 'v2_s':
        model = efficientnet_v2_s(weights=None)
        model.load_state_dict(torch.load("/data/rsna-breast-cancer-detection/.cache/efficientnet_v2_s-dd5fe13b.pth"))
    elif model_type == 'v2_m':
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    elif model_type == 'v2_l':
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Network type not implemented yet!")

    enc = list(model.children())[:-2][0]
    num_features = enc[-1][0].out_channels
    return enc, num_features


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
        B, V, C, H, W = x.shape
        x = x.reshape((B*V, C, H, W))
        feats = self.module(x)
        feats = feats.reshape((B, V, -1))
        # x = torch.stack([self.module(x_i) for x_i in x.transpose(self.stacked_axis, 0)], self.stacked_axis)
        feats = torch.mean(feats, self.stacked_axis)
        # x = torch.logsumexp(x, self.axis) / x.size(self.axis)
        return feats

class SimChiaBlock(torch.nn.Module):
    def __init__(self, module, hidden_dim, proj_dim = 1280, feat_dim=1280):
        """

        Args:
            module (_type_): _description_
            stacked_axis (int, optional): Axis where images from same patient are stacked.
        """
        super().__init__()
        self.module = module
        self.projection = nn.Linear(feat_dim, proj_dim)
        self.fc = nn.Linear(feat_dim, hidden_dim)
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, x: torch.Tensor):
        B, V, C, H, W = x.shape
        x = x.reshape((B*V, C, H, W))
        feats = self.module(x)
        feats = feats.reshape((B, V, -1))
        
        # Calculate output
        outputs = torch.stack([self.fc(feat) for feat in feats.transpose(1, 0)], 1)
        outputs = F.mish(torch.mean(outputs, 1))

        # Calculate losses
        projections = [self.projection(x) for x in feats.transpose(1, 0)]

        losses = torch.stack([self.criterion(torch.flatten(x1), torch.flatten(x2), torch.tensor(1).to(x1.device)) for x1, x2 in combinations(projections, 2)], 0)
        loss = torch.mean(losses)

        return outputs, loss

class NewChiaBlock(torch.nn.Module):
    def __init__(self, module, hidden_dim, proj_dim = 1280, feat_dim=1280):
        """

        Args:
            module (_type_): _description_
            stacked_axis (int, optional): Axis where images from same patient are stacked.
        """
        super().__init__()
        self.module = module
        self.fc = nn.Linear(feat_dim, hidden_dim)
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, x: torch.Tensor):
        B, V, C, H, W = x.shape
        x = x.reshape((B*V, C, H, W))
        feats = self.module(x)
        feats = feats.reshape((B, V, -1))
        
        # Calculate output
        outputs = torch.stack([self.fc(feat) for feat in feats.transpose(1, 0)], 1)
        outputs = F.mish(torch.mean(outputs, 1))

        return outputs

class ChiaResNet(nn.Module):
    def __init__(self, backbone='resnext101_32x8d', n_classes=1, hidden_dim=1024, freeze_weights=False, dropout=0.):
        super().__init__()

        if backbone == "resnet50":
            self.enc = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnext101_32x8d":
            self.enc = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        elif backbone == "resnet18":
            self.enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Network type not implemented yet!")

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
        enc, num_features = eff_selection(backbone)
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
    def __init__(self, backbone='b4', n_classes=1, hidden_dim=1024, proj_dim=None, dropout=0.):
        super().__init__()
        model, num_features = eff_selection(backbone)
        if proj_dim is None:
            proj_dim = num_features
        
        self.encoder = nn.Sequential(model, nn.AdaptiveAvgPool2d(1), nn.Mish())

        self.projection = nn.Linear(num_features, proj_dim)
        self.hidden = nn.Linear(num_features, hidden_dim)
        self.cancer = nn.Linear(hidden_dim, n_classes)
        
        self.criterion = nn.CosineEmbeddingLoss()
        self.dropout_rate = dropout
        
    def forward(self, x: torch.Tensor):
        B, V, C, H, W = x.shape
        x = x.reshape((B*V, C, H, W))
        feats: torch.Tensor = self.encoder(x)
        feats = feats.reshape((B, V, -1))
        
        # Calculate output
        outputs = torch.stack([self.hidden(feat) for feat in feats.transpose(1, 0)], 1)
        outputs = torch.mean(outputs, 1)
        outputs = F.dropout(outputs, self.dropout_rate)
        outputs = F.mish(outputs)
        outputs = self.cancer(outputs)

        # Calculate losses
        projections = [self.projection(feat) for feat in feats.transpose(1, 0)]

        losses = torch.stack([self.criterion(torch.flatten(x1), torch.flatten(x2), torch.tensor(1).to(x1.device)) for x1, x2 in combinations(projections, 2)], 0)
        loss = torch.mean(losses)

        return outputs, loss * 0.1

class NewChiaEfficientNet(nn.Module):
    def __init__(self, backbone='b4', n_classes=1, hidden_dim=1024, freeze_weights=False, dropout=0., act=None):
        super().__init__()
        model, num_features = eff_selection(backbone)
        
        encoder = nn.Sequential(model, nn.AdaptiveAvgPool2d(1), nn.Mish())
        self.dropout = dropout

        self.backbone = NewChiaBlock(encoder, hidden_dim=hidden_dim, feat_dim=num_features)
        
        self.cancer = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.cancer(F.dropout(features, self.dropout)), None


if "__main__" in __name__:
    input = torch.zeros((2, 4, 3, 512, 512))
    model = ChiaEfficientNet('v2_s')
    print(model.__repr__)
    logits, simloss = model(input)
