import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.functional_tensor import normalize
from dataset import RSNA_BCD_Dataset, TestBatchSampler
import network
import torch
import torch.nn as nn
import albumentations as A
from tqdm import tqdm
from metrics import best_pfbeta, precision_recall, pfbeta
import numpy as np
from torchmetrics.classification import BinaryAUROC
from torch.utils.tensorboard import SummaryWriter


def main(cfg_path: Path, weights_path: Path, summary: SummaryWriter, device: str):
    metric = BinaryAUROC()
    print(f"Evaluating weight: {str(weights_path.name).split('.')[0].split('_')[1]}")
    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Define transforms
    transforms = [A.Resize(args['img_size'], args['img_size'], p=1)]

    dataset = RSNA_BCD_Dataset(
        dataset_path=Path(args["dataset_path"]), 
        patient_ids_path=Path(args["val_dataset"]), 
        keep_num=None,
        transform=A.Compose(transforms)
    )

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=TestBatchSampler(dataset, 1),
        collate_fn=dataset.collate_fn,
        num_workers=16,
        pin_memory=True
    )

    network_name = list(args['network'][0].keys())[0]
    network_params = list(args['network'][0].values())[0]
    model: nn.Module = network.__dict__[network_name](**network_params)
    
    model.to(device)
    # Load weights to model
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)

    mean = (torch.tensor([0.485, 0.456, 0.406])*(2**args['color_space'] - 1)).to(device)
    std = (torch.tensor([0.229, 0.224, 0.225])*(2**args['color_space'] - 1)).to(device)

    # For validation
    predictions = []
    targets = []
    model.eval()

    i = 0
    for batch in tqdm(loader):
        # Put model in eval mode
        # Fetch data
        test_images = batch[0].to(device)
        # Normalize images
        test_images = normalize(
            test_images, 
            mean=mean,
            std=std
        )
        test_classes = batch[1].to(device)

        with torch.no_grad():
            pred_logits = model(test_images)
            if isinstance(pred_logits, tuple):
                pred_logits, _ = pred_logits

        # Remember predictions and targets
        targets.append(test_classes.cpu().squeeze().item())
        predictions.append(torch.sigmoid(pred_logits).cpu().squeeze().item())

    targets = torch.tensor(targets)
    predictions = torch.tensor(predictions)

    auc = metric(predictions, targets)

    # Add metric to writer
    num_images = int(str(weight_path.name).split(".")[0].split("_")[1])
    summary.add_scalar("Metrics/AUC", auc, num_images)
    summary.flush()

    print(f"AUC: {auc:.4f}")

if "__main__" in __name__:
    test_paths = [
        "/data/rsna-breast-cancer-detection/exp/effv2s_heavyshift_brightness_clahe_light_newlr",
        "/data/rsna-breast-cancer-detection/exp/effv2s_heavyshift_brightness_clahe_light_smooth_newlr",
        "/data/rsna-breast-cancer-detection/exp/effv2s_heavyshift_newlr",
        "/data/rsna-breast-cancer-detection/exp/effv2s_heavyshift_smooth_newlr"
    ]
    for test_path in test_paths:
        test_path = Path(test_path)
        summary = SummaryWriter(test_path / Path("log_dir"))

        weights = list(test_path.glob("*.tar"))
        weights = sorted(weights, key=lambda x: int(str(x.name).split('.')[0].split('_')[1]))
        device = 'cuda'
        for weight_path in weights:
            main(test_path / Path("config.yaml"), weight_path, summary, device)
