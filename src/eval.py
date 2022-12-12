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

def main(cfg_path: Path, weights_path: Path, device: str):
    print(f"Evaluating weight: {str(weights_path.name).split('.')[0].split('_')[1]}")
    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Define transforms
    transforms = [A.Resize(args['img_size'], args['img_size'], p=1)]

    dataset = RSNA_BCD_Dataset(
        dataset_path=Path(args["dataset_path"]), 
        patient_ids_path=Path(args["val_dataset"]), 
        keep_num=args['keep_num'],
        transform=A.Compose(transforms)
    )
    
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=TestBatchSampler(dataset, args['batch_size']),
        collate_fn=dataset.collate_fn,
        num_workers=args['test_workers'],
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
        targets += test_classes.squeeze().tolist()
        predictions += torch.sigmoid(pred_logits).cpu().squeeze().tolist()

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    beta, pF1 = best_pfbeta(targets, predictions)

    print(f"Beta {beta:.4f} is best for pF1: {pF1:.4f}")

    with open(cfg_path.parent / Path(f"ckp_{str(weights_path.name).split('.')[0].split('_')[1]}_{beta:.4f}_{pF1:.4f}.txt"), 'w') as writer:
        writer.write(f"{beta} {pF1}")

if "__main__" in __name__:
    test_path = Path("/data/rsna-breast-cancer-detection/exp/EffB4_allaug_sim_long_INTERR/config.yaml")
    weights = list(test_path.parent.glob("*.tar"))
    weights = sorted(weights, reverse=True, key=lambda x: int(str(x.name).split('.')[0].split('_')[1]))
    device = 'cuda'
    for weight_path in weights:
        main(test_path, weight_path, device)
