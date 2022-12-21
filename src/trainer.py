import argparse
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import RSNA_BCD_Dataset, TrainBatchSampler, ValBatchSampler, TestBatchSampler
from model import RSNABCE
import albumentations as A
import torch

def main(cfg_path: str):
    torch.backends.cudnn.benchmark = True
    # Speedup performance!

    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Define transforms
    val_transforms = [A.Resize(args['img_size'], args['img_size'], p=1)]
    train_transforms = [
        A.__dict__[list(aug.keys())[0]](**list(aug.values())[0]) 
            for aug in args['augmentations']
        ]

    # Define dataset
    train_set = RSNA_BCD_Dataset(
        dataset_path=Path(args["dataset_path"]), 
        patient_ids_path=Path(args["train_dataset"]), 
        keep_num=args['keep_num'],
        smooth=args['smooth'],
        transform=A.Compose(train_transforms)
    )
    
    val_set = RSNA_BCD_Dataset(
        dataset_path=Path(args["dataset_path"]), 
        patient_ids_path=Path(args["val_dataset"]), 
        keep_num=args['keep_num'],
        transform=A.Compose(val_transforms)
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=TrainBatchSampler(train_set, args['batch_size'], args['neg_percent']),
        collate_fn=train_set.collate_fn,
        num_workers=args['train_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_sampler=ValBatchSampler(val_set, args['batch_size']),
        collate_fn=val_set.collate_fn,
        num_workers=args['val_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=val_set,
        batch_sampler=TestBatchSampler(val_set, args['batch_size']),
        collate_fn=val_set.collate_fn,
        num_workers=args['test_workers'],
        pin_memory=True
    )

    # Create model folder
    exp_path = Path(args['exp_path']) / Path(str(Path(cfg_path).name).split(".")[0])
    args['exp_path'] = str(exp_path)
    exp_path.mkdir(exist_ok=True, parents=True)
    # Path(args["exp_path"]).mkdir(exist_ok=True, parents=True)

    # Save configuration
    with open(Path(args['exp_path']) / Path("config.yaml"), 'w') as writer:
        yaml.safe_dump(data=args, stream=writer, sort_keys=False)
    
    # Define and train model
    RSNABCE(args).train(train_loader, val_loader, test_loader)

if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path/to/config.yaml", default="/projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_baseline.yaml")
    args = parser.parse_args()
    main(args.path)
