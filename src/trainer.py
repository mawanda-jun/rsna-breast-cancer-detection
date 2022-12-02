import argparse
import yaml
from pathlib import Path
import os
from torch.utils.data import DataLoader
import SimCLR_dataset
from model import SimCLRContrastiveLearning

def main(cfg_path: str):
    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Redefine args so it takes epochs in consideration
    args['steps'] = args['train_features'] // args['batch_size']
    args['steps'] *= args['epochs']
    args['save_steps'] *= args['epochs']

    # Define dataset
    dataset_path = Path(args["dataset_path"])
    train_ids = open(args["train_dataset"], 'r').read().splitlines()
    val_ids = open(args["val_dataset"], 'r').read().splitlines()

    train_set = SimCLR_dataset.__dict__[args['dataset_type']](
        dataset_path, 
        train_ids, 
        args['multiplier'],
        args['train_features']
        )
    val_set = SimCLR_dataset.__dict__[args['dataset_type']](
        dataset_path, 
        val_ids, 
        args['multiplier'],
        mock_length=args['val_features']
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=SimCLR_dataset.__dict__['CustomBatchSampler'](train_set, args['batch_size']),
        collate_fn=train_set.collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_sampler=SimCLR_dataset.__dict__['CustomBatchSampler'](val_set, args['batch_size']),
        collate_fn=val_set.collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    # Create model folder
    Path(args["exp_path"]).mkdir(exist_ok=True, parents=True)

    # Save configuration
    with open(Path(args['exp_path']) / Path("config.yaml"), 'w') as writer:
        yaml.safe_dump(data=args, stream=writer)
    
    # Define and train model
    SimCLRContrastiveLearning(args).train(train_loader, val_loader)

if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path/to/config.yaml", default="/projects/GoogleUniversalImageEmbedding/config/config.yaml")
    args = parser.parse_args()
    main(args.path)
