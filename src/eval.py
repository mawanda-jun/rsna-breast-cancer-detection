import argparse
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import RSNA_BCD_Dataset, TrainBatchSampler, ValBatchSampler, TestBatchSampler
from model import RSNABCE
import albumentations as A
import torch

def main(cfg_path: str, model_path, num_images, global_key):
    
    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Define transforms
    val_transforms = [A.Resize(args['img_size'], args['img_size'], p=1)]
    
    val_set = RSNA_BCD_Dataset(
        dataset_path=Path(args["dataset_path"]), 
        patient_ids_path=Path(args["val_dataset"]), 
        keep_num=args['keep_num'],
        transform=A.Compose(val_transforms)
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
   
    # Define and train model
    model = RSNABCE(args)
    model.load_model(model_path)
    model.eval(test_loader, num_images, global_key)

if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path/to/config.yaml", default="/projects/rsna-breast-cancer-detection/src/configs_fortheval/effv2s_sim_l2f_heavyshift_lightsmooth.yaml")
    args = parser.parse_args()
    main(args.path)
