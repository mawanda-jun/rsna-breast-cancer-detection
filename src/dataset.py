import numpy as np
from typing import List
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import yaml
from yaml import CSafeLoader
import numpy as np
from PIL import Image
from albumentations import Compose, Resize


class RSNA_BCD_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            patient_ids_path: str,
            keep_num: int,
            transform = None
        ):
        super().__init__()
        self.dataset_path = dataset_path
        print("Loading ids...")
        with open(patient_ids_path, 'r') as reader:
            self.patient_ids = yaml.load(reader, Loader=CSafeLoader)
        print("Ids loaded!")
        self.keep_num = keep_num
        self.transform = transform
    
    def __len__(self):
        return len(list(self.patient_ids.keys()))
    
    def __getitem__(self, patient_id_laterality):
        img_ids, categories = self.patient_ids[patient_id_laterality]
        patient_id = patient_id_laterality.split("_")[0]
        img_paths = [self.dataset_path / Path(str(patient_id)) / Path(f"{img_id}.png") for img_id in img_ids]
        random.shuffle(img_paths)  # So that, in case there are more then 3, we select always different images.
        
        imgs = [np.asarray(Image.open(img_path)) for img_path in img_paths]
        # Keep always <keep_num> images. If there are less, add an empty image
        while len(imgs) < self.keep_num:
            imgs.append(np.zeros_like(imgs[0]))
        if len(imgs) > self.keep_num:
            imgs = imgs[:self.keep_num]

        # Apply transform
        if self.transform is not None:
            imgs = [np.repeat(np.expand_dims(self.transform(image=img)['image'], -1), 3, -1) for img in imgs]
        
        return imgs, categories[0]  # Every item in patient_id/laterality has the same category!

    @staticmethod
    def collate_fn(batch):
        """
        This function is responsible for managing shapes. We want to have:
        (B, C, H, W, G) aka batch size, channels, height, width and group of mammographies.
        """
        images = []
        categories = []
        for block in batch:
            # Here image comes as (G, H, W, C), so we permute the first and fourth axis.
            img = torch.tensor(np.array(block[0])).permute(3, 1, 2, 0)
            label = torch.tensor(np.array(block[1]))
            images.append(img)
            categories.append(label)
        # Prepare batch on first dimension.
        images = torch.stack(images, 0)
        categories = torch.stack(categories, 0)
        return images, categories

class TrainBatchSampler(Sampler):
    """
    Create custom batch sampler in order to create a batch of indexes with balanced positives/negatives.
    """
    def __init__(self, dataset: RSNA_BCD_Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        # Inizialize positives and negatives.
        self.positives = []
        self.negatives = []
        for patient_id, arr in dataset.patient_ids.items():
            if arr[1][0] == 0:
                self.negatives.append(patient_id)
            else:
                self.positives.append(patient_id)

    def generate_batch(self):
        # Make a balanced loader!
        while True:
            batch = []
            for _ in range(self.batch_size):
                if random.random() > 0.5:
                    # Select positive index
                    batch.append(random.choice(self.positives))
                else:
                    batch.append(random.choice(self.negatives))
            yield batch

    def __iter__(self):
        return iter(self.generate_batch())
    
    def __len__(self):
        return len(self.dataset)

class ValBatchSampler(BatchSampler):
    """
    Create custom batch sampler in order to create a batch of indexes with balanced positives/negatives.
    """
    def __init__(self, dataset: RSNA_BCD_Dataset, batch_size: int):
        super().__init__(sampler=list(dataset.patient_ids.keys()), batch_size=batch_size, drop_last=False)


if "__main__" in __name__:
    transform = Compose([Resize(512, 512, p=1)])
    dataset = RSNA_BCD_Dataset(
        dataset_path = Path("/data/rsna-breast-cancer-detection/train_images_png"),
        patient_ids_path = Path("/projects/rsna-breast-cancer-detection/src/configs/train_ids.yaml"),
        keep_num=3,
        transform=transform
    )
    batch_size = 40
    dataloader = DataLoader(
        batch_sampler=TrainBatchSampler(dataset, batch_size),
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        num_workers=0,
        pin_memory=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)

        

        
