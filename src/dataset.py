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

random.seed(42)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class RSNA_BCD_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            patient_ids_path: str,
            keep_num: int,
            transform = None,
            smooth = False
        ):
        super().__init__()
        self.dataset_path = dataset_path
        self.smooth = smooth
        print(f"Loading ids from {patient_ids_path.name}...")
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
        # # Keep always <keep_num> images. If there are less, repeat present images
        while len(img_paths) < self.keep_num:
            img_paths += img_paths
        
        if len(img_paths) > self.keep_num:
            img_paths = img_paths[:self.keep_num]

        imgs = [np.asarray(Image.open(img_path).convert('RGB')) for img_path in img_paths]
        # # Keep always <keep_num> images. If there are less, add an empty image
        # while len(imgs) < self.keep_num:
        #     imgs.append(np.zeros_like(imgs[0]))
        # if len(imgs) > self.keep_num:
        #     imgs = imgs[:self.keep_num]

        # Apply transform
        if self.transform is not None:
            imgs = np.array([self.transform(image=img)['image'] for img in imgs])
        
        # Every item in patient_id/laterality has the same category!
        label = categories[0]
        
        if self.smooth:
            if label == 1:
                label = random.uniform(0.75, 1)
            else:
                label = random.uniform(0, 0.25)

        return imgs, label  

    @staticmethod
    def collate_fn(batch):
        """
        This function is responsible for managing shapes. We want to have:
        (B, G, C, H, W) aka batch size, group of mammographies, channels, height, width.
        """
        images = []
        categories = []
        for block in batch:
            # Here image comes as (G, H, W, C).
            img = torch.tensor(np.array(block[0])).permute(0, 3, 1, 2)
            label = torch.tensor(np.array(block[1]))
            images.append(img)
            categories.append(label)
        # Prepare batch on first dimension.
        images = torch.stack(images, 0).to(torch.float32)
        categories = torch.stack(categories, 0).to(torch.float32).unsqueeze(-1)
        return images, categories

class TrainBatchSampler(Sampler):
    """
    Create custom batch sampler in order to create a batch of indexes with balanced positives/negatives.
    """
    def __init__(self, dataset: RSNA_BCD_Dataset, batch_size: int, neg_percent:float):
        self.dataset = dataset
        self.batch_size = batch_size
        self.neg_percent = neg_percent
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
                if random.random() > self.neg_percent:
                    # Select positive index
                    batch.append(random.choice(self.positives))
                else:
                    batch.append(random.choice(self.negatives))
            yield batch

    def __iter__(self):
        return iter(self.generate_batch())
    
    def __len__(self):
        return len(self.dataset) * 100  # Make an infinite iterator.

class ValBatchSampler(Sampler):
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
        val_set = [p for p in self.positives]  # COPY LIST!
        # Take some samples from negative distribution, se we keep equal number of negative and positives.
        val_set += random.sample(self.negatives, len(self.positives))
        random.shuffle(val_set)

        for chunk in chunks(val_set, self.batch_size):
            yield chunk

    def __iter__(self):
        return iter(self.generate_batch())
    
    def __len__(self):
        loader_len = (len(self.positives) * 2) // self.batch_size
        if loader_len * self.batch_size < len(self.positives)*2: # division is not exact
            loader_len += 1
        return loader_len

class TestBatchSampler(BatchSampler):
    """
    Create custom batch sampler that returns patiend_ids strings instead of indexes.
    """
    def __init__(self, dataset: RSNA_BCD_Dataset, batch_size: int):
        super().__init__(sampler=list(dataset.patient_ids.keys()), batch_size=batch_size, drop_last=False)


if "__main__" in __name__:
    transform = Compose([Resize(512, 512, p=1)])
    dataset = RSNA_BCD_Dataset(
        dataset_path = Path("/data/rsna-breast-cancer-detection/train_images_png"),
        patient_ids_path = Path("/projects/rsna-breast-cancer-detection/src/configs/train_ids.yaml"),
        keep_num=2,
        transform=transform
    )
    batch_size = 40
    dataloader = DataLoader(
        batch_sampler=TrainBatchSampler(dataset, batch_size),
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        num_workers=6,
        pin_memory=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)

        

        
