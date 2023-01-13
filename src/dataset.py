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
            smooth = 0
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

    def __gray_to_rgb(self, img):
        if len(img.shape) == 2:
            img = np.stack([img, img, img], -1)
        elif len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.concatenate([img, img, img], -1)
        return img
    
    def __getitem__(self, patient_id_laterality):
        img_ids, categories = self.patient_ids[patient_id_laterality]
        patient_id = patient_id_laterality.split("_")[0]
        img_paths = [self.dataset_path / Path(str(patient_id)) / Path(f"{img_id}.png") for img_id in img_ids]
        random.shuffle(img_paths)  # So that, in case there are more then 3, we select always different images.
        # Keep always <keep_num> images. If there are less, repeat present images
        if self.keep_num is not None:
            while len(img_paths) < self.keep_num:
                img_paths += img_paths
            
            if len(img_paths) > self.keep_num:
                img_paths = img_paths[:self.keep_num]

        imgs = [self.__gray_to_rgb(np.array(Image.open(img_path))) for img_path in img_paths]
        # # Keep always <keep_num> images. If there are less, add an empty image
        # while len(imgs) < self.keep_num:
        #     imgs.append(np.zeros_like(imgs[0]))
        # if len(imgs) > self.keep_num:
        #     imgs = imgs[:self.keep_num]

        # Apply transform
        if self.transform is not None:
            imgs = np.stack([self.transform(image=img)['image'] for img in imgs], 0)
        
        # Every item in patient_id/laterality has the same category!
        label = categories[0]

        # Apply smooth labeling. If self.smooth is 0, then no smoothing is applied.
        # Otherwise, it is applied. self.smoot == 0.3 -> label in [0.85, 1], label.mean ~= 0.925
        new_label = label * (1.0 - self.smooth) + 0.5 * self.smooth
        label = random.uniform(new_label, label)

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
            # Convert block to int16 so there are no problems of uint8 or uint16
            img = np.array(block[0]).astype(np.int16)
            # Here image comes as (G, H, W, C).
            img = torch.tensor(img).permute(0, 3, 1, 2)
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
        positives = np.arange(len(self.positives))
        pos_avail = np.ones(len(self.positives), dtype=np.bool)
        negatives = np.arange(len(self.negatives))
        neg_avail = np.ones(len(self.negatives), dtype=np.bool)
        while True:
            batch = []
            for _ in range(self.batch_size):
                if random.random() > self.neg_percent:
                    if pos_avail.sum() == 0:  # Reset indexes since they are exausted!
                        pos_avail = np.ones(len(self.positives), dtype=np.bool)
                    # Select positive index
                    pos = np.random.choice(positives[pos_avail], replace=False)
                    pos_avail[pos] = False
                    batch.append(self.positives[pos])
                else:
                    if neg_avail.sum() == 0:  # Reset indexes since they are exausted!
                        neg_avail = np.ones(len(self.negatives), dtype=np.bool)
                    neg = np.random.choice(negatives[neg_avail], replace=False)
                    neg_avail[neg] = False
                    batch.append(self.negatives[pos])
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


def visualize_augs(augs_path):
    import albumentations as A

    with open(augs_path, 'r') as reader:
            augs = yaml.load(reader, Loader=CSafeLoader)
    
    transforms = A.Compose([
            A.__dict__[list(aug.keys())[0]](**list(aug.values())[0]) 
            for aug in augs['augmentations']
        ])
    dataset = RSNA_BCD_Dataset(
        dataset_path = Path("/data/rsna-breast-cancer-detection/train_images_png"),
        patient_ids_path = Path("/projects/rsna-breast-cancer-detection/src/dataset_info/val_ids.yaml"),
        keep_num=1,
        transform=transforms
    )
    imgs = []
    edge = 5
    for i in range(edge**2):
        patient_id = list(dataset.patient_ids.keys())[i]
        img, label = dataset[patient_id]
        imgs.append(img[0])
    cols = []
    for i in range(edge):
        cols.append(np.concatenate(imgs[edge*i:edge*(i+1)], 1))
    imgs = np.concatenate(cols, 0)

    # imgs = (imgs * 2**(8-16)).astype(np.uint8)
    Image.fromarray(imgs).save("deleteme.png")


if "__main__" in __name__:
    transform = Compose([Resize(1024, 1024, p=1)])
    dataset = RSNA_BCD_Dataset(
        dataset_path=Path("/data/rsna-breast-cancer-detection/train_images_png"), 
        patient_ids_path=Path("/projects/rsna-breast-cancer-detection/src/dataset_info/train_ids.yaml"), 
        keep_num=3,
        smooth=0.05,
        transform=transform
    )
    batch_size = 20
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=TrainBatchSampler(dataset, batch_size, 0.5),
        collate_fn=dataset.collate_fn,
        num_workers=1,
        pin_memory=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)
    # augs_path = "/projects/rsna-breast-cancer-detection/src/all_augs.yaml"
    # visualize_augs(augs_path)
        

        
