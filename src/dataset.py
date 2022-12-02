import numpy as np
from typing import List
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import numpy as np

from utils import parse_pb


class GUIE_BaseDataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            synset_ids: List[str],
            multiplier: int,
            mock_length: int
        ):
        """_summary_
        Use mock_length so dataloader is resetted only after one epoch - and not after
        finishing the far-less classes.
        Args:
            dataset_path (Path): _description_
            synset_ids (List[str]): _description_
            multiplier (int): _description_
            mock_length (int): _description_
        """
        super().__init__()
        # Extract information about dataset
        all_files = dataset_path.glob("*.pb")
        self.multiplier = multiplier
        self.mock_length = mock_length
        self.synset_ids = synset_ids

        # Index file by category: keep path/to/cat/chunks together, indexed by synset cat id provided.
        synset_paths = {k: [] for k in synset_ids}
        for file in tqdm(all_files, desc='Creating index...', total=6581546):
            cat = str(file.name).split("_")[0]
            if cat in synset_paths:
                synset_paths[cat].append(str(file))

        # Create mapping
        # self.idx_mapper = [synset_id for synset_id in synset_paths.keys()]
        # Avoid memory leak using numpy: https://github.com/pytorch/pytorch/issues/13246#issuecomment-1164905242
        # All array must have the same length in order to convert them to a numpy matrix
        max_paths = max([len(v) for v in synset_paths.values()])
        for k, v in tqdm(synset_paths.items(), desc="Fixing number of paths for the DataFrame..."):
            synset_paths[k] += [""]*(max_paths - len(v))
        # Keep only list of lists of paths
        synset_paths = list(synset_paths.values())
        self.synset_paths = np.array(synset_paths).astype(np.string_)
    
    def __len__(self):
        return self.mock_length

    def _get_features(self, idx, num=None):
        # Rationale of getitem:
        # - idx select which category we are dealing with
        # - from the category, select two random paths: we will extract the features from these two files
        #   -> the two paths might be the same (random.choices instead of random.sample). This is 
        #      intentional since we want to possibly return the same features, or the features inside the
        #      same file
        # - parse the two paths and extract the features. For each file, in this configuration, there will be
        #   2 features.
        # - select one of the two features for each file.
        # Now we have two features of the same category, which we are going to return.

        synset_paths = [l.decode() for l in self.synset_paths[idx] if l.decode() != ""]
        if num == None:
            selected_synset_paths = synset_paths
        else:
            selected_synset_paths = random.choices(population=synset_paths, k=num)
        
        features = [random.sample(parse_pb(path)[0], 1)[0] for path in selected_synset_paths]
        features = [f.astype(np.float32) / self.multiplier for f in features]
        return features
    

class GUIE_DualFeatures(GUIE_BaseDataset):
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)

    @staticmethod
    def collate_fn(batch):
        batch = np.asarray(batch)
        batch_0 = batch[:, 0, ...]
        batch_1 = batch[:, 1, ...]
        return torch.tensor(batch_0), torch.tensor(batch_1)

class GUIE_SingleFeatures(GUIE_BaseDataset):
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)

    @staticmethod
    def collate_fn(batch):
        batch = np.asarray(batch)
        return [torch.tensor(batch)]

class GUIE_AllFeatures(GUIE_BaseDataset):
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)

    @staticmethod
    def collate_fn(batch):
        batch = np.asarray(batch)
        batch = torch.tensor(batch)
        batch = torch.cat(batch, 1)
        return torch.tensor(batch)


#############################
# CUSTOM AND USEFUL DATASETS
#############################
class SameClassDataset(GUIE_DualFeatures):
    """
    Returns two features that comes from the same class. Might be the same feature altogehter
    (even though it's a quite rare occurrence).
    """
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)
    
    def __getitem__(self, idx):
        features = self._get_features(idx, num=2)
        return features

class NoiseItemDataset(GUIE_DualFeatures):
    """
    Returns one feature and the same feature with some gaussian noise applied.
    """
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)

    def __getitem__(self, idx):
        features = self._get_features(idx, num=1)[0]
        # Add random noise
        noise = np.random.normal(0, 0.1, size=features.shape).astype(features.dtype)
        return features, features + noise

class SingleItemDataset(GUIE_SingleFeatures):
    """
    Returns only one feature, but it's in the same list format for the sake of the training.
    """
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)

    def __getitem__(self, idx):
        return self._get_features(idx, num=1)[0]


class EvalDataset(GUIE_AllFeatures):
    """
    Returns only one feature, but it's in the same list format for the sake of the training.
    """
    def __init__(self, dataset_path: Path, synset_ids: List[str], multiplier: int, mock_length: int):
        super().__init__(dataset_path, synset_ids, multiplier, mock_length)

    def __getitem__(self, idx):
        return self._get_features(idx)


class CustomBatchSampler(Sampler):
    """
    Create custom batch sampler in order to create a batch of indexes that are not repeated for each batch.

    Args:
        Sampler (_type_): _description_
    """
    def __init__(self, dataset: GUIE_BaseDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        # self.real_index = np.arange(0, len(self.dataset.synset_ids))
        assert len(dataset) >= batch_size, f"Not enough elements for a batch! Please input a batch size <= {len(self.real_index)}"

    def generate_batch(self):
        while True:
            yield np.random.choice(len(self.dataset.synset_ids), self.batch_size, replace=False)

    def __iter__(self):
        return iter(self.generate_batch())

    def __len__(self):
        return len(self.dataset)

if "__main__" in __name__:
    dataset = GUIE_BaseDataset(
        dataset_path = Path("/data/GoogleUniversalImageEmbedding/data/by_chunks"),
        synset_ids = open("/projects/GoogleUniversalImageEmbedding/dataset_info/train_synset_ids.txt").read().splitlines(),
        multiplier = 10000,
        mock_length = 100000
    )
    batch_size = 153
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=CustomBatchSampler(dataset, batch_size),
        collate_fn=dataset.collate_fn,
        num_workers=10,
        pin_memory=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)

        

        
