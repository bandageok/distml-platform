"""
Storage Module - Data Loading and Checkpoint Storage
"""

import logging
from typing import Optional
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if self.transform:
            item = self.transform(item)
        return item


def create_dataloader(dataset, batch_size: int, shuffle: bool = False, 
                     num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )


import torch


__all__ = ["TransformDataset", "create_dataloader"]
