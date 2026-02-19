"""
Utilities Module
"""

import os
import json
import yaml
import logging
import random
import numpy as np
import torch


def set_cuda_devices(device_ids):
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def get_device(rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: dict, path: str):
    with open(path, 'w') as f:
        if path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif path.endswith('.yaml'):
            yaml.dump(config, f)


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith('.yaml'):
            return yaml.safe_load(f)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


__all__ = ["set_cuda_devices", "get_device", "count_parameters", "save_config", "load_config", "seed_everything", "AverageMeter", "EarlyStopping"]
