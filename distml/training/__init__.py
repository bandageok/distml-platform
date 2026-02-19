"""
Training Module - DataParallel, ModelParallel, PipelineParallel
"""

import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    gradient_clip: float = 1.0
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"


class BaseStrategy:
    """Base training strategy"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
    
    def setup(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


class DataParallel(BaseStrategy):
    """Data Parallel Training Strategy"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.ddp_model = None
    
    def setup(self, model, optimizer):
        super().setup(model, optimizer)
        
        device = torch.device(f"cuda:{self.config.rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if self.config.world_size > 1:
            self.ddp_model = DDP(model, device_ids=[self.config.rank] if torch.cuda.is_available() else None)
        else:
            self.ddp_model = model
        
        logger.info(f"DataParallel setup: rank={self.config.rank}, world_size={self.config.world_size}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.ddp_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data = data.to(self.config.rank)
            target = target.to(self.config.rank)
            
            self.optimizer.zero_grad()
            output = self.ddp_model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Gather metrics across workers
        if self.config.world_size > 1:
            loss_tensor = torch.tensor(total_loss / num_batches).to(self.config.rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            total_loss = loss_tensor.item()
        
        return {"loss": total_loss, "batches": num_batches}
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        self.ddp_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.config.rank)
                target = target.to(self.config.rank)
                
                output = self.ddp_model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += data.size(0)
        
        if self.config.world_size > 1:
            loss_tensor = torch.tensor(total_loss).to(self.config.rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            
            correct_tensor = torch.tensor(correct).to(self.config.rank)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            correct = correct_tensor.item()
        
        return {
            "loss": total_loss / total if total > 0 else 0.0,
            "accuracy": correct / total if total > 0 else 0.0
        }


class ModelParallel(BaseStrategy):
    """Model Parallel Training Strategy"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_parts: List[nn.Module] = []
        self.devices: List[torch.device] = []
    
    def partition_model(self, model: nn.Module, num_partitions: int):
        """Partition model across devices"""
        layers = list(model.modules())
        partition_size = len(layers) // num_partitions
        
        for i in range(num_partitions):
            start = i * partition_size
            end = start + partition_size if i < num_partitions - 1 else len(layers)
            
            partition = nn.Sequential(*layers[start:end])
            device = torch.device(f"cuda:{i}" if torch.cuda.is_available() and i < torch.cuda.device_count() else "cpu")
            partition = partition.to(device)
            
            self.model_parts.append(partition)
            self.devices.append(device)
        
        logger.info(f"Model partitioned into {num_partitions} parts")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        for part in self.model_parts:
            part.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data = data.to(self.devices[0])
            
            # Forward through partitions
            for i, partition in enumerate(self.model_parts):
                data = partition(data)
            
            # Compute loss
            target = target.to(self.devices[-1])
            loss = nn.functional.cross_entropy(data, target)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}


class PipelineParallel(BaseStrategy):
    """Pipeline Parallel Training Strategy"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.pipeline_size = 1
        self.micro_batch_size = 4
        self.stage_id = 0
        self.num_stages = 1
    
    def set_stage_info(self, stage_id: int, num_stages: int):
        self.stage_id = stage_id
        self.num_stages = num_stages
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        num_microbatches = 0
        
        data_iter = iter(train_loader)
        
        for batch_idx in range(len(train_loader)):
            try:
                data, target = next(data_iter)
            except StopIteration:
                break
            
            # Split into microbatches
            microbatches = self._create_microbatches(data, target)
            
            for mb_data, mb_target in microbatches:
                output = self.model(mb_data)
                loss = nn.functional.cross_entropy(output, mb_target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_microbatches += 1
        
        return {"loss": total_loss / num_microbatches if num_microbatches > 0 else 0.0}
    
    def _create_microbatches(self, data: torch.Tensor, target: torch.Tensor):
        microbatches = []
        for i in range(0, data.size(0), self.micro_batch_size):
            mb_data = data[i:i + self.micro_batch_size]
            mb_target = target[i:i + self.micro_batch_size]
            microbatches.append((mb_data, mb_target))
        return microbatches


class HybridParallel(DataParallel, ModelParallel, PipelineParallel):
    """Hybrid Parallel - Combines all strategies"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.data_parallel_size = 1
        self.model_parallel_size = 1
        self.pipeline_parallel_size = 1


__all__ = [
    "TrainingConfig",
    "BaseStrategy",
    "DataParallel", 
    "ModelParallel",
    "PipelineParallel",
    "HybridParallel"
]
