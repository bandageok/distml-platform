"""
Fault Tolerance Module - Checkpoint, Recovery, Fault Detection
"""

import logging
import os
import time
import shutil
import hashlib
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 100
    max_checkpoints: int = 5
    save_optimizer: bool = True


@dataclass
class Checkpoint:
    checkpoint_id: str
    step: int
    epoch: int
    timestamp: datetime
    file_path: str
    file_size: int = 0


class CheckpointManager:
    """Checkpoint Manager - Save and load model checkpoints"""
    
    def __init__(self, config: CheckpointConfig = None):
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
        self._lock = threading.RLock()
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer = None,
             step: int = 0, epoch: int = 0, metrics: Dict = None) -> Checkpoint:
        with self._lock:
            checkpoint_id = f"ckpt_step{step}_epoch{epoch}_{int(time.time())}"
            file_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            
            checkpoint_data = {
                "model_state": model.state_dict(),
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if self.config.save_optimizer and optimizer:
                checkpoint_data["optimizer_state"] = optimizer.state_dict()
            
            if metrics:
                checkpoint_data["metrics"] = metrics
            
            torch.save(checkpoint_data, file_path)
            
            file_size = file_path.stat().st_size
            
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                step=step,
                epoch=epoch,
                timestamp=datetime.utcnow(),
                file_path=str(file_path),
                file_size=file_size
            )
            
            self.checkpoints.append(checkpoint)
            self._rotate_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_id}")
            return checkpoint
    
    def load(self, file_path: str = None, checkpoint_id: str = None) -> Dict:
        if file_path:
            path = Path(file_path)
        elif checkpoint_id:
            path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        else:
            if not self.checkpoints:
                raise FileNotFoundError("No checkpoints available")
            path = Path(self.checkpoints[-1].file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        return torch.load(path, map_location="cpu")
    
    def _rotate_checkpoints(self):
        if len(self.checkpoints) > self.config.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            try:
                os.remove(oldest.file_path)
                logger.info(f"Removed old checkpoint: {oldest.checkpoint_id}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint: {e}")


class FaultDetector:
    """Fault Detector - Monitor node health"""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._heartbeats: Dict[str, datetime] = {}
        self._failure_callbacks = []
    
    def register_node(self, node_id: str):
        self._heartbeats[node_id] = datetime.utcnow()
    
    def heartbeat(self, node_id: str) -> bool:
        self._heartbeats[node_id] = datetime.utcnow()
        return True
    
    def is_healthy(self, node_id: str) -> bool:
        if node_id not in self._heartbeats:
            return False
        elapsed = (datetime.utcnow() - self._heartbeats[node_id]).total_seconds()
        return elapsed < self.timeout
    
    def get_status(self, node_id: str) -> Dict:
        if node_id not in self._heartbeats:
            return {"status": "unknown"}
        
        elapsed = (datetime.utcnow() - self._heartbeats[node_id]).total_seconds()
        return {
            "status": "healthy" if elapsed < self.timeout else "failed",
            "last_heartbeat": self._heartbeats[node_id].isoformat(),
            "elapsed_seconds": elapsed
        }


class RecoveryManager:
    """Recovery Manager - Handle failures"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self._recovery_strategies = {}
    
    def register_strategy(self, name: str, strategy: Callable):
        self._recovery_strategies[name] = strategy
    
    def recover(self, job_id: str, model: nn.Module, optimizer: torch.optim.Optimizer = None,
                strategy: str = "latest") -> int:
        checkpoint_data = self.checkpoint_manager.load()
        
        if "model_state" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state"])
        
        if optimizer and "optimizer_state" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        
        step = checkpoint_data.get("step", 0)
        logger.info(f"Recovered job {job_id} from step {step}")
        
        return step


__all__ = ["CheckpointConfig", "Checkpoint", "CheckpointManager", "FaultDetector", "RecoveryManager"]
