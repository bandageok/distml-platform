"""
Core Module for DistML Platform - Consolidated

Master, Worker, Parameter Server, and Communicator implementations.
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class NodeType(Enum):
    MASTER = "master"
    PARAMETER_SERVER = "parameter_server"
    WORKER = "worker"


class NodeStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    IDLE = "idle"
    FAILED = "failed"


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NodeInfo:
    node_id: str
    node_type: NodeType
    rank: int = 0
    host: str = ""
    port: int = 0
    status: NodeStatus = NodeStatus.INITIALIZING


@dataclass
class JobConfig:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "training_job"
    num_workers: int = 1
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class Job:
    job_id: str
    config: JobConfig
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClusterConfig:
    master_host: str = "localhost"
    master_port: int = 50051
    backend: str = "nccl"
    heartbeat_timeout: int = 60


class MasterNode:
    """Master Node - Cluster Coordinator"""
    
    def __init__(self, config: ClusterConfig = None):
        self.config = config or ClusterConfig()
        self.node_id = str(uuid.uuid4())
        self.nodes: Dict[str, NodeInfo] = {}
        self.jobs: Dict[str, Job] = {}
        self._running = False
        self._lock = threading.RLock()
        logger.info(f"MasterNode initialized: {self.node_id}")
    
    def start(self):
        self._running = True
        logger.info("MasterNode started")
    
    def stop(self):
        self._running = False
        logger.info("MasterNode stopped")
    
    def register_node(self, node_info: Dict) -> NodeInfo:
        with self._lock:
            node = NodeInfo(
                node_id=node_info.get("node_id", str(uuid.uuid4())),
                node_type=NodeType(node_info.get("type", "worker")),
                rank=len(self.nodes),
                host=node_info.get("host", ""),
                port=node_info.get("port", 0)
            )
            self.nodes[node.node_id] = node
            return node
    
    def submit_job(self, config: JobConfig) -> Job:
        with self._lock:
            job = Job(job_id=config.job_id, config=config)
            self.jobs[job.job_id] = job
            return job
    
    def get_cluster_state(self) -> Dict:
        return {
            "node_id": self.node_id,
            "total_nodes": len(self.nodes),
            "total_jobs": len(self.jobs)
        }


class WorkerNode:
    """Worker Node - Training Executor"""
    
    def __init__(self, rank: int = 0, world_size: int = 1):
        self.node_id = str(uuid.uuid4())
        self.rank = rank
        self.world_size = world_size
        self._running = False
        self.model = None
        self.optimizer = None
        logger.info(f"WorkerNode initialized: rank={rank}")
    
    def initialize(self) -> bool:
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.rank % torch.cuda.device_count())
            logger.info(f"Worker initialized: rank={self.rank}")
            return True
        except Exception as e:
            logger.error(f"Worker init failed: {e}")
            return False
    
    def set_model(self, model):
        self.model = model
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def train_epoch(self, data_loader, epoch: int) -> Dict[str, float]:
        if not self.model or not self.optimizer:
            raise RuntimeError("Model and optimizer must be set")
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for data, target in data_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ParameterServer:
    """Parameter Server - Parameter storage and updates"""
    
    def __init__(self, rank: int = 0):
        self.server_id = str(uuid.uuid4())
        self.rank = rank
        self.parameters: Dict[str, torch.Tensor] = {}
        logger.info(f"ParameterServer initialized: rank={rank}")
    
    def register_parameters(self, param_dict: Dict[str, torch.Tensor]):
        for name, tensor in param_dict.items():
            self.parameters[name] = tensor.clone()
    
    def get_parameter(self, name: str) -> Optional[torch.Tensor]:
        return self.parameters.get(name)
    
    def receive_gradient(self, name: str, gradient: torch.Tensor, lr: float = 0.001):
        if name in self.parameters:
            self.parameters[name].sub_(lr * gradient)


class Trainer:
    """High-level Trainer"""
    
    def __init__(self, model, optimizer, config: Dict = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}
        self.current_epoch = 0
    
    def train(self, train_loader, epochs: int = 10):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for data, target in train_loader:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    data = data.cuda()
                    target = target.cuda()
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            self.current_epoch += 1
        
        return {"final_loss": avg_loss}


__all__ = [
    "NodeType", "NodeStatus", "JobStatus",
    "NodeInfo", "JobConfig", "Job", "ClusterConfig",
    "MasterNode", "WorkerNode", "ParameterServer", "Trainer"
]
