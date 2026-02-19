"""
Scheduling Module - Resource Management and Job Scheduling
"""

import logging
import time
import heapq
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    RESOURCES = "resources"
    DEADLINE = "deadline"


class AllocationStrategy(Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"


@dataclass
class ResourceRequest:
    job_id: str
    cpu_cores: int = 1
    memory_mb: int = 1024
    gpu_count: int = 0
    priority: int = 0


@dataclass
class NodeResources:
    node_id: str
    cpu_cores: int
    memory_mb: int
    gpu_count: int
    cpu_used: int = 0
    memory_used: int = 0
    gpu_used: int = 0
    
    def available(self) -> Dict[str, int]:
        return {
            "cpu_cores": self.cpu_cores - self.cpu_used,
            "memory_mb": self.memory_mb - self.memory_used,
            "gpu_count": self.gpu_count - self.gpu_used
        }
    
    def can_satisfy(self, request: ResourceRequest) -> bool:
        avail = self.available()
        return (avail["cpu_cores"] >= request.cpu_cores and
                avail["memory_mb"] >= request.memory_mb and
                avail["gpu_count"] >= request.gpu_count)
    
    def allocate(self, request: ResourceRequest):
        self.cpu_used += request.cpu_cores
        self.memory_used += request.memory_mb
        self.gpu_used += request.gpu_count
    
    def release(self, request: ResourceRequest):
        self.cpu_used -= request.cpu_cores
        self.memory_used -= request.memory_mb
        self.gpu_used -= request.gpu_count


class Scheduler:
    """Job Scheduler"""
    
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.RESOURCES):
        self.policy = policy
        self._nodes: Dict[str, NodeResources] = {}
        self._pending_queue: List[ResourceRequest] = []
        self._scheduled_jobs: Dict[str, ResourceRequest] = {}
        self._lock = threading.RLock()
    
    def add_node(self, node: NodeResources):
        with self._lock:
            self._nodes[node.node_id] = node
            logger.info(f"Node added: {node.node_id}")
    
    def remove_node(self, node_id: str):
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
    
    def submit(self, request: ResourceRequest) -> bool:
        with self._lock:
            self._pending_queue.append(request)
            self._sort_queue()
            logger.info(f"Request submitted: {request.job_id}")
            return True
    
    def _sort_queue(self):
        if self.policy == SchedulingPolicy.PRIORITY:
            self._pending_queue.sort(key=lambda r: -r.priority)
        elif self.policy == SchedulingPolicy.DEADLINE:
            self._pending_queue.sort(key=lambda r: r.priority)
    
    def schedule(self) -> List[ResourceRequest]:
        with self._lock:
            scheduled = []
            
            for request in self._pending_queue[:]:
                node = self._find_node(request)
                
                if node:
                    node.allocate(request)
                    self._scheduled_jobs[request.job_id] = request
                    self._pending_queue.remove(request)
                    scheduled.append(request)
                    logger.info(f"Job scheduled: {request.job_id} on {node.node_id}")
            
            return scheduled
    
    def _find_node(self, request: ResourceRequest) -> Optional[NodeResources]:
        for node in self._nodes.values():
            if node.can_satisfy(request):
                return node
        return None
    
    def complete(self, job_id: str):
        with self._lock:
            if job_id in self._scheduled_jobs:
                request = self._scheduled_jobs[job_id]
                
                for node in self._nodes.values():
                    if node.node_id == job_id.split("_node_")[-1]:
                        node.release(request)
                        break
                
                del self._scheduled_jobs[job_id]


class ResourceManager:
    """Resource Manager"""
    
    def __init__(self):
        self._total: Dict[str, int] = {"cpu": 0, "memory": 0, "gpu": 0}
        self._used: Dict[str, int] = {"cpu": 0, "memory": 0, "gpu": 0}
        self._lock = threading.RLock()
    
    def register_node(self, resources: NodeResources):
        with self._lock:
            self._total["cpu"] += resources.cpu_cores
            self._total["memory"] += resources.memory_mb
            self._total["gpu"] += resources.gpu_count
    
    def allocate(self, request: ResourceRequest) -> bool:
        with self._lock:
            if self._can_allocate(request):
                self._used["cpu"] += request.cpu_cores
                self._used["memory"] += request.memory_mb
                self._used["gpu"] += request.gpu_count
                return True
            return False
    
    def release(self, request: ResourceRequest):
        with self._lock:
            self._used["cpu"] -= request.cpu_cores
            self._used["memory"] -= request.memory_mb
            self._used["gpu"] -= request.gpu_count
    
    def _can_allocate(self, request: ResourceRequest) -> bool:
        return (self._total["cpu"] - self._used["cpu"] >= request.cpu_cores and
                self._total["memory"] - self._used["memory"] >= request.memory_mb and
                self._total["gpu"] - self._used["gpu"] >= request.gpu_count)
    
    def get_available(self) -> Dict[str, int]:
        with self._lock:
            return {
                "cpu": self._total["cpu"] - self._used["cpu"],
                "memory": self._total["memory"] - self._used["memory"],
                "gpu": self._total["gpu"] - self._used["gpu"]
            }


__all__ = ["SchedulingPolicy", "AllocationStrategy", "ResourceRequest", "NodeResources", "Scheduler", "ResourceManager"]
