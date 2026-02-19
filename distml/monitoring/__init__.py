"""
Monitoring Module - Metrics Collection and System Monitoring
"""

import logging
import time
import threading
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import socket

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_used: int
    memory_total: int
    memory_percent: float


@dataclass
class TrainingMetrics:
    timestamp: datetime
    step: int
    epoch: int
    loss: float
    accuracy: float


class MetricsCollector:
    """Metrics Collector"""
    
    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = {}
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def counter(self, name: str, value: float = 1.0):
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
    
    def gauge(self, name: str, value: float):
        with self._lock:
            self._gauges[name] = value
    
    def histogram(self, name: str, value: float):
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = deque(maxlen=1000)
            self._histograms[name].append(value)
    
    def get_metrics(self) -> Dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: len(v) for k, v in self._histograms.items()}
            }


class SystemMonitor:
    """System Resource Monitor"""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self._running = False
        self._thread = None
        self._callbacks = []
        self._latest: Optional[SystemMetrics] = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
    
    def get_metrics(self) -> Optional[SystemMetrics]:
        return self._latest
    
    def _monitor_loop(self):
        import psutil
        
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                
                self._latest = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=cpu,
                    memory_used=mem.used,
                    memory_total=mem.total,
                    memory_percent=mem.percent
                )
                
                for callback in self._callbacks:
                    callback(self._latest)
                
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    def on_metrics(self, callback: Callable):
        self._callbacks.append(callback)


class TrainingMonitor:
    """Training Monitor"""
    
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.loss_history = deque(maxlen=1000)
        self._callbacks = []
    
    def log_batch(self, loss: float):
        self.step += 1
        self.loss_history.append(loss)
    
    def log_epoch(self, epoch: int):
        self.epoch = epoch
    
    def get_metrics(self) -> TrainingMetrics:
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
        
        return TrainingMetrics(
            timestamp=datetime.utcnow(),
            step=self.step,
            epoch=self.epoch,
            loss=avg_loss,
            accuracy=0.0
        )


class TensorBoardLogger:
    """TensorBoard Logger"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            logger.warning("TensorBoard not available")
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def flush(self):
        if self.writer:
            self.writer.flush()


__all__ = ["SystemMetrics", "TrainingMetrics", "MetricsCollector", "SystemMonitor", "TrainingMonitor", "TensorBoardLogger"]
