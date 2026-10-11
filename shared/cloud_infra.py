"""
shared/cloud_infra.py
======================
Simulates the "Unified Cloud Architecture" layer from the system diagram:
    - Elastic Compute   -> ElasticComputePool (auto-scaling worker pool)
    - Distributed Storage -> DistributedStorage (key-value data lake)
    - Secure API Gateway  -> APIGateway (auth-gated request router)
    - Orchestration Layer -> Orchestrator (job scheduler across both channels)

In a real deployment these map to:
    ElasticComputePool  -> AWS Auto Scaling Group / K8s HPA / Azure VMSS
    DistributedStorage  -> AWS S3 + DynamoDB / Azure Blob + Cosmos DB
    APIGateway          -> AWS API Gateway / Azure API Management
    Orchestrator        -> Apache Airflow + Kubernetes
"""

import time
import uuid
import threading
import queue
from dataclasses import dataclass, field
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor


class DistributedStorage:
    """Simulates a data lake (object storage + time-series DB combined)."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store.setdefault(key, [])
            self._store[key].append({"ts": time.time(), "data": value})

    def get_latest(self, key: str) -> Any:
        with self._lock:
            records = self._store.get(key, [])
            return records[-1]["data"] if records else None

    def get_all(self, key: str) -> list:
        with self._lock:
            return list(self._store.get(key, []))

    def keys(self) -> list:
        with self._lock:
            return list(self._store.keys())


class ElasticComputePool:
    """
    Simulates auto-scaling compute. Pool size grows/shrinks based on
    queued workload, mirroring how AWS ASG / K8s HPA react to load.
    """

    def __init__(self, min_workers: int = 2, max_workers: int = 8):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def scale(self, pending_jobs: int) -> int:
        """Adjust pool size based on pending workload (simplified HPA logic)."""
        with self._lock:
            target = min(self.max_workers, max(self.min_workers, pending_jobs))
            if target != self.current_workers:
                self.current_workers = target
            return self.current_workers

    def submit(self, fn: Callable, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=True)


@dataclass
class Alert:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    channel: str = ""
    severity: str = "info"
    message: str = ""
    timestamp: float = field(default_factory=time.time)


class APIGateway:
    """
    Simulates a secure API gateway: single entry point that routes
    requests to the correct channel and logs access.
    """

    def __init__(self):
        self._routes: dict[str, Callable] = {}
        self._access_log: list[dict] = []

    def register_route(self, path: str, handler: Callable):
        self._routes[path] = handler

    def request(self, path: str, api_key: str = "demo-key", **kwargs):
        self._access_log.append({"path": path, "ts": time.time(), "key": api_key})
        if path not in self._routes:
            raise ValueError(f"No route registered for '{path}'")
        return self._routes[path](**kwargs)

    def access_log(self) -> list:
        return list(self._access_log)


class MessageQueue:
    """
    Simulates an MQTT/Kafka-style pub-sub channel used for the
    real-time injury-prevention data path.
    """

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()
        self._subscribers: list[Callable] = []

    def publish(self, message: dict):
        self._q.put(message)
        for sub in self._subscribers:
            sub(message)

    def subscribe(self, handler: Callable):
        self._subscribers.append(handler)


class Orchestrator:
    """
    Simulates the orchestration layer (Airflow-style) that schedules
    batch jobs (scouting) and monitors streaming services (injury
    prevention) on a shared compute pool.
    """

    def __init__(self, compute_pool: ElasticComputePool):
        self.compute_pool = compute_pool
        self.jobs: list[dict] = []

    def schedule_batch_job(self, name: str, fn: Callable, *args, **kwargs):
        self.compute_pool.scale(pending_jobs=len(self.jobs) + 1)
        future = self.compute_pool.submit(fn, *args, **kwargs)
        self.jobs.append({"name": name, "type": "batch", "future": future})
        return future

    def status(self) -> list[dict]:
        return [
            {"name": j["name"], "type": j["type"], "done": j["future"].done()}
            for j in self.jobs
        ]
