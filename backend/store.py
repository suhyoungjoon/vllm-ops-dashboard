"""최근 N초 동안의 메트릭 스냅샷을 프로세스 메모리 내 deque로 유지한다.

영구 저장소는 두지 않는다 — 재시작 시 초기화되는 것이 의도된 동작이다.
"""

from collections import deque
from dataclasses import dataclass

from backend.poller import VLLMMetrics

DEFAULT_RETENTION_SECONDS = 600.0  # 10분


@dataclass
class MetricsSnapshot:
    timestamp: float
    metrics: VLLMMetrics


class MetricsStore:
    def __init__(self, retention_seconds: float = DEFAULT_RETENTION_SECONDS):
        self._retention_seconds = retention_seconds
        self._snapshots: deque[MetricsSnapshot] = deque()

    def add(self, metrics: VLLMMetrics, timestamp: float) -> None:
        self._snapshots.append(MetricsSnapshot(timestamp=timestamp, metrics=metrics))
        self._prune(timestamp)

    def _prune(self, latest_timestamp: float) -> None:
        cutoff = latest_timestamp - self._retention_seconds
        while self._snapshots and self._snapshots[0].timestamp < cutoff:
            self._snapshots.popleft()

    def get_recent(self) -> list[MetricsSnapshot]:
        return list(self._snapshots)
