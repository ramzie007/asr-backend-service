import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CaptioningSLA:
    """Service-level targets for live video captioning."""
    ftl_p95_ms: float = 800.0    # first-token latency: first audio byte -> first caption
    fl_p95_ms: float = 1200.0    # finalization latency: VAD end -> final caption
    rtf_p95: float = 1.0         # real-time factor must stay below 1.0
    max_queue_growth: bool = True # enforce zero unbounded queue growth
    drop_policy: str = "drop_newest"  # real-time recording: prefer dropping over delay


@dataclass
class PipelineConfig:
    num_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) // 2))
    queue_maxsize: int = 0  # resolved in __post_init__
    drop_policy: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"
    worker_timeout_s: float = 10.0
    shutdown_timeout_s: float = 5.0
    worker_join_timeout_s: float = 5.0
    vad_aggressiveness: int = 2
    vad_hangover_ms: int = 300
    model_size: str = "base"
    partial_emit_window_s: float = 2.0
    sample_rate: int = 16000
    host: str = "localhost"
    port: int = 8765
    metrics_live_enabled: bool = False
    metrics_live_interval_s: float = 5.0
    metrics_live_min_samples: int = 10

    def __post_init__(self):
        if self.queue_maxsize == 0:
            self.queue_maxsize = self.num_workers * 2
