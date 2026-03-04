from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WorkerRequest:
    segment_id: str
    stream_id: str
    audio_bytes: bytes
    sample_rate: int
    start_ts: float
    end_ts: float
    enqueue_ts: float = 0.0


@dataclass
class WorkerResult:
    segment_id: str
    stream_id: str
    transcript: str
    start_ts: float
    end_ts: float
    enqueue_ts: float
    inference_start_ts: float
    inference_end_ts: float
    emit_ts: float
    is_final: bool
    confidence: float | None = None
    error: str | None = None
