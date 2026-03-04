from __future__ import annotations
import time
import uuid
import webrtcvad
from config import PipelineConfig
from pipeline.contracts import WorkerRequest


class VADSegmenter:
    """Detects speech segments using webrtcvad with configurable hangover."""

    FRAME_DURATION_MS = 30

    def __init__(self, config: PipelineConfig):
        self._vad = webrtcvad.Vad(config.vad_aggressiveness)
        self._hangover_frames = max(1, config.vad_hangover_ms // self.FRAME_DURATION_MS)
        self._sample_rate = config.sample_rate
        self._speech_buf: list[bytes] = []
        self._hangover_count: int = 0
        self._in_speech: bool = False
        self._start_ts: float = 0.0

    def process_frame(self, frame: bytes, stream_id: str) -> WorkerRequest | None:
        """Process one 30ms frame. Returns WorkerRequest when segment complete."""
        is_speech = self._vad.is_speech(frame, self._sample_rate)
        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._start_ts = time.monotonic()
                self._speech_buf = []
                self._hangover_count = 0
            self._speech_buf.append(frame)
            self._hangover_count = 0
            return None

        if self._in_speech:
            self._speech_buf.append(frame)
            self._hangover_count += 1
            if self._hangover_count >= self._hangover_frames:
                return self._emit_segment(stream_id)
        return None

    def flush(self, stream_id: str) -> WorkerRequest | None:
        """Force-emit any buffered speech (call on stream end)."""
        if self._in_speech and self._speech_buf:
            return self._emit_segment(stream_id)
        return None

    def _emit_segment(self, stream_id: str) -> WorkerRequest:
        end_ts = time.monotonic()
        audio_bytes = b"".join(self._speech_buf)
        self._in_speech = False
        self._speech_buf = []
        self._hangover_count = 0
        return WorkerRequest(
            segment_id=str(uuid.uuid4()),
            stream_id=stream_id,
            audio_bytes=audio_bytes,
            sample_rate=self._sample_rate,
            start_ts=self._start_ts,
            end_ts=end_ts,
            enqueue_ts=0.0,
        )
