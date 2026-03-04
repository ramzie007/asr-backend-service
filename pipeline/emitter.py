from __future__ import annotations
import json
import time
from pipeline.contracts import WorkerResult


class Emitter:
    """Converts WorkerResult to typed WebSocket JSON messages."""

    def format(self, result: WorkerResult, queue_depth: int = 0) -> str:
        result.emit_ts = time.monotonic()
        msg_type = "error" if result.error else ("final" if result.is_final else "partial")
        msg: dict = {
            "type": msg_type,
            "segment_id": result.segment_id,
            "stream_id": result.stream_id,
            "transcript": result.transcript,
            "start_ts": result.start_ts,
            "end_ts": result.end_ts,
            "confidence": result.confidence,
            "is_final": result.is_final,
            "meta": {"queue_depth": queue_depth},
        }
        if result.error:
            msg["error"] = result.error
        return json.dumps(msg)

    def format_status(self, event: str, segment_id: str = "", message: str = "") -> str:
        return json.dumps({
            "type": "status",
            "event": event,
            "segment_id": segment_id,
            "message": message,
        })
