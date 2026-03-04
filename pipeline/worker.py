from __future__ import annotations
import multiprocessing as mp
import signal
import time
import numpy as np
from faster_whisper import WhisperModel
from pipeline.contracts import WorkerRequest, WorkerResult
from config import PipelineConfig


def process_segment(model: WhisperModel, request: WorkerRequest) -> WorkerResult:
    """Run inference on one segment. Pure function — testable without subprocess."""
    inference_start_ts = time.monotonic()
    try:
        audio = np.frombuffer(request.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio, beam_size=1, language="en")
        transcript = " ".join(s.text for s in segments).strip()
        error = None
    except Exception as e:
        transcript = ""
        error = str(e)
    inference_end_ts = time.monotonic()

    return WorkerResult(
        segment_id=request.segment_id,
        stream_id=request.stream_id,
        transcript=transcript,
        start_ts=request.start_ts,
        end_ts=request.end_ts,
        enqueue_ts=request.enqueue_ts,
        inference_start_ts=inference_start_ts,
        inference_end_ts=inference_end_ts,
        emit_ts=0.0,  # set by emitter
        is_final=True,
        confidence=None,
        error=error,
    )


def worker_process(
    worker_id: int,
    config: PipelineConfig,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
) -> None:
    """Worker process entry point. Loads model once, loops until shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    model = WhisperModel(config.model_size, device="cpu", compute_type="int8")
    try:
        while not shutdown_event.is_set():
            try:
                request: WorkerRequest = work_queue.get(timeout=1.0)
            except Exception:
                continue
            result = process_segment(model, request)
            result_queue.put(result)
    except KeyboardInterrupt:
        return
