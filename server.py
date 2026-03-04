from __future__ import annotations
import asyncio
import json
import multiprocessing as mp
import signal
import uuid
from contextlib import asynccontextmanager, suppress
from typing import AsyncGenerator

import websockets
from websockets.asyncio.server import ServerConnection, Server

from config import CaptioningSLA, PipelineConfig
from pipeline.contracts import WorkerResult
from pipeline.emitter import Emitter
from pipeline.ingestion import AudioIngestion
from pipeline.metrics import MetricsCollector
from pipeline.pool import WorkerPool
from pipeline.vad import VADSegmenter

# Module-level state — shared across connections
_config: PipelineConfig
_pool: WorkerPool
_result_queue: mp.Queue
_metrics: MetricsCollector
_emitter: Emitter
_connections: dict[str, ServerConnection] = {}
_drop_count: int = 0
_peak_streams: int = 0


async def _live_metrics_reporter() -> None:
    """Periodically print rolling latency snapshots while server is running."""
    while True:
        await asyncio.sleep(_config.metrics_live_interval_s)
        if _metrics.count() < _config.metrics_live_min_samples:
            continue
        _metrics.print_live_snapshot(
            streams=max(1, _peak_streams),
            drop_policy=_config.drop_policy,
            num_workers=_config.num_workers,
            drops=_drop_count,
        )


async def _result_dispatcher() -> None:
    """Read results from worker result_queue, route to correct WebSocket."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            result: WorkerResult = await loop.run_in_executor(
                None, lambda: _result_queue.get(timeout=0.1)
            )
        except Exception:
            await asyncio.sleep(0.01)
            continue

        _pool.mark_completed()
        payload = _emitter.format(result, queue_depth=_pool.queue_depth())
        _metrics.record(result)
        q = _pool.queue_depth()
        # Structured segment trace — visible in demo, enables per-segment latency attribution
        queue_wait_ms = (result.inference_start_ts - result.enqueue_ts) * 1000
        inference_ms = (result.inference_end_ts - result.inference_start_ts) * 1000
        print(
            f"[trace] seg={result.segment_id[:8]} "
            f"q={q} "
            f"queue_wait={queue_wait_ms:.0f}ms "
            f"inference={inference_ms:.0f}ms "
            f"transcript='{result.transcript[:40]}'"
        )
        ws = _connections.get(result.stream_id)
        if ws is None:
            continue
        try:
            await ws.send(payload)
        except websockets.exceptions.ConnectionClosed:
            _connections.pop(result.stream_id, None)


async def _handle_connection(websocket: ServerConnection) -> None:
    global _drop_count, _peak_streams
    stream_id = str(uuid.uuid4())
    _connections[stream_id] = websocket
    _peak_streams = max(_peak_streams, len(_connections))
    ingestion = AudioIngestion(_config)
    vad = VADSegmenter(_config)

    try:
        async for message in websocket:
            if isinstance(message, str):
                if message == "__end_stream__":
                    final_request = vad.flush(stream_id)
                    if final_request:
                        accepted = _pool.submit(final_request)
                        if not accepted:
                            _drop_count += 1
                            await websocket.send(
                                _emitter.format_status("segment_dropped", final_request.segment_id, "queue full")
                            )
                continue
            if not isinstance(message, bytes):
                continue
            for frame in ingestion.feed(message):
                request = vad.process_frame(frame, stream_id)
                if request is None:
                    continue
                accepted = _pool.submit(request)
                if not accepted:
                    _drop_count += 1
                    await websocket.send(
                        _emitter.format_status("segment_dropped", request.segment_id, "queue full")
                    )
        # Stream ended — flush any buffered speech
        final_request = vad.flush(stream_id)
        if final_request:
            accepted = _pool.submit(final_request)
            if not accepted:
                _drop_count += 1
                await websocket.send(
                    _emitter.format_status("segment_dropped", final_request.segment_id, "queue full")
                )
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _connections.pop(stream_id, None)


@asynccontextmanager
async def create_app(
    config: PipelineConfig | None = None,
    port: int = 8765,
    captioning_sla: CaptioningSLA | None = None,
) -> AsyncGenerator[tuple[Server, int], None]:
    global _config, _pool, _result_queue, _metrics, _emitter, _connections, _drop_count, _peak_streams
    _config = config or PipelineConfig()
    _result_queue = mp.Queue()
    _pool = WorkerPool(_config, _result_queue)
    _metrics = MetricsCollector()
    _emitter = Emitter()
    _connections = {}
    _drop_count = 0
    _peak_streams = 0

    dispatcher = asyncio.create_task(_result_dispatcher())
    live_reporter = None
    if _config.metrics_live_enabled:
        live_reporter = asyncio.create_task(_live_metrics_reporter())
    async with websockets.serve(_handle_connection, _config.host, port) as server:
        actual_port = server.sockets[0].getsockname()[1]
        try:
            yield server, actual_port
        finally:
            deadline = asyncio.get_event_loop().time() + _config.shutdown_timeout_s
            pending_fn = getattr(_pool, "pending", None)
            while callable(pending_fn) and asyncio.get_event_loop().time() < deadline:
                pending = pending_fn()
                if not isinstance(pending, int) or pending <= 0:
                    break
                await asyncio.sleep(0.05)
            dispatcher.cancel()
            if live_reporter is not None:
                live_reporter.cancel()
            with suppress(asyncio.CancelledError):
                await dispatcher
            if live_reporter is not None:
                with suppress(asyncio.CancelledError):
                    await live_reporter
            _pool.shutdown(wait_for_drain=False)
            if captioning_sla:
                _metrics.print_captioning_report(
                    streams=max(1, _peak_streams),
                    drop_policy=_config.drop_policy,
                    num_workers=_config.num_workers,
                    drops=_drop_count,
                    sla=captioning_sla,
                )
            else:
                _metrics.print_table(
                    streams=max(1, _peak_streams),
                    drop_policy=_config.drop_policy,
                    num_workers=_config.num_workers,
                    drops=_drop_count,
                )


async def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Streaming ASR server")
    parser.add_argument(
        "--scenario", default="default", choices=["default", "live-captioning"],
        help="Server mode: 'default' or 'live-captioning' (prints SLA evaluation on shutdown)",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    sla = CaptioningSLA() if args.scenario == "live-captioning" else None

    def _on_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_signal)

    if sla:
        print(f"Live Captioning Mode | {config.host}:{config.port} | workers={config.num_workers} | model={config.model_size}")
        print(f"SLA targets: FTL p95 < {sla.ftl_p95_ms:.0f}ms | FL p95 < {sla.fl_p95_ms:.0f}ms | RTF p95 < {sla.rtf_p95:.1f}")
    else:
        print(f"Starting on {config.host}:{config.port} | workers={config.num_workers} | model={config.model_size}")

    async with create_app(config, config.port, captioning_sla=sla):
        try:
            await asyncio.Future()
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutting down...")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
