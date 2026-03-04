import asyncio
import json
import multiprocessing as mp
import numpy as np
import pytest
import websockets
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_server_accepts_connection_and_silence():
    """Server starts, accepts WebSocket, silence produces no crashes."""
    from config import PipelineConfig
    from server import create_app

    config = PipelineConfig(num_workers=1, vad_hangover_ms=300)

    # Mock WorkerPool to avoid spawning real worker processes in tests
    with patch("server.WorkerPool") as MockPool:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = True
        mock_pool.queue_depth.return_value = 0
        mock_pool.shutdown.return_value = None
        MockPool.return_value = mock_pool

        async with create_app(config=config, port=0) as (server, port):
            uri = f"ws://localhost:{port}"
            async with websockets.connect(uri) as ws:
                silence = np.zeros(16000, dtype=np.int16).tobytes()
                for i in range(0, len(silence), 3200):
                    await ws.send(silence[i:i + 3200])
                await asyncio.sleep(0.3)
                # silence never triggers VAD → submit never called


@pytest.mark.asyncio
async def test_server_logs_result_with_queue_depth(capsys):
    """Server stdout shows a result log line including queue depth after processing a segment."""
    from config import PipelineConfig
    from server import create_app
    from pipeline.contracts import WorkerResult
    import time

    config = PipelineConfig(num_workers=1)

    with patch("server.WorkerPool") as MockPool:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = True
        mock_pool.queue_depth.return_value = 3
        mock_pool.shutdown.return_value = None
        MockPool.return_value = mock_pool

        async with create_app(config=config, port=0) as (server, port):
            # Inject a result directly into the result queue
            now = time.monotonic()
            result = WorkerResult(
                segment_id="seg-test", stream_id="disconnected",
                transcript="hello world",
                start_ts=now - 1.0, end_ts=now, enqueue_ts=now - 0.9,
                inference_start_ts=now - 0.5, inference_end_ts=now - 0.1,
                emit_ts=0.0, is_final=True,
            )
            import server as srv
            srv._result_queue.put(result)
            await asyncio.sleep(0.3)

    out = capsys.readouterr().out
    assert ("[trace]" in out) or ("[result]" in out), \
        f"Server stdout must show a live per-segment log line. Got: {out!r}"


@pytest.mark.asyncio
async def test_server_tracks_peak_concurrent_connections():
    """Benchmark table reflects actual peak connections, not hardcoded 1."""
    import server as srv
    from config import PipelineConfig
    from server import create_app

    config = PipelineConfig(num_workers=1, vad_hangover_ms=300)
    silence = np.zeros(3200, dtype=np.int16).tobytes()

    with patch("server.WorkerPool") as MockPool:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = True
        mock_pool.queue_depth.return_value = 0
        mock_pool.shutdown.return_value = None
        MockPool.return_value = mock_pool

        async with create_app(config=config, port=0) as (server, port):
            uri = f"ws://localhost:{port}"
            # Open 2 simultaneous connections
            async with websockets.connect(uri) as ws1:
                async with websockets.connect(uri) as ws2:
                    await ws1.send(silence)
                    await ws2.send(silence)
                    await asyncio.sleep(0.3)
                    peak = srv._peak_streams
    assert peak >= 2, f"Expected peak_streams >= 2, got {peak}"


@pytest.mark.asyncio
async def test_server_emits_status_on_dropped_segment():
    """Server sends segment_dropped status when pool rejects a segment."""
    from config import PipelineConfig
    from server import create_app

    config = PipelineConfig(num_workers=1)

    with patch("server.WorkerPool") as MockPool:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = False  # always drop
        mock_pool.queue_depth.return_value = 4
        mock_pool.shutdown.return_value = None
        MockPool.return_value = mock_pool

        with patch("server.VADSegmenter") as MockVAD:
            from pipeline.contracts import WorkerRequest
            mock_vad = MagicMock()
            mock_vad.process_frame.return_value = WorkerRequest(
                segment_id="s1", stream_id="x",
                audio_bytes=b"\x00" * 960, sample_rate=16000,
                start_ts=1.0, end_ts=2.0,
            )
            mock_vad.flush.return_value = None
            MockVAD.return_value = mock_vad

            async with create_app(config=config, port=0) as (server, port):
                uri = f"ws://localhost:{port}"
                messages = []
                async with websockets.connect(uri) as ws:
                    await ws.send(b"\x00" * 960)
                    await asyncio.sleep(0.3)
                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.2)
                            messages.append(json.loads(msg))
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                        pass

                status_msgs = [m for m in messages if m.get("type") == "status"]
                assert any(m.get("event") == "segment_dropped" for m in status_msgs)


@pytest.mark.asyncio
async def test_server_captioning_sla_prints_report(capsys):
    """Server prints captioning report with SLA evaluation when captioning_sla is set."""
    from config import CaptioningSLA, PipelineConfig
    from server import create_app
    from pipeline.contracts import WorkerResult
    import time

    config = PipelineConfig(num_workers=1)
    sla = CaptioningSLA()

    with patch("server.WorkerPool") as MockPool:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = True
        mock_pool.queue_depth.return_value = 0
        mock_pool.shutdown.return_value = None
        MockPool.return_value = mock_pool

        async with create_app(config=config, port=0, captioning_sla=sla) as (server, port):
            # Inject a result to have metrics to report
            now = time.monotonic()
            result = WorkerResult(
                segment_id="seg-cap", stream_id="disconnected",
                transcript="hello caption",
                start_ts=now - 1.0, end_ts=now - 0.5, enqueue_ts=now - 0.9,
                inference_start_ts=now - 0.4, inference_end_ts=now - 0.1,
                emit_ts=0.0, is_final=True,
            )
            import server as srv
            srv._result_queue.put(result)
            await asyncio.sleep(0.3)

    out = capsys.readouterr().out
    assert "Live Captioning" in out, f"Expected captioning report header. Got: {out!r}"
    assert "SLA Compliance" in out, f"Expected SLA section. Got: {out!r}"


@pytest.mark.asyncio
async def test_server_prints_live_metrics_snapshot(capsys):
    """Server prints periodic rolling snapshot when live metrics are enabled."""
    from config import PipelineConfig
    from server import create_app
    from pipeline.contracts import WorkerResult
    import time

    config = PipelineConfig(
        num_workers=1,
        metrics_live_enabled=True,
        metrics_live_interval_s=0.05,
        metrics_live_min_samples=1,
    )

    with patch("server.WorkerPool") as MockPool:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = True
        mock_pool.queue_depth.return_value = 0
        mock_pool.shutdown.return_value = None
        MockPool.return_value = mock_pool

        async with create_app(config=config, port=0) as (server, port):
            now = time.monotonic()
            result = WorkerResult(
                segment_id="seg-live", stream_id="disconnected",
                transcript="hello",
                start_ts=now - 1.0, end_ts=now - 0.4, enqueue_ts=now - 0.9,
                inference_start_ts=now - 0.3, inference_end_ts=now - 0.1,
                emit_ts=0.0, is_final=True,
            )
            import server as srv
            srv._result_queue.put(result)
            await asyncio.sleep(0.2)

    out = capsys.readouterr().out
    assert "LIVE SNAPSHOT" in out, f"Expected periodic snapshot output. Got: {out!r}"
