from config import CaptioningSLA
from pipeline.contracts import WorkerRequest, WorkerResult


def test_worker_request_fields():
    req = WorkerRequest(
        segment_id="seg-1",
        stream_id="stream-1",
        audio_bytes=b"\x00" * 960,
        sample_rate=16000,
        start_ts=1.0,
        end_ts=2.0,
        enqueue_ts=0.0,
    )
    assert req.segment_id == "seg-1"
    assert req.sample_rate == 16000


def test_worker_result_optional_fields_default_none():
    res = WorkerResult(
        segment_id="seg-1",
        stream_id="stream-1",
        transcript="hello",
        start_ts=1.0,
        end_ts=2.0,
        enqueue_ts=1.0,
        inference_start_ts=1.1,
        inference_end_ts=1.5,
        emit_ts=0.0,
        is_final=True,
    )
    assert res.confidence is None
    assert res.error is None


def test_worker_result_error_field():
    res = WorkerResult(
        segment_id="seg-1",
        stream_id="stream-1",
        transcript="",
        start_ts=1.0,
        end_ts=2.0,
        enqueue_ts=1.0,
        inference_start_ts=1.1,
        inference_end_ts=1.5,
        emit_ts=0.0,
        is_final=True,
        error="timeout",
    )
    assert res.error == "timeout"


def test_captioning_sla_defaults():
    sla = CaptioningSLA()
    assert sla.ftl_p95_ms == 800.0
    assert sla.fl_p95_ms == 1200.0
    assert sla.rtf_p95 == 1.0
    assert sla.max_queue_growth is True
    assert sla.drop_policy == "drop_newest"


def test_captioning_sla_custom_targets():
    sla = CaptioningSLA(ftl_p95_ms=500.0, fl_p95_ms=900.0, rtf_p95=0.8)
    assert sla.ftl_p95_ms == 500.0
    assert sla.fl_p95_ms == 900.0
    assert sla.rtf_p95 == 0.8
