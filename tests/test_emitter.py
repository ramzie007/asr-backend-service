import json
from pipeline.emitter import Emitter
from pipeline.contracts import WorkerResult


def make_result(**kwargs):
    defaults = dict(
        segment_id="seg-1", stream_id="stream-1", transcript="hello",
        start_ts=1.0, end_ts=2.0, enqueue_ts=1.0,
        inference_start_ts=1.1, inference_end_ts=1.5,
        emit_ts=0.0, is_final=True, confidence=None, error=None,
    )
    defaults.update(kwargs)
    return WorkerResult(**defaults)


def test_final_message_type():
    msg = json.loads(Emitter().format(make_result(is_final=True), queue_depth=2))
    assert msg["type"] == "final"
    assert msg["meta"]["queue_depth"] == 2
    assert msg["is_final"] is True


def test_partial_message_type():
    msg = json.loads(Emitter().format(make_result(is_final=False)))
    assert msg["type"] == "partial"


def test_error_message_type():
    msg = json.loads(Emitter().format(make_result(error="timeout", transcript="")))
    assert msg["type"] == "error"
    assert msg["error"] == "timeout"


def test_emit_ts_stamped():
    result = make_result()
    assert result.emit_ts == 0.0
    Emitter().format(result)
    assert result.emit_ts > 0.0


def test_status_message_schema():
    msg = json.loads(Emitter().format_status("segment_dropped", "seg-1", "queue full"))
    assert msg["type"] == "status"
    assert msg["event"] == "segment_dropped"
    assert msg["segment_id"] == "seg-1"
