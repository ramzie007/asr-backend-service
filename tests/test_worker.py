import numpy as np
from unittest.mock import MagicMock
from pipeline.contracts import WorkerRequest
from config import PipelineConfig


def make_request():
    samples = np.zeros(16000, dtype=np.int16)
    return WorkerRequest(
        segment_id="seg-1",
        stream_id="stream-1",
        audio_bytes=samples.tobytes(),
        sample_rate=16000,
        start_ts=1.0,
        end_ts=2.0,
        enqueue_ts=1.0,
    )


def test_process_segment_returns_transcript():
    from pipeline.worker import process_segment
    mock_model = MagicMock()
    mock_seg = MagicMock()
    mock_seg.text = " hello world"
    mock_model.transcribe.return_value = ([mock_seg], MagicMock())

    result = process_segment(mock_model, make_request())

    assert result.segment_id == "seg-1"
    assert result.transcript == "hello world"
    assert result.is_final is True
    assert result.error is None
    assert result.inference_end_ts >= result.inference_start_ts


def test_process_segment_handles_inference_error():
    from pipeline.worker import process_segment
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("OOM")

    result = process_segment(mock_model, make_request())

    assert result.transcript == ""
    assert result.error == "OOM"
    assert result.is_final is True


def test_process_segment_normalizes_audio_to_float32():
    from pipeline.worker import process_segment
    mock_model = MagicMock()
    captured = {}

    def capture(audio, **kwargs):
        captured["audio"] = audio
        return ([], MagicMock())

    mock_model.transcribe.side_effect = capture
    process_segment(mock_model, make_request())

    assert captured["audio"].dtype == np.float32
    assert captured["audio"].max() <= 1.0
    assert captured["audio"].min() >= -1.0
