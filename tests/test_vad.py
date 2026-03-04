import numpy as np
from unittest.mock import patch
from pipeline.vad import VADSegmenter
from config import PipelineConfig

FRAME_BYTES = 960


def make_config(vad_hangover_ms: int = 300) -> PipelineConfig:
    return PipelineConfig(sample_rate=16000, vad_aggressiveness=2, vad_hangover_ms=vad_hangover_ms)


def silence_frame() -> bytes:
    return bytes(FRAME_BYTES)


def speech_frame() -> bytes:
    samples = np.random.randint(8000, 16000, FRAME_BYTES // 2, dtype=np.int16)
    return samples.tobytes()


def test_silence_emits_nothing():
    vad = VADSegmenter(make_config())
    results = [vad.process_frame(silence_frame(), "s") for _ in range(10)]
    assert all(r is None for r in results)


def test_speech_then_hangover_emits_segment():
    # hangover_ms=90 → 3 frames hangover
    vad = VADSegmenter(make_config(vad_hangover_ms=90))
    with patch.object(vad._vad, "is_speech", side_effect=[True] * 5 + [False] * 3):
        results = [vad.process_frame(speech_frame(), "s") for _ in range(8)]
    segment = next((r for r in results if r is not None), None)
    assert segment is not None
    assert segment.stream_id == "s"
    assert len(segment.segment_id) > 0
    assert segment.start_ts < segment.end_ts
    # 5 speech frames + 3 hangover frames
    assert len(segment.audio_bytes) == 8 * FRAME_BYTES


def test_timestamps_are_monotonic():
    vad = VADSegmenter(make_config(vad_hangover_ms=90))
    with patch.object(vad._vad, "is_speech", side_effect=[True] * 3 + [False] * 3):
        results = [vad.process_frame(speech_frame(), "s") for _ in range(6)]
    segment = next((r for r in results if r is not None), None)
    assert segment is not None
    assert segment.end_ts >= segment.start_ts


def test_flush_emits_buffered_speech():
    vad = VADSegmenter(make_config(vad_hangover_ms=300))
    with patch.object(vad._vad, "is_speech", return_value=True):
        for _ in range(5):
            vad.process_frame(speech_frame(), "s")
    segment = vad.flush("s")
    assert segment is not None
    assert len(segment.audio_bytes) == 5 * FRAME_BYTES
