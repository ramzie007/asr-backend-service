from pipeline.ingestion import AudioIngestion
from config import PipelineConfig

FRAME_BYTES = 960  # 30ms at 16kHz int16 = 480 samples * 2 bytes


def make_config():
    return PipelineConfig(sample_rate=16000)


def test_yields_complete_frames():
    ing = AudioIngestion(make_config())
    frames = ing.feed(bytes(FRAME_BYTES * 3))
    assert len(frames) == 3
    assert all(len(f) == FRAME_BYTES for f in frames)


def test_buffers_partial_frame():
    ing = AudioIngestion(make_config())
    frames = ing.feed(bytes(FRAME_BYTES - 10))
    assert len(frames) == 0


def test_partial_then_complete():
    ing = AudioIngestion(make_config())
    ing.feed(bytes(FRAME_BYTES - 10))
    frames = ing.feed(bytes(20))
    assert len(frames) == 1
    assert len(frames[0]) == FRAME_BYTES


def test_reset_clears_buffer():
    ing = AudioIngestion(make_config())
    ing.feed(bytes(FRAME_BYTES - 10))
    ing.reset()
    frames = ing.feed(bytes(FRAME_BYTES))
    assert len(frames) == 1
