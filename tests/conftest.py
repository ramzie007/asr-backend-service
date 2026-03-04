import pytest
from config import PipelineConfig


@pytest.fixture
def config():
    return PipelineConfig(
        num_workers=2,
        vad_aggressiveness=2,
        vad_hangover_ms=300,
        model_size="base",
        sample_rate=16000,
    )
