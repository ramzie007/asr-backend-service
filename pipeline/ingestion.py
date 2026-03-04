from config import PipelineConfig


class AudioIngestion:
    """Buffers raw PCM bytes and yields fixed-size 30ms frames."""

    def __init__(self, config: PipelineConfig):
        # 30ms at sample_rate, int16 = 2 bytes/sample
        self._frame_bytes = int(config.sample_rate * 0.030) * 2
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """Accept raw PCM bytes, return list of complete 30ms frames."""
        self._buf.extend(data)
        frames = []
        while len(self._buf) >= self._frame_bytes:
            frames.append(bytes(self._buf[: self._frame_bytes]))
            del self._buf[: self._frame_bytes]
        return frames

    def reset(self) -> None:
        self._buf.clear()
