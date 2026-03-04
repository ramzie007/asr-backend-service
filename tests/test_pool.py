import multiprocessing as mp
import queue
from pipeline.pool import WorkerPool
from pipeline.contracts import WorkerRequest
from config import PipelineConfig


def make_request(segment_id="seg-1"):
    return WorkerRequest(
        segment_id=segment_id,
        stream_id="stream-1",
        audio_bytes=b"\x00" * 960,
        sample_rate=16000,
        start_ts=1.0,
        end_ts=2.0,
        enqueue_ts=0.0,
    )


def make_pool(drop_policy="drop_newest", maxsize=4):
    """Build a WorkerPool without spawning actual worker processes."""
    config = PipelineConfig(num_workers=2, drop_policy=drop_policy)
    pool = WorkerPool.__new__(WorkerPool)
    pool._config = config
    pool._work_queue = mp.Queue(maxsize=maxsize)
    pool._shutdown = mp.Event()
    pool._workers = []
    pool._queued = 0
    return pool


def test_submit_accepted_when_space_available():
    pool = make_pool()
    accepted = pool.submit(make_request())
    assert accepted is True
    assert pool.queue_depth() == 1


def test_submit_dropped_when_full_drop_newest():
    pool = make_pool(drop_policy="drop_newest", maxsize=1)
    pool.submit(make_request("seg-1"))       # fills queue
    dropped = pool.submit(make_request("seg-2"))  # should drop
    assert dropped is False
    assert pool.queue_depth() == 1


def test_submit_evicts_oldest_on_drop_oldest():
    pool = make_pool(drop_policy="drop_oldest", maxsize=1)
    pool.submit(make_request("old"))
    accepted = pool.submit(make_request("new"))
    assert accepted is True
    assert pool.queue_depth() == 1


def test_submit_stamps_enqueue_ts():
    pool = make_pool()
    req = make_request()
    assert req.enqueue_ts == 0.0
    pool.submit(req)
    item = pool._work_queue.get(timeout=1.0)
    assert item.enqueue_ts > 0.0
