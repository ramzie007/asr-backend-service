from __future__ import annotations
import multiprocessing as mp
import queue
import time
from pipeline.contracts import WorkerRequest
from pipeline.worker import worker_process
from config import PipelineConfig


class WorkerPool:
    """Manages N worker processes sharing a bounded work_queue."""

    def __init__(self, config: PipelineConfig, result_queue: mp.Queue):
        self._config = config
        self._result_queue = result_queue
        self._work_queue: mp.Queue = mp.Queue(maxsize=config.queue_maxsize)
        self._shutdown = mp.Event()
        self._workers: list[mp.Process] = []
        self._queued: int = 0
        self._start_workers()

    def _start_workers(self) -> None:
        for i in range(self._config.num_workers):
            self._spawn_worker(i)

    def _spawn_worker(self, worker_id: int) -> None:
        p = mp.Process(
            target=worker_process,
            args=(worker_id, self._config, self._work_queue, self._result_queue, self._shutdown),
            daemon=True,
        )
        p.start()
        self._workers.append(p)

    def submit(self, request: WorkerRequest) -> bool:
        """Submit a segment. Returns True if accepted, False if dropped."""
        request.enqueue_ts = time.monotonic()
        if self._config.drop_policy == "block":
            self._work_queue.put(request)
            self._queued += 1
            return True
        try:
            self._work_queue.put_nowait(request)
            self._queued += 1
            return True
        except queue.Full:
            if self._config.drop_policy == "drop_oldest":
                try:
                    # mp.Queue feeder thread may need a moment — use short timeout
                    self._work_queue.get(timeout=0.1)
                    self._queued -= 1
                except queue.Empty:
                    pass
                try:
                    self._work_queue.put_nowait(request)
                    self._queued += 1
                    return True
                except queue.Full:
                    pass
            return False  # drop_newest: discard incoming

    def queue_depth(self) -> int:
        # track manually
        return self._queued

    def mark_completed(self) -> None:
        if self._queued > 0:
            self._queued -= 1

    def pending(self) -> int:
        return self._queued

    def shutdown(self, timeout: float | None = None, wait_for_drain: bool = True) -> None:
        if timeout is None:
            timeout = self._config.worker_join_timeout_s
        if wait_for_drain:
            deadline = time.monotonic() + self._config.shutdown_timeout_s
            while self._queued > 0 and time.monotonic() < deadline:
                time.sleep(0.01)
        self._shutdown.set()
        for w in self._workers:
            w.join(timeout=timeout)
            if w.is_alive():
                w.kill()
