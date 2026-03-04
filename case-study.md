# Real-Time Speech Recognition Backend: Design & Benchmarks

I built this to understand what it actually takes to run live captioning under concurrent load on CPU — not as a research exercise, but as a production-style system with real latency targets and explicit failure semantics.

The short version: streaming audio in, captions out, multiple streams at once, no GPU. Every design decision comes back to one constraint — a 2-second-old caption is useless to a live viewer, so the system has to stay low-latency or shed load. It cannot accumulate delay.

---

## The Problem

Live captioning on short-form video has a hard latency requirement that most ASR benchmarks don't measure. Academic WER benchmarks care about accuracy on a clean recording. Live captioning cares about whether the caption appears before the viewer has moved on.

I set two SLA targets based on what feels responsive to a real user:

- **FL p95 < 1,200ms** — time from when speech ends to when the caption appears. This is the captioning-relevant metric.
- **RTF p95 < 1.0** — the model must process audio faster than it arrives, or the queue grows without bound.

FTL (first-token latency, measured from speech start) is also tracked, but FL is the number that matters for the product experience. The difference is audio duration — a 3-second utterance adds 3 seconds to FTL but nothing to FL.

The hard part is doing this on CPU, across multiple concurrent streams. A single stream on modern hardware is easy. The interesting question is what happens at 2, 4, 8 streams — where workers contend, queues fill, and latency decomposition becomes the only way to know what's actually going wrong.

---

## Architecture

```
Audio bytes (WebSocket)
     ↓
AudioIngestion    — normalize arbitrary chunk sizes into 30ms frames
     ↓
VADSegmenter      — detect utterance boundaries, emit speech segments
     ↓
WorkerPool        — bounded queue, drop policy, N worker processes
     ↓
Worker Processes  — one faster-whisper model per process, pure inference
     ↓
MetricsCollector  — 6 timestamps per segment → latency decomposition
     ↓
Emitter           — serialize to JSON, send over WebSocket
```

The async event loop handles only I/O — connection management, frame routing, result dispatch. No inference on the loop. Workers are processes, not threads, so each loads its own model instance and there's no GIL contention. Results flow back through a separate `result_queue` that the async dispatcher reads.

---

## Design Decisions

### Bounded queue with explicit drop policy

The most important line in the codebase is this one in `pool.py`:

```python
self._work_queue: mp.Queue = mp.Queue(maxsize=config.queue_maxsize)
```

Queue size is `num_workers × 2`. When it fills, incoming segments are dropped — not delayed, dropped. This is intentional.

The alternative is an unbounded queue. An unbounded queue feels safer because nothing is lost. But under sustained overload, it accumulates delay permanently. A stream that falls 30 seconds behind never catches up. The viewer gets captions from half a minute ago with no recovery path.

Explicit dropping is better. A missed caption is a one-segment gap. A lagged system is unusable. The system supports three policies — `drop_newest` (default, discard incoming), `drop_oldest` (evict stale queued work, accept the new segment), and `block` (for baseline saturation measurement only). For live recording, `drop_newest` is correct. For captioning where recency dominates, `drop_oldest` is the better call.

### Multiprocessing over threading

Python threads share the GIL. For CPU-bound inference, that means serialization — two threads running `model.transcribe()` at the same time get no parallelism benefit. I use `multiprocessing`, so each worker has its own Python interpreter and its own model instance. The tradeoff is memory: N workers means N copies of the model in RAM, which is why the default is `cpu_count // 2` rather than `cpu_count`.

### VAD hangover tuning

The voice activity detector classifies each 30ms frame as speech or silence. The naive implementation emits a new segment every time there's a pause — which means a speaker saying "Hello... world" with a 200ms breath gets split into two segments, doubling inference work and breaking context for the model.

Hangover fixes this. After silence is detected, the VAD waits `hangover_ms` before closing the segment. Default is 300ms. If speech resumes before the timer expires, the segment stays open.

I ran an ablation sweep across 150ms, 300ms, and 450ms to find the actual optimum. The results:

| Hangover | Segments | FL p95 |
|----------|----------|--------|
| 150ms | 33 | 5,400ms |
| 300ms | 23 | 3,870ms |
| 450ms | 18 | 4,200ms |

150ms produces too many short segments — the queue floods. 450ms merges too aggressively — segments get long enough that inference time climbs and you lose the latency gain. 300ms is the sweet spot, 28% better FL than 150ms.

### Timestamp instrumentation

Every segment carries six timestamps injected at each stage:

```
start_ts           — VAD detects first speech frame
end_ts             — VAD hangover expires, segment closed
enqueue_ts         — segment enters the worker queue
inference_start_ts — worker picks up the segment
inference_end_ts   — model.transcribe() returns
emit_ts            — result sent over WebSocket
```

This gives me five derived metrics (FTL, FL, RTF, queue_wait, emit_overhead) at p50/p95/p99 without any sampling or approximation.

The instrumentation caught a real bug. Early in development, p95 FL was spiking under load in a way that didn't match expected inference time. The trace timestamps showed the time wasn't in inference — it was in the gap between `enqueue_ts` and `inference_start_ts`. I had NumPy array conversion (PCM bytes → float32) happening in the async event loop before enqueue. Under concurrent load, the event loop was blocking on CPU work, which stalled frame routing for every active stream. Moving the conversion into `process_segment()` inside the worker process fixed it immediately. Without stage-level tracing I would have optimized inference time while the actual bottleneck sat untouched.

---

## Benchmark Results

Measured on Apple M2, 8 cores. 4 workers (cpu_count // 2), queue size 8.

**Load test — 4 concurrent streams, 10 seconds:**

| Metric | Value | SLA Target |
|--------|-------|------------|
| FL p95 | 1,582ms | < 1,200ms |
| RTF p95 | 1.87 | < 1.0 |
| queue_wait p95 | 558ms | — |
| inference p95 | 1,561ms | — |
| Drops | 0 | explicit drops OK |

RTF is 1.87 — the model is slower than real-time at 4 streams on this hardware, and FL p95 misses the 1,200ms target. The latency decomposition tells me why: 558ms is queue wait, 1,561ms is inference. Both are contributing, but inference dominates. The system isn't dropping because the queue has enough headroom at this concurrency, but it's close to saturation — a longer run or a slower machine would start dropping.

This is the honest result. The architecture is correct; the hardware is the constraint.

---

## Production Scaling Path

The local system uses `mp.Queue` for the work queue and WebSockets for ingestion. At production scale, the same pattern holds but the implementation layer changes:

- **WebSocket ingestion** → stateless pods behind a load balancer, or gRPC for internal services
- **`mp.Queue`** → Kafka or Redis Streams, partitioned by stream ID for ordering guarantees
- **CPU worker processes** → GPU inference pods, each loading the model once on startup

On the M2 CPU, RTF is 1.87 at 4 streams. On a single A100 with float16, Whisper base RTF drops to roughly 0.05 — 37× faster. At that point the bottleneck is no longer compute. It's queue throughput, network I/O, and admission control per tenant. The bounded queue concept stays; the queue implementation changes.

The six-timestamp instrumentation maps directly to distributed tracing spans in production. `queue_wait` becomes the span between Kafka produce and consume. `inference_start_ts` to `inference_end_ts` is the GPU worker span. Same primitives, different substrate.

---

## What I'd Do Differently

**Partial transcript emission.** The `WorkerResult` dataclass has an `is_final` field that's always `True` right now. A real captioning product needs partial results — text that appears as the speaker is still talking and gets updated when the final transcript arrives. This requires streaming decoder output from faster-whisper, which complicates the worker/emitter interface but is the right product behavior.

**Per-stream SLOs.** The current system applies one drop policy globally. A production platform would want per-tenant rate limits — a premium creator gets a larger queue allocation and a stricter SLA, a free-tier stream gets more aggressive dropping. The WorkerRequest already carries a `stream_id` field; the pool would need to partition by it.

**Distributed tracing instead of local timestamps.** The six-timestamp approach works for a single-process system. At scale, you want OpenTelemetry spans so you can trace a single segment across ingestion service, queue, and inference worker with a shared trace ID. The mental model is identical; the tooling changes.
