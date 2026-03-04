# Live Auto-Captioning Backend for Short-Form Video

Streaming speech recognition backend built for live captioning under concurrent load — CPU-only, no GPU. The focus is latency and predictable degradation, not transcription accuracy.

→ [Design & Benchmark Write-up](case-study.md)

---

## The Problem

Live captioning has a hard constraint: a 2-second-old caption is useless. Under overload, the system has to drop audio rather than queue it — a lagged system never recovers, a system that drops a segment moves on.

SLA targets:

| Metric | Target |
|--------|--------|
| FL p95 | < 1,200ms |
| RTF p95 | < 1.0 |
| Queue growth | Zero unbounded |

## Architecture

```
WebSocket (audio bytes)
  → AudioIngestion     # normalize into 30ms PCM frames
  → VADSegmenter       # detect utterance boundaries, hangover FSM
  → mp.Queue(N×2)      # admission gate — drop if full
      → Worker 0       # WhisperModel(int8, CPU), beam=1
      → Worker 1
      → ...
  → result_queue
  → caption JSON (WebSocket)
```

Workers are processes, not threads — each loads its own model, no GIL contention. Every segment carries six `time.monotonic()` timestamps, giving per-stage p95 decomposition (queue wait vs. inference) without sampling.

## Benchmark Results

Measured on Apple M2, 8 cores — 4 workers, queue size 8, 4 concurrent streams.

| Metric | Result | Target |
|--------|--------|--------|
| FL p95 | 1,582ms | < 1,200ms |
| RTF p95 | 1.87 | < 1.0 |
| queue_wait p95 | 558ms | — |
| inference p95 | 1,561ms | — |
| Drops | 0 | explicit drops OK |

RTF exceeds 1.0 at 4 streams on this hardware — the model is the bottleneck, not the architecture. On a GPU (A100, float16), RTF drops to ~0.05 and the bottleneck shifts to queue throughput. The [case study](case-study.md) covers the decomposition and scaling path.

## Running

```bash
pip install -r requirements.txt

# self-contained demo
python live_captioning_demo.py --streams 4 --duration 10

# or manually (two terminals)
python server.py --scenario live-captioning
python load_test.py --scenario live-captioning --streams 1,2,4 --duration 10
# Ctrl+C server to print SLA evaluation
```

## Key Config

| Setting | Default | Notes |
|---------|---------|-------|
| `num_workers` | `cpu_count // 2` | one model per process |
| `drop_policy` | `drop_newest` | `drop_oldest` or `block` also available |
| `vad_hangover_ms` | `300` | sweep with `bash ablation.sh` |
| `queue_maxsize` | `num_workers × 2` | admission gate size |

## Ablation

```bash
bash ablation.sh
```

Sweeps `vad_hangover_ms` over 150 / 300 / 450ms. 300ms is the sweet spot — 28% better FL p95 than 150ms, before inference time climbs at 450ms.
