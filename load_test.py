#!/usr/bin/env python3
"""
load_test.py — concurrent stream simulator and benchmark reporter.

Usage:
    python load_test.py [--streams 1,2,4] [--duration 10] [--host localhost] [--port 8765] [--real]
    python load_test.py --scenario live-captioning --streams 4 --duration 10

Scenarios:
    default          — generic ASR load test
    live-captioning  — simulates creators recording short-form videos with live captions

Requires server.py to be running first:
    python server.py
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
import websockets
from tabulate import tabulate

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


DEFAULT_REAL_AUDIO_URL = (
    "https://dn721501.ca.archive.org/0/items/art_of_war_librivox/"
    "art_of_war_09-10_sun_tzu.mp3"
)


def generate_synthetic(duration_s: float, sample_rate: int = 16000) -> bytes:
    """White noise — reliably triggers VAD as speech."""
    n = int(duration_s * sample_rate)
    noise = (np.random.rand(n) * 2 - 1) * 16000
    return noise.astype(np.int16).tobytes()


def _extract_window_or_repeat(data: bytes, duration_s: float, sample_rate: int, offset_s: float = 0.0) -> bytes:
    n_needed = int(duration_s * sample_rate) * 2
    start = max(0, int(offset_s * sample_rate) * 2)
    if len(data) == 0:
        return generate_synthetic(duration_s, sample_rate)

    if start + n_needed <= len(data):
        return data[start:start + n_needed]

    # Wrap around if the selected window runs past file end.
    if start >= len(data):
        start = start % len(data)
    tail = data[start:]
    remaining = n_needed - len(tail)
    repeated = (data * ((remaining // len(data)) + 1))[:remaining]
    return tail + repeated


def load_real_audio(
    duration_s: float = 10.0,
    sample_rate: int = 16000,
    url: str | None = None,
    offset_s: float = 0.0,
) -> bytes:
    """Returns real-speech PCM window from a URL (cached decode on first run)."""
    source_url = url or DEFAULT_REAL_AUDIO_URL
    cache_key = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]
    cache = Path(f".cache/real_audio_{cache_key}_{sample_rate}hz.raw")
    cache.parent.mkdir(exist_ok=True)

    if cache.exists():
        data = cache.read_bytes()
        return _extract_window_or_repeat(data, duration_s, sample_rate, offset_s)

    print(f"Downloading real audio sample: {source_url}")
    try:
        resp = requests.get(source_url, timeout=30)
        resp.raise_for_status()
        import subprocess
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
            input=resp.content, capture_output=True, check=True,
        )
        cache.write_bytes(proc.stdout)
        return load_real_audio(duration_s, sample_rate, source_url, offset_s)
    except Exception as e:
        print(f"Real audio failed ({e}), falling back to synthetic.")
        return generate_synthetic(duration_s, sample_rate)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_offsets(value: str) -> list[float]:
    offsets: list[float] = []
    for part in _parse_csv(value):
        try:
            offsets.append(float(part))
        except ValueError:
            continue
    return offsets or [0.0]


def build_real_audio_pool(
    duration_s: float,
    sample_rate: int,
    real_urls: str,
    real_offsets: str,
) -> tuple[list[bytes], list[str]]:
    urls = _parse_csv(real_urls) if real_urls else [DEFAULT_REAL_AUDIO_URL]
    offsets = _parse_offsets(real_offsets)
    audio_pool: list[bytes] = []
    labels: list[str] = []
    for url in urls:
        for offset in offsets:
            clip = load_real_audio(duration_s, sample_rate, url=url, offset_s=offset)
            audio_pool.append(clip)
            labels.append(f"{url} @ {offset:.0f}s")
    return audio_pool, labels


async def run_stream(
    uri: str,
    audio: bytes,
    chunk_size: int = 3200,
    wait_for_final_timeout_s: float = 60.0,
    sample_rate: int = 16000,
    vad_frame_ms: int = 30,
    tail_silence_ms: int = 360,
) -> list[dict]:
    """Stream audio over one WebSocket connection, collect all messages."""
    results = []
    final_received = asyncio.Event()
    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            async def sender():
                for i in range(0, len(audio), chunk_size):
                    await ws.send(audio[i:i + chunk_size])
                    await asyncio.sleep(0.1)  # simulate real-time ingestion

                # Trigger VAD finalization before socket close by sending
                # a short silence tail (hangover + margin).
                frame_bytes = int(sample_rate * (vad_frame_ms / 1000.0)) * 2
                silence_frame = bytes(frame_bytes)
                tail_frames = max(1, tail_silence_ms // vad_frame_ms)
                for _ in range(tail_frames):
                    await ws.send(silence_frame)
                    await asyncio.sleep(vad_frame_ms / 1000.0)

                # Signal stream end so server can flush VAD while socket is still open.
                await ws.send("__end_stream__")

                # VAD flush happens on stream end; wait for final caption before close.
                try:
                    await asyncio.wait_for(final_received.wait(), timeout=wait_for_final_timeout_s)
                except asyncio.TimeoutError:
                    pass
                await ws.close()

            async def receiver():
                try:
                    async for msg in ws:
                        payload = json.loads(msg)
                        results.append(payload)
                        if payload.get("type") == "final":
                            final_received.set()
                except websockets.exceptions.ConnectionClosed:
                    pass

            await asyncio.gather(sender(), receiver())
    except Exception as e:
        print(f"  Stream error: {e}")
    return results


async def run_concurrent(
    uri: str,
    n: int,
    audio_pool: list[bytes],
    assignment: str = "round-robin",
    seed: int = 42,
) -> list[list[dict]]:
    if not audio_pool:
        raise ValueError("audio_pool must not be empty")
    rng = np.random.default_rng(seed)
    tasks = []
    for stream_idx in range(n):
        if assignment == "random":
            clip_idx = int(rng.integers(0, len(audio_pool)))
        else:
            clip_idx = stream_idx % len(audio_pool)
        tasks.append(run_stream(uri, audio_pool[clip_idx]))
    return list(await asyncio.gather(*tasks))


def _print_captioning_narrative(stream_counts: list[int]) -> None:
    """Print the live captioning scenario narrative."""
    print()
    print("=" * 72)
    print(" Live Auto-Captioning — Short-Form Video Platform Simulation")
    print("=" * 72)
    print()
    print("  Scenario: A creator records a video. Speech is transcribed live.")
    print("  Captions must appear in near real-time (< ~800ms).")
    print("  The system must not stall recording, and must degrade gracefully.")
    print()
    print("  Each stream simulates one creator recording a video.")
    print("  FTL measures time from first spoken word to caption appearing.")
    print("  Drop policy ensures stale segments are discarded rather than")
    print("  building latency — real-time recording semantics.")
    print()
    print("-" * 72)
    print(" Live Captioning Requirements")
    print("-" * 72)
    print("  Target FTL p95      < 800ms")
    print("  Target FL p95       < 1200ms")
    print("  Target RTF p95      < 1.0")
    print("  Queue growth        zero unbounded growth")
    print("  Load shedding       explicit (drop_newest)")
    print("-" * 72)
    print()


async def main(args: argparse.Namespace) -> None:
    uri = f"ws://{args.host}:{args.port}"
    stream_counts = [int(s) for s in args.streams.split(",")]
    is_captioning = args.scenario == "live-captioning"

    if is_captioning:
        _print_captioning_narrative(stream_counts)

    print(f"Connecting to {uri}")
    if args.real:
        audio_pool, labels = build_real_audio_pool(
            duration_s=args.duration,
            sample_rate=16000,
            real_urls=args.real_urls,
            real_offsets=args.real_offsets,
        )
        source = f"real ({len(labels)} clip variants, assignment={args.clip_assignment})"
    else:
        audio_pool = [generate_synthetic(args.duration)]
        source = "synthetic"
    if is_captioning:
        print(f"Audio: {source} | {args.duration}s per creator session\n")
    else:
        print(f"Audio: {source} | {args.duration}s per stream\n")

    rows = []
    for n in stream_counts:
        label = "creator(s)" if is_captioning else "stream(s)"
        print(f"Running {n} concurrent {label}...")
        all_results = await run_concurrent(
            uri,
            n,
            audio_pool,
            assignment=args.clip_assignment,
            seed=args.seed,
        )
        finals = sum(1 for r in all_results for m in r if m.get("type") == "final")
        drops = sum(
            1 for r in all_results for m in r
            if m.get("type") == "status" and m.get("event") == "segment_dropped"
        )
        if is_captioning:
            rows.append([n, finals, drops, "see server for SLA evaluation"])
        else:
            rows.append([n, finals, drops, "see server stdout for latency"])

    print()
    if is_captioning:
        col_label = "Creators"
        finals_label = "Captions Rx"
    else:
        col_label = "Streams"
        finals_label = "Finals Rx"

    print("=" * 65)
    title = "Live Captioning Results" if is_captioning else "Load Test Results"
    print(f" {title} — {args.host}:{args.port}")
    print("=" * 65)
    print(tabulate(rows, headers=[col_label, finals_label, "Drops", "Note"], tablefmt="simple"))
    print()
    if is_captioning:
        print("SLA evaluation (FTL/FL p95 vs targets) printed by server on shutdown.")
        print("Stop the server (Ctrl+C) to see the captioning product evaluation.")
    else:
        print("Latency percentiles (FTL/FL/RTF) are printed by the server on shutdown.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech ASR load tester")
    parser.add_argument("--streams", default="1,2,4", help="Comma-separated concurrency levels")
    parser.add_argument("--duration", type=float, default=10.0, help="Audio duration per stream (s)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--real", action="store_true", help="Use real audio (requires ffmpeg)")
    parser.add_argument(
        "--real-urls",
        default="",
        help="Comma-separated real-audio URLs. If omitted, uses one default LibriVox clip.",
    )
    parser.add_argument(
        "--real-offsets",
        default="0",
        help="Comma-separated start offsets in seconds for each URL (e.g. 0,20,40).",
    )
    parser.add_argument(
        "--clip-assignment",
        default="round-robin",
        choices=["round-robin", "random"],
        help="How streams pick clips from real-audio pool.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used when clip-assignment=random")
    parser.add_argument(
        "--scenario", default="default", choices=["default", "live-captioning"],
        help="Test scenario: 'default' or 'live-captioning' (simulates video creators)",
    )
    asyncio.run(main(parser.parse_args()))
