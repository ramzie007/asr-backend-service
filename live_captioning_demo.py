#!/usr/bin/env python3
"""
live_captioning_demo.py — End-to-end live captioning demo.

Starts the server in captioning mode, runs the load test simulating
concurrent creators recording short-form videos, then prints SLA evaluation.

Usage:
    python live_captioning_demo.py [--streams 4] [--duration 10]

This is a self-contained demo: no need to start the server separately.
"""
from __future__ import annotations
import argparse
import asyncio
import multiprocessing as mp

from config import CaptioningSLA, PipelineConfig
from load_test import generate_synthetic, load_real_audio, run_concurrent
from server import create_app


async def run_demo(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    sla = CaptioningSLA()
    stream_counts = [int(s) for s in args.streams.split(",")]

    print()
    print("=" * 72)
    print(" Live Auto-Captioning Backend — Short-Form Video Platform Demo")
    print("=" * 72)
    print()
    print("  Imagine a creator recording a video.")
    print("  As they speak, captions must appear quickly.")
    print("  Latency directly impacts perceived quality.")
    print()
    print("  Each stream simulates a creator recording a video.")
    print("  FTL measures time from first spoken word to caption appearing.")
    print("  Drop policy ensures stale segments are discarded rather than")
    print("  building latency — you'd rather lose a segment than delay")
    print("  captions by 2 seconds. That's real product thinking.")
    print()
    print("-" * 72)
    print(" Live Captioning Requirements")
    print("-" * 72)
    print(f"  Target FTL p95      < {sla.ftl_p95_ms:.0f}ms")
    print(f"  Target FL p95       < {sla.fl_p95_ms:.0f}ms")
    print(f"  Target RTF p95      < {sla.rtf_p95:.1f}")
    print(f"  Queue growth        zero unbounded growth")
    print(f"  Load shedding       explicit ({sla.drop_policy})")
    print("-" * 72)
    print()

    audio = load_real_audio(args.duration) if args.real else generate_synthetic(args.duration)
    source = "real speech" if args.real else "synthetic"
    print(f"Audio: {source} | {args.duration}s per creator session")
    print(f"Server: workers={config.num_workers} | model={config.model_size} | policy={config.drop_policy}")
    print()

    async with create_app(config, port=0, captioning_sla=sla) as (_server, port):
        uri = f"ws://localhost:{port}"
        for n in stream_counts:
            print(f"Simulating {n} creator(s) recording simultaneously...")
            all_results = await run_concurrent(uri, n, audio)
            captions = sum(1 for r in all_results for m in r if m.get("type") == "final")
            drops = sum(
                1 for r in all_results for m in r
                if m.get("type") == "status" and m.get("event") == "segment_dropped"
            )
            print(f"  -> {captions} captions delivered, {drops} segments shed")
            print()

    # Captioning report with SLA evaluation is printed automatically by create_app on exit


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Live auto-captioning demo for short-form video platform",
    )
    parser.add_argument("--streams", default="4", help="Comma-separated creator counts (e.g. 1,2,4)")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration per creator (s)")
    parser.add_argument("--real", action="store_true", help="Use real audio (requires ffmpeg)")
    asyncio.run(run_demo(parser.parse_args()))
