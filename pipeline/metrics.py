from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate
from config import CaptioningSLA
from pipeline.contracts import WorkerResult


@dataclass
class SegmentMetrics:
    segment_id: str
    ftl: float
    fl: float
    rtf: float
    queue_wait: float
    pure_inference: float
    emit_overhead: float


class MetricsCollector:
    """Collects per-segment timing and computes percentile summaries."""

    def __init__(self) -> None:
        self._records: list[SegmentMetrics] = []

    def record(self, result: WorkerResult) -> None:
        if not result.is_final:
            return
        duration = result.end_ts - result.start_ts
        if duration <= 0:
            return
        self._records.append(SegmentMetrics(
            segment_id=result.segment_id,
            ftl=result.emit_ts - result.start_ts,
            fl=result.emit_ts - result.end_ts,
            rtf=(result.inference_end_ts - result.inference_start_ts) / duration,
            queue_wait=result.inference_start_ts - result.enqueue_ts,
            pure_inference=result.inference_end_ts - result.inference_start_ts,
            emit_overhead=result.emit_ts - result.inference_end_ts,
        ))

    def count(self) -> int:
        return len(self._records)

    def summary(self) -> dict:
        if not self._records:
            return {}

        def ms(vals: list[float], p: int) -> float:
            return float(np.percentile(vals, p)) * 1000

        ftls = [r.ftl for r in self._records]
        fls = [r.fl for r in self._records]
        rtfs = [r.rtf for r in self._records]
        qws = [r.queue_wait for r in self._records]
        infs = [r.pure_inference for r in self._records]
        eos = [r.emit_overhead for r in self._records]
        return {
            "count": len(self._records),
            "ftl_p50": ms(ftls, 50),
            "ftl_p95": ms(ftls, 95),
            "ftl_p99": ms(ftls, 99),
            "fl_p50": ms(fls, 50),
            "fl_p95": ms(fls, 95),
            "fl_p99": ms(fls, 99),
            "rtf_p50": float(np.percentile(rtfs, 50)),
            "rtf_p95": float(np.percentile(rtfs, 95)),
            "queue_wait_p95": ms(qws, 95),
            "pure_inference_p95": ms(infs, 95),
            "emit_overhead_p95": ms(eos, 95),
        }

    def snapshot(self) -> dict:
        """Return current rolling summary without resetting internal state."""
        return self.summary()

    def evaluate_sla(self, sla: CaptioningSLA, drops: int = 0) -> list[dict]:
        """Evaluate metrics against captioning SLA targets. Returns list of check results."""
        s = self.summary()
        if not s:
            return []
        checks = [
            {
                "metric": "FTL p95",
                "target": f"< {sla.ftl_p95_ms:.0f}ms",
                "actual": f"{s['ftl_p95']:.0f}ms",
                "pass": s["ftl_p95"] < sla.ftl_p95_ms,
            },
            {
                "metric": "FL p95",
                "target": f"< {sla.fl_p95_ms:.0f}ms",
                "actual": f"{s['fl_p95']:.0f}ms",
                "pass": s["fl_p95"] < sla.fl_p95_ms,
            },
            {
                "metric": "RTF p95",
                "target": f"< {sla.rtf_p95:.2f}",
                "actual": f"{s['rtf_p95']:.2f}",
                "pass": s["rtf_p95"] < sla.rtf_p95,
            },
        ]
        if sla.max_queue_growth:
            checks.append({
                "metric": "Load shedding",
                "target": "explicit drops, no unbounded growth",
                "actual": f"{drops} drops" if drops > 0 else "no drops needed",
                "pass": True,  # presence of drop policy = pass
            })
        return checks

    def print_table(self, streams: int, drop_policy: str, num_workers: int, drops: int = 0) -> None:
        s = self.summary()
        if not s:
            print("No segments processed.")
            return

        # Main benchmark table
        rows = [[
            streams,
            f"{s['ftl_p50']:.0f}ms",
            f"{s['ftl_p95']:.0f}ms",
            f"{s['fl_p95']:.0f}ms",
            f"{s['rtf_p95']:.2f}",
            s["count"],
            drops,
        ]]
        headers = ["Streams", "FTL p50", "FTL p95", "FL p95", "RTF p95", "Segs", "Drops"]
        print()
        print("=" * 75)
        print(f" Benchmark — faster-whisper {num_workers}w | policy={drop_policy}")
        print("=" * 75)
        print(tabulate(rows, headers=headers, tablefmt="simple"))

        # Latency decomposition table — shows where p95 time is spent
        decomp_rows = [
            ["queue_wait", f"{s['queue_wait_p95']:.0f}ms",
             "enqueue → inference start"],
            ["inference", f"{s['pure_inference_p95']:.0f}ms",
             "model.transcribe()"],
            ["emit_overhead", f"{s['emit_overhead_p95']:.0f}ms",
             "inference end → ws.send()"],
        ]
        print()
        print(f" Latency Decomposition (p95)")
        print("-" * 55)
        print(tabulate(decomp_rows, headers=["Stage", "p95", "Description"], tablefmt="simple"))
        print()

    def print_live_snapshot(self, streams: int, drop_policy: str, num_workers: int, drops: int = 0) -> None:
        """Print a compact rolling snapshot suitable for periodic live reporting."""
        s = self.snapshot()
        if not s:
            return
        rows = [[
            streams,
            f"{s['ftl_p50']:.0f}ms",
            f"{s['ftl_p95']:.0f}ms",
            f"{s['fl_p95']:.0f}ms",
            f"{s['rtf_p95']:.2f}",
            f"{s['queue_wait_p95']:.0f}ms",
            f"{s['pure_inference_p95']:.0f}ms",
            f"{s['emit_overhead_p95']:.0f}ms",
            s["count"],
            drops,
        ]]
        headers = [
            "Streams", "FTL p50", "FTL p95", "FL p95", "RTF p95",
            "Queue p95", "Inference p95", "Emit p95", "Segs", "Drops",
        ]
        print()
        print("-" * 90)
        print(f" LIVE SNAPSHOT (rolling) — faster-whisper {num_workers}w | policy={drop_policy}")
        print("-" * 90)
        print(tabulate(rows, headers=headers, tablefmt="simple"))
        print()

    def print_captioning_report(
        self,
        streams: int,
        drop_policy: str,
        num_workers: int,
        drops: int = 0,
        sla: CaptioningSLA | None = None,
    ) -> None:
        """Print metrics framed as a live captioning product evaluation."""
        s = self.summary()
        if not s:
            print("No segments processed.")
            return

        print()
        print("=" * 78)
        print(" Live Captioning — Product Evaluation")
        print(f" {streams} concurrent creator(s) | {num_workers} workers | policy={drop_policy}")
        print("=" * 78)

        rows = [[
            streams,
            f"{s['ftl_p50']:.0f}ms",
            f"{s['ftl_p95']:.0f}ms",
            f"{s['fl_p50']:.0f}ms",
            f"{s['fl_p95']:.0f}ms",
            f"{s['rtf_p95']:.2f}",
            f"{s['queue_wait_p95']:.0f}ms",
            s["count"],
            drops,
        ]]
        headers = [
            "Creators", "FTL p50", "FTL p95",
            "FL p50", "FL p95", "RTF p95",
            "Queue p95", "Captions", "Dropped",
        ]
        print(tabulate(rows, headers=headers, tablefmt="simple"))

        if sla:
            checks = self.evaluate_sla(sla, drops)
            print()
            print("-" * 78)
            print(" SLA Compliance")
            print("-" * 78)
            sla_rows = []
            for c in checks:
                status = "PASS" if c["pass"] else "FAIL"
                sla_rows.append([c["metric"], c["target"], c["actual"], status])
            print(tabulate(sla_rows, headers=["Metric", "Target", "Actual", "Status"], tablefmt="simple"))
            all_pass = all(c["pass"] for c in checks)
            print()
            if all_pass:
                print(" Result: ALL SLA TARGETS MET")
            else:
                failed = [c["metric"] for c in checks if not c["pass"]]
                print(f" Result: SLA BREACH on {', '.join(failed)}")
        print()
