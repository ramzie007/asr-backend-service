from config import CaptioningSLA
from pipeline.metrics import MetricsCollector
from pipeline.contracts import WorkerResult


def make_result(is_final=True, start_ts=0.0, end_ts=1.0,
                inference_start=0.1, inference_end=0.5,
                emit_ts=0.6, enqueue_ts=0.05):
    return WorkerResult(
        segment_id="seg-1", stream_id="s", transcript="hi",
        start_ts=start_ts, end_ts=end_ts, enqueue_ts=enqueue_ts,
        inference_start_ts=inference_start, inference_end_ts=inference_end,
        emit_ts=emit_ts, is_final=is_final,
    )


def test_records_final_results():
    mc = MetricsCollector()
    mc.record(make_result(is_final=True))
    assert mc.count() == 1


def test_ignores_partial_results():
    mc = MetricsCollector()
    mc.record(make_result(is_final=False))
    assert mc.count() == 0


def test_summary_has_all_keys():
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    s = mc.summary()
    for key in ("ftl_p50", "ftl_p95", "ftl_p99", "fl_p95", "rtf_p95", "count"):
        assert key in s


def test_ftl_calculation():
    # emit_ts=0.6, start_ts=0.0 → FTL = 600ms
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    assert abs(mc.summary()["ftl_p50"] - 600.0) < 1.0


def test_empty_summary_returns_empty_dict():
    assert MetricsCollector().summary() == {}


def test_print_table_outputs_headers(capsys):
    mc = MetricsCollector()
    for _ in range(5):
        mc.record(make_result())
    mc.print_table(streams=2, drop_policy="drop_newest", num_workers=2, drops=0)
    out = capsys.readouterr().out
    assert "FTL" in out or "p50" in out


def test_summary_includes_queue_wait_and_inference_percentiles():
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    s = mc.summary()
    assert "queue_wait_p95" in s, "summary() must include queue_wait_p95"
    assert "pure_inference_p95" in s, "summary() must include pure_inference_p95"


def test_snapshot_matches_summary():
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    assert mc.snapshot() == mc.summary()


def test_print_table_shows_queue_wait_and_inference_cols(capsys):
    mc = MetricsCollector()
    for _ in range(5):
        mc.record(make_result())
    mc.print_table(streams=2, drop_policy="drop_newest", num_workers=2, drops=0)
    out = capsys.readouterr().out
    assert "queue_wait" in out, "print_table must show queue_wait in decomposition"
    assert "inference" in out, "print_table must show inference in decomposition"
    assert "emit_overhead" in out, "print_table must show emit_overhead in decomposition"
    assert "Latency Decomposition" in out, "print_table must show decomposition section"


def test_print_live_snapshot_outputs_header(capsys):
    mc = MetricsCollector()
    for _ in range(5):
        mc.record(make_result())
    mc.print_live_snapshot(streams=2, drop_policy="drop_newest", num_workers=2, drops=0)
    out = capsys.readouterr().out
    assert "LIVE SNAPSHOT" in out
    assert "Queue p95" in out


# --- CaptioningSLA evaluation tests ---


def test_evaluate_sla_all_pass():
    # FTL p95 = 600ms < 800ms target, FL p95 = -400ms < 1200ms, RTF p95 = 0.4 < 1.0
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    sla = CaptioningSLA()
    checks = mc.evaluate_sla(sla)
    assert len(checks) >= 3
    assert all(c["pass"] for c in checks), f"Expected all SLA checks to pass: {checks}"


def test_evaluate_sla_ftl_breach():
    # emit_ts=2.0, start_ts=0.0 → FTL = 2000ms > 800ms target
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result(emit_ts=2.0))
    sla = CaptioningSLA(ftl_p95_ms=800.0)
    checks = mc.evaluate_sla(sla)
    ftl_check = next(c for c in checks if c["metric"] == "FTL p95")
    assert ftl_check["pass"] is False


def test_evaluate_sla_fl_breach():
    # emit_ts=3.0, end_ts=0.5 → FL = 2500ms > 1200ms target
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result(end_ts=0.5, emit_ts=3.0))
    sla = CaptioningSLA(fl_p95_ms=1200.0)
    checks = mc.evaluate_sla(sla)
    fl_check = next(c for c in checks if c["metric"] == "FL p95")
    assert fl_check["pass"] is False


def test_evaluate_sla_empty_returns_empty():
    mc = MetricsCollector()
    assert mc.evaluate_sla(CaptioningSLA()) == []


def test_evaluate_sla_load_shedding_check():
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    sla = CaptioningSLA(max_queue_growth=True)
    checks = mc.evaluate_sla(sla, drops=5)
    shedding = next(c for c in checks if c["metric"] == "Load shedding")
    assert shedding["pass"] is True
    assert "5 drops" in shedding["actual"]


def test_print_captioning_report_shows_sla(capsys):
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    sla = CaptioningSLA()
    mc.print_captioning_report(
        streams=4, drop_policy="drop_newest", num_workers=2, drops=0, sla=sla,
    )
    out = capsys.readouterr().out
    assert "Live Captioning" in out
    assert "SLA Compliance" in out
    assert "PASS" in out


def test_print_captioning_report_shows_breach(capsys):
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result(emit_ts=2.0))  # FTL = 2000ms
    sla = CaptioningSLA(ftl_p95_ms=800.0)
    mc.print_captioning_report(
        streams=4, drop_policy="drop_newest", num_workers=2, drops=0, sla=sla,
    )
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "SLA BREACH" in out


def test_print_captioning_report_without_sla(capsys):
    mc = MetricsCollector()
    for _ in range(10):
        mc.record(make_result())
    mc.print_captioning_report(
        streams=2, drop_policy="drop_newest", num_workers=2, drops=0,
    )
    out = capsys.readouterr().out
    assert "Live Captioning" in out
    assert "Creators" in out
    assert "SLA Compliance" not in out


def test_print_captioning_report_empty(capsys):
    mc = MetricsCollector()
    mc.print_captioning_report(
        streams=1, drop_policy="drop_newest", num_workers=1,
    )
    out = capsys.readouterr().out
    assert "No segments" in out
