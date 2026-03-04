#!/usr/bin/env bash
# ablation.sh — VAD hangover sweep: produces a clean summary table showing
# FL p95 tradeoff across hangover settings. Used as the "ablation slide" in the demo.
#
# Low  hangover_ms → shorter segments, more false splits, lower FL
# High hangover_ms → longer segments, fewer false splits, higher FL
#
# Usage: bash ablation.sh [duration_s] [streams] [repeats]
set -e

DURATION="${1:-8}"
STREAMS="${2:-1}"
REPEATS="${3:-3}"
PORT_BASE=8800
RESULTS_FILE="/tmp/ablation_results_$$.txt"

DEFAULT_REAL_AUDIO_URLS="https://dn721503.ca.archive.org/0/items/uponastone_2602.poem_librivox/uponastone_wordsworth_aps_64kb.mp3,https://dn710903.ca.archive.org/0/items/philosophycivilizationmiddleages_2602_librivox/philosophycivilization_24_dewulf_64kb.mp3,https://ia903405.us.archive.org/15/items/frost_to-night_1710.poem_librivox/frosttonight_thomas_bk_64kb.mp3,https://dn720705.ca.archive.org/0/items/theemptybowl_2603.poem_librivox/emptybowl_wilcox_bwc_64kb.mp3,https://dn720705.ca.archive.org/0/items/theemptybowl_2603.poem_librivox/emptybowl_wilcox_slm_64kb.mp3"
REAL_AUDIO_URLS="${REAL_AUDIO_URLS:-$DEFAULT_REAL_AUDIO_URLS}"
REAL_AUDIO_OFFSETS="${REAL_AUDIO_OFFSETS:-0,20,40}"
CLIP_ASSIGNMENT="${CLIP_ASSIGNMENT:-round-robin}"
CLIP_SEED="${CLIP_SEED:-42}"
SERVER_PID=""

if [ -n "${PYTHON_BIN:-}" ]; then
    PY="${PYTHON_BIN}"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    PY="python3"
fi

format_ms() {
    if [ "$1" = "N/A" ] || [ -z "$1" ]; then
        printf "N/A"
    else
        printf "%sms" "$1"
    fi
}

compute_median_numeric_file() {
    local file="$1"
    local sorted_file="/tmp/ablation_sorted_$$.txt"
    grep -E '^[0-9]+([.][0-9]+)?$' "$file" | sort -n > "$sorted_file" || true
    local count
    count=$(wc -l < "$sorted_file" | tr -d ' ')
    if [ "$count" -eq 0 ]; then
        rm -f "$sorted_file"
        echo "N/A"
        return
    fi
    if [ $((count % 2)) -eq 1 ]; then
        local mid=$((count / 2 + 1))
        sed -n "${mid}p" "$sorted_file"
    else
        local mid1=$((count / 2))
        local mid2=$((mid1 + 1))
        local v1 v2
        v1=$(sed -n "${mid1}p" "$sorted_file")
        v2=$(sed -n "${mid2}p" "$sorted_file")
        awk -v a="$v1" -v b="$v2" 'BEGIN { printf "%.0f\n", (a + b) / 2 }'
    fi
    rm -f "$sorted_file"
}

stop_server_process() {
    if [ -z "${SERVER_PID:-}" ]; then
        return
    fi
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        # Stop child processes first (if any), then parent.
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 0.3
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

cleanup_on_exit() {
    stop_server_process
}

trap cleanup_on_exit EXIT INT TERM

# Pre-cleanup before starting this run.
stop_server_process

wait_for_server_port() {
    local port="$1"
    local timeout_s="${2:-45}"
    local log_file="$3"
    local waited=0

    while [ "$waited" -lt "$timeout_s" ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  server exited before startup completed (port $port)."
            if [ -f "$log_file" ]; then
                echo "  ---- server startup log ----"
                cat "$log_file"
                echo "  ----------------------------"
            fi
            return 1
        fi

        if "$PY" -c "import socket; s=socket.socket(); s.settimeout(0.3); rc=s.connect_ex(('127.0.0.1', $port)); s.close(); raise SystemExit(0 if rc == 0 else 1)" >/dev/null 2>&1; then
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    echo "  server did not become ready within ${timeout_s}s (port $port)."
    if [ -f "$log_file" ]; then
        echo "  ---- server startup log ----"
        cat "$log_file"
        echo "  ----------------------------"
    fi
    return 1
}

echo "============================================================"
echo " VAD Hangover Ablation — faster-whisper base | CPU | 4 workers"
echo " Sweeping: vad_hangover_ms = 150, 300, 450"
echo " Duration: ${DURATION}s per run | Streams: ${STREAMS} | Repeats: ${REPEATS}"
echo " Real clips: $(echo "$REAL_AUDIO_URLS" | awk -F',' '{print NF}') URLs | offsets=${REAL_AUDIO_OFFSETS}"
echo "============================================================"
echo ""

# Header for summary collection
> "$RESULTS_FILE"

for HANGOVER in 150 300 450; do
    FTL_FILE="/tmp/ablation_ftl_${HANGOVER}_$$.txt"
    FL_FILE="/tmp/ablation_fl_${HANGOVER}_$$.txt"
    SEGS_FILE="/tmp/ablation_segs_${HANGOVER}_$$.txt"
    > "$FTL_FILE"
    > "$FL_FILE"
    > "$SEGS_FILE"

    for RUN in $(seq 1 "$REPEATS"); do
        PORT=$((PORT_BASE + HANGOVER + RUN - 1))
        SERVER_LOG="/tmp/ablation_server_${HANGOVER}_${RUN}_$$.log"
        echo "--- vad_hangover_ms=$HANGOVER (run ${RUN}/${REPEATS}, port $PORT) ---"

        # Start server with patched hangover, capture stdout for FL p95 extraction
        VAD_HANGOVER_MS=$HANGOVER "$PY" -c "
import asyncio, logging, multiprocessing as mp, os, signal
from config import PipelineConfig
from server import create_app
logging.getLogger('websockets').setLevel(logging.CRITICAL)

async def run():
    hangover = int(os.environ.get('VAD_HANGOVER_MS', 300))
    config = PipelineConfig(vad_hangover_ms=hangover, num_workers=4)
    async with create_app(config=config, port=$PORT):
        try:
            await asyncio.sleep($DURATION + 15)
        except KeyboardInterrupt:
            pass

def _on_signal(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, _on_signal)

mp.set_start_method('spawn', force=True)
try:
    asyncio.run(run())
except KeyboardInterrupt:
    pass
" > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!

    if ! wait_for_server_port "$PORT" 45 "$SERVER_LOG"; then
        stop_server_process
        echo "  run ${RUN}/${REPEATS}: hangover=${HANGOVER}ms → FTL p95=N/A  FL p95=N/A  segments=N/A"
        rm -f "$SERVER_LOG"
        echo ""
        continue
    fi

        echo "  Running load test..."
        "$PY" load_test.py \
            --real \
            --streams "$STREAMS" \
            --duration "$DURATION" \
            --real-urls "$REAL_AUDIO_URLS" \
            --real-offsets "$REAL_AUDIO_OFFSETS" \
            --clip-assignment "$CLIP_ASSIGNMENT" \
            --seed "$CLIP_SEED" \
            --port "$PORT" 2>/dev/null || true

        # Give server a short grace window to finish naturally, then trigger graceful shutdown.
        sleep 2
        for _ in $(seq 1 5); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        stop_server_process

        # Extract FTL p95 / FL p95 / Segs from benchmark table (portable: no grep -P).
        METRICS_LINE=$(awk '
    BEGIN { ftl="N/A"; fl="N/A"; segs="N/A"; in_bench=0 }
    /Benchmark — faster-whisper/ { in_bench=1; next }
    in_bench && $0 ~ /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+ms[[:space:]]+[0-9]+ms[[:space:]]+[0-9]+ms[[:space:]]+[0-9.]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]*$/ {
        ftl=$3
        fl=$4
        segs=$6
        gsub(/ms/, "", ftl)
        gsub(/ms/, "", fl)
    }
    END { print ftl, fl, segs }
' "$SERVER_LOG")
        read -r FTL_P95 FL_P95 SEGS <<< "$METRICS_LINE"

        case "$FTL_P95" in
            ""|"N/A") ;;
            *) echo "$FTL_P95" >> "$FTL_FILE" ;;
        esac
        case "$FL_P95" in
            ""|"N/A") ;;
            *) echo "$FL_P95" >> "$FL_FILE" ;;
        esac
        case "$SEGS" in
            ""|"N/A") ;;
            *) echo "$SEGS" >> "$SEGS_FILE" ;;
        esac

        echo "  run ${RUN}/${REPEATS}: hangover=${HANGOVER}ms → FTL p95=$(format_ms "$FTL_P95")  FL p95=$(format_ms "$FL_P95")  segments=${SEGS}"

        cat "$SERVER_LOG"
        rm -f "$SERVER_LOG"
        echo ""
    done

    FTL_MEDIAN=$(compute_median_numeric_file "$FTL_FILE")
    FL_MEDIAN=$(compute_median_numeric_file "$FL_FILE")
    SEGS_MEDIAN=$(compute_median_numeric_file "$SEGS_FILE")

    echo "  median(${REPEATS} runs): hangover=${HANGOVER}ms → FTL p95=$(format_ms "$FTL_MEDIAN")  FL p95=$(format_ms "$FL_MEDIAN")  segments=${SEGS_MEDIAN}"
    echo "${HANGOVER} ${FTL_MEDIAN} ${FL_MEDIAN} ${SEGS_MEDIAN}" >> "$RESULTS_FILE"
    rm -f "$FTL_FILE" "$FL_FILE" "$SEGS_FILE"
done

# Print clean summary table
echo ""
echo "============================================================"
echo " Ablation Summary: VAD Hangover vs. Latency (p95 median, ${REPEATS} runs)"
echo "============================================================"
printf "  %-14s  %-10s  %-10s  %-8s\n" "hangover_ms" "FTL p95" "FL p95" "Segments"
printf "  %-14s  %-10s  %-10s  %-8s\n" "-----------" "-------" "------" "--------"
while read -r HANGOVER FTL FL SEGS; do
    printf "  %-14s  %-10s  %-10s  %-8s\n" "${HANGOVER}ms" "$(format_ms "$FTL")" "$(format_ms "$FL")" "$SEGS"
done < "$RESULTS_FILE"
echo ""
echo " Higher hangover → higher FL p95 (more latency, fewer false splits)"
echo " Lower  hangover → lower  FL p95 (less latency, more false splits)"
echo "============================================================"

rm -f "$RESULTS_FILE"
