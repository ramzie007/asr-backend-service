#!/usr/bin/env bash
set -e

HOST="${HOST:-localhost}"
PORT="${PORT:-8765}"
DURATION="${DURATION:-10}"
STREAMS="${STREAMS:-1,2,4}"

echo "=============================================="
echo " Speech ASR Benchmark"
echo " Model: faster-whisper base | CPU only"
echo " Streams: $STREAMS | Duration: ${DURATION}s"
echo "=============================================="
echo ""
echo "Ensure server is running in another terminal:"
echo "  python server.py"
echo ""
read -p "Press Enter when server is ready..."

python load_test.py \
    --streams "$STREAMS" \
    --duration "$DURATION" \
    --host "$HOST" \
    --port "$PORT"

echo ""
echo "Stop the server (Ctrl+C) to see latency percentile tables."
