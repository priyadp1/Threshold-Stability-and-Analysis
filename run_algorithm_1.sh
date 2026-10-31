#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

log="$LOG_DIR/algorithm_1_$STAMP.log"
echo "==> algorithm_1.py (log: $log)"
start=$SECONDS
"$PYTHON" -u algorithm_1.py 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -eq 0 ]; then
    echo "==> algorithm_1.py finished in $((SECONDS - start))s"
else
    echo "==> algorithm_1.py FAILED after $((SECONDS - start))s, see $log" >&2
fi

exit "$status"
