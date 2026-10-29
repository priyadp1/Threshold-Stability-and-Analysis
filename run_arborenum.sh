set -u

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

status=0
for script in run_standard_arborenum.py run_anytime_arborenum.py; do
    log="$LOG_DIR/${script%.py}_$STAMP.log"
    echo "==> $script (log: $log)"
    start=$SECONDS
    if "$PYTHON" -u "$script" 2>&1 | tee "$log"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
        echo "==> $script finished in $((SECONDS - start))s"
    else
        echo "==> $script FAILED after $((SECONDS - start))s, see $log" >&2
        status=1
    fi
done

exit $status
