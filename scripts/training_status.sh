#!/bin/bash
# Usage: bash scripts/training_status.sh [logfile]
LOG="${1:-training_arch_v3.log}"

if [ ! -f "$LOG" ]; then
    echo "Log file not found: $LOG"
    exit 1
fi

# Parse SB3 table format: |    key    | value    |
extract() { grep "$1" "$LOG" | tail -1 | awk -F'|' '{gsub(/[ \t]+/,"",$3); print $3}'; }

STEPS=$(extract "total_timesteps")
FPS=$(extract "fps")

if [ -z "$STEPS" ] || [ -z "$FPS" ] || [ "$FPS" = "0" ]; then
    echo "Could not parse training metrics from $LOG"
    exit 1
fi

PCT=$(awk "BEGIN {printf \"%.1f\", $STEPS * 100 / 25000000}")
ETA_HRS=$(awk "BEGIN {printf \"%.1f\", (25000000 - $STEPS) / $FPS / 3600}")

echo "=== Training Status ==="
echo "Steps: ${STEPS} / 25,000,000 (${PCT}%)"
echo "FPS: ${FPS}"
echo "ETA: ~${ETA_HRS} hours"
echo ""

# Eval summary
HEURISTIC_EVALS=$(grep "Eval.*Heuristic" "$LOG")
SELFPLAY_EVALS=$(grep "Eval.*ckpt" "$LOG")
N_HEUR=$(echo "$HEURISTIC_EVALS" | grep -c "Heuristic")
N_SELF=$(echo "$SELFPLAY_EVALS" | grep -c "ckpt")

echo "=== Evals: ${N_HEUR} heuristic, ${N_SELF} self-play ==="
echo ""

# Last 10 heuristic evals
echo "--- Last 10 vs Heuristic ---"
echo "$HEURISTIC_EVALS" | tail -10 | sed 's/\[Eval\] vs Heuristic: //'
echo ""

# Heuristic win rate trend (last 20 in groups of 5)
echo "--- Heuristic Win Rate (5-eval averages) ---"
echo "$HEURISTIC_EVALS" | grep -oP '\d+%' | tail -20 | sed 's/%//' | awk '
{
    sum += $1; n++
    if (n % 5 == 0) {
        printf "  Evals %d-%d: %.0f%%\n", NR-4, NR, sum/5
        sum = 0
    }
}
END {
    if (n % 5 != 0) printf "  Evals %d-%d: %.0f%%\n", NR-(n%5)+1, NR, sum/(n%5)
}'
echo ""

# Peak and recent
PEAK=$(echo "$HEURISTIC_EVALS" | grep -oP '\d+(?=%)' | sort -n | tail -1)
LAST5=$(echo "$HEURISTIC_EVALS" | grep -oP '\d+(?=%)' | tail -5 | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
echo "Peak: ${PEAK}%  |  Last 5 avg: ${LAST5}%"
echo ""

# Self-play evals if any
if [ "$N_SELF" -gt 0 ]; then
    echo "--- Last 5 vs Self-Play ---"
    echo "$SELFPLAY_EVALS" | tail -5 | sed 's/\[Eval\] vs //'
    echo ""
fi

# Key training metrics
echo "--- Latest Training Metrics ---"
for metric in entropy_loss explained_variance approx_kl clip_fraction; do
    val=$(extract "$metric")
    printf "  %-22s %s\n" "$metric" "$val"
done
