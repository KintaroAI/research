#!/bin/bash
# Parse experiment logs and extract val loss at each step
# Usage: ./parse_logs.sh

echo "=== Validation Loss Summary ==="
echo ""

for log in logs/log_*.txt; do
    if [ -f "$log" ]; then
        name=$(basename "$log" .txt | sed 's/log_//')
        echo "--- $name ---"
        grep "^s:" "$log" | grep "tel:" | while read line; do
            step=$(echo "$line" | sed 's/s:\([0-9]*\).*/\1/')
            loss=$(echo "$line" | sed 's/.*tel:\([0-9.]*\).*/\1/')
            printf "Step %5s: %s\n" "$step" "$loss"
        done
        echo ""
    fi
done

echo "=== CSV Format (for easy copy-paste) ==="
echo "config,step,val_loss"
for log in logs/log_*.txt; do
    if [ -f "$log" ]; then
        name=$(basename "$log" .txt | sed 's/log_//')
        grep "^s:" "$log" | grep "tel:" | while read line; do
            step=$(echo "$line" | sed 's/s:\([0-9]*\).*/\1/')
            loss=$(echo "$line" | sed 's/.*tel:\([0-9.]*\).*/\1/')
            echo "$name,$step,$loss"
        done
    fi
done
