#!/bin/bash
# RGB garden chain: 50k increments up to 2M ticks, warm-starting from previous model
# Each run saves 100 frames (save_every=500) and KNN stats

set -e
cd "$(dirname "$0")"
source venv/bin/activate

EXP=15
TICKS=50000
SAVE_EVERY=500  # 50000/500 = 100 frames per run
KNN_EVERY=5000
TOTAL_TARGET=2000000
INCREMENTS=$((TOTAL_TARGET / TICKS))  # 40 runs

PREV_MODEL=""

for i in $(seq 1 $INCREMENTS); do
    TOTAL=$((i * TICKS))
    TOTAL_K=$((TOTAL / 1000))
    DESC="rgb_garden_chain_${TOTAL_K}k"
    OUTDIR=$(python output_name.py $EXP $DESC)

    echo ""
    echo "=== Run $i/$INCREMENTS: ${TOTAL_K}k total ticks ==="
    echo "  Output: $OUTDIR"

    WARM_FLAG=""
    if [ -n "$PREV_MODEL" ]; then
        WARM_FLAG="--warm-start $PREV_MODEL"
        echo "  Warm start: $PREV_MODEL"
    fi

    python main.py word2vec --preset rgb_80x80_garden \
        --anchor-batches 3 \
        -f $TICKS \
        --knn-track 10 --knn-report-every $KNN_EVERY \
        --save-every $SAVE_EVERY \
        $WARM_FLAG \
        -o "$OUTDIR" 2>&1 | tee /tmp/rgb_garden_chain_${TOTAL_K}k.log | \
        grep -E "KNN @|Training done|eval:|model saved"

    PREV_MODEL="$OUTDIR/model.npy"

    # Check if model was saved
    if [ ! -f "$PREV_MODEL" ]; then
        echo "ERROR: model not found at $PREV_MODEL, aborting chain"
        exit 1
    fi

    echo "  Completed: ${TOTAL_K}k total ticks"
done

echo ""
echo "=== Chain complete: $INCREMENTS runs, ${TOTAL_TARGET} total ticks ==="
