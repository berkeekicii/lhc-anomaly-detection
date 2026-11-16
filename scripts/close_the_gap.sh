#!/usr/bin/env bash
# Experiments that move the baseline towards the setup of Govorkova et al.
# (arXiv:2108.03986): train until the validation loss stops improving (up to
# 100 epochs, patience 10), ignore empty object slots in the loss, and use
# 8 million background events instead of 4 million.
#
# Run from the project root:  bash scripts/close_the_gap.sh
# Uses a GPU when there is one; on a free Colab GPU it takes about 30 minutes.
set -euo pipefail

python -m lhcad.download

[ -f data/processed/background_test.npy ] || python -m lhcad.prepare
[ -f data/processed_8m/background_test.npy ] || \
    python -m lhcad.prepare --max-events 8000000 --out-dir data/processed_8m

run () {  # name, processed dir, extra training options
    local name=$1 processed=$2; shift 2
    python -m lhcad.train --name "$name" --processed-dir "$processed" "$@"
    python -m lhcad.evaluate --run "runs/$name" --processed-dir "$processed"
    python -m lhcad.export --run "runs/$name" --processed-dir "$processed"
    python -m lhcad.figures "results/$name"
}

LONG=(--epochs 100 --patience 10)
run long                data/processed    "${LONG[@]}"
run long-masked         data/processed    "${LONG[@]}" --masked-loss
run long-masked-8m      data/processed_8m "${LONG[@]}" --masked-loss

python -m lhcad.compare
