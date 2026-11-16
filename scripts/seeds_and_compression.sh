#!/usr/bin/env bash
# Two things, in one go:
#   1. repeat the best setting (the "long" run) with three more seeds, to see
#      how much the numbers move on their own;
#   2. shrink that model and measure what the shrinking costs.
#
# Run from the project root:  bash scripts/seeds_and_compression.sh
# About an hour on a free Colab CPU.
set -euo pipefail

python -m lhcad.download
[ -f data/processed/background_test.npy ] || python -m lhcad.prepare

for seed in 1 2 3; do
    name="long-seed$seed"
    [ -f "results/$name/metrics.json" ] && continue
    python -m lhcad.train --name "$name" --epochs 100 --patience 10 --seed "$seed"
    python -m lhcad.evaluate --run "runs/$name"
    python -m lhcad.export --run "runs/$name"
    python -m lhcad.figures "results/$name"
done

# the seed-0 run of the same setting is called "long"
if [ ! -f results/long/metrics.json ]; then
    python -m lhcad.train --name long --epochs 100 --patience 10 --seed 0
    python -m lhcad.evaluate --run runs/long
    python -m lhcad.export --run runs/long
fi

python scripts/seed_summary.py results/long results/long-seed1 results/long-seed2 results/long-seed3

python -m lhcad.compress --run runs/long
cp runs/long/compression.json results/long/compression.json

python -m lhcad.compare
