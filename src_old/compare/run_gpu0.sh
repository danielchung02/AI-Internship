#!/usr/bin/env bash
# GPU 0: vector-observation comparison runs, then their 100-episode exports.
set -euo pipefail

python -m src.vec_discrete.vec_discrete_train \
  --total-steps 0 \
  --epsilon-decay-steps 1000000 \
  --action-repeat 2 \
  --evaluation-interval 50000 \
  --evaluation-episodes 10 \
  --checkpoint-interval 50000 \
  --stop-mean-reward 900 \
  --output-dir src/compare/vec_discrete

python -m src.vec_cont.vec_cont_train \
  --total-steps 0 \
  --action-repeat 2 \
  --evaluation-interval 50000 \
  --evaluation-episodes 10 \
  --checkpoint-interval 50000 \
  --stop-mean-reward 900 \
  --output-dir src/compare/vec_cont

python -m src.compare.evaluate_checkpoints \
  --method vec_discrete \
  --checkpoint-dir src/compare/vec_discrete/checkpoints \
  --output-csv src/compare/vec_discrete/frame_score.csv \
  --episodes 100 --seed-start 10000

python -m src.compare.evaluate_checkpoints \
  --method vec_cont \
  --checkpoint-dir src/compare/vec_cont/checkpoints \
  --output-csv src/compare/vec_cont/frame_score.csv \
  --episodes 100 --seed-start 10000
