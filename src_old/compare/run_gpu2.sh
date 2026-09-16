#!/usr/bin/env bash
# GPU 2: image-observation comparison runs, then their 100-episode exports.
set -euo pipefail

python -m src.img_discrete.img_discrete_train \
  --total-steps 0 \
  --replay-capacity 50000 \
  --warmup-steps 20000 \
  --epsilon-decay-steps 2000000 \
  --action-repeat 2 \
  --evaluation-interval 50000 \
  --evaluation-episodes 10 \
  --checkpoint-interval 50000 \
  --stop-mean-reward 900 \
  --output-dir src/compare/img_discrete

python -m src.img_cont.img_cont_train \
  --total-steps 0 \
  --schedule-steps 5000000 \
  --action-repeat 2 \
  --evaluation-interval 50000 \
  --evaluation-episodes 10 \
  --checkpoint-interval 50000 \
  --stop-mean-reward 900 \
  --output-dir src/compare/img_cont

python -m src.compare.evaluate_checkpoints \
  --method img_discrete \
  --checkpoint-dir src/compare/img_discrete/checkpoints \
  --output-csv src/compare/img_discrete/frame_score.csv \
  --episodes 100 --seed-start 10000

python -m src.compare.evaluate_checkpoints \
  --method img_cont \
  --checkpoint-dir src/compare/img_cont/checkpoints \
  --output-csv src/compare/img_cont/frame_score.csv \
  --episodes 100 --seed-start 10000
