#!/bin/bash
# Retrain the subgoal/monolith pair on the v2 data.
#   MODE=fsdp8   both arms one after the other on all 8 GPUs (FSDP 8, batch 128, 4000 steps = 512k samples, about 2x the
#                samples at which the v1 subgoal arm plateaued; checkpoints every 250 steps)
#   MODE=single  both arms in parallel, one GPU each (batch 16, 15000 steps, checkpoints every 1000), as the v1 runs
#   ARMS="subgoal monolith" MODE=fsdp8 scripts/train_v2_avalon.sh
set -uo pipefail
WORK=${WORK:-$HOME/vla-memory}; MODE=${MODE:-fsdp8}; ARMS=${ARMS:-"subgoal monolith"}; EXP=${EXP:-v2_$(date +%Y%m%d)}
cd "$WORK"
if [ "$MODE" = fsdp8 ]; then
  for arm in $ARMS; do
    echo "[train] $(date -Is) fsdp8 $arm"
    GPU=0,1,2,3,4,5,6,7 FSDP=8 CONFIG=pi05_red_blue_v2_$arm EXP=${EXP}_fsdp8 LOG_INTERVAL=10 \
      EXTRA_ARGS="${STEPS_ARGS:---num-train-steps 4000 --batch-size 128 --save-interval 250 --keep-period 250}" scripts/train_avalon.sh
    echo "[train] $(date -Is) $arm rc=$?"
  done
else
  g=0
  for arm in $ARMS; do
    echo "[train] $(date -Is) single-GPU $arm on GPU $g"
    GPU=$g CONFIG=pi05_red_blue_v2_$arm EXP=${EXP}_1gpu LOG_INTERVAL=10 \
      EXTRA_ARGS="${STEPS_ARGS:---num-train-steps 15000 --batch-size 16 --save-interval 1000 --keep-period 1000}" setsid nohup scripts/train_avalon.sh >/dev/null 2>&1 &
    g=$((g + 4))
  done
  wait
fi
echo "[train] $(date -Is) done"
