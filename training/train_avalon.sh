#!/bin/bash
# openpi fine-tuning on avalon1 -- the non-SLURM counterpart of
# sbatch/12_finetune_pi05.sbatch (same container, same env, same checks).
#
# avalon1: 8x RTX 3090 (24 GB, no ECC, no NVLink), GPUs 0-3 on NUMA node 0 and
# 4-7 on node 1. Shared with other users, no scheduler. DEFAULT: ONE GPU, no
# FSDP -- the multi-GPU FSDP step is what corrupted weights on TCML. The weight
# guard in train.py (OPENPI_WEIGHT_GUARD, exit code 3) is on in every mode.
#
# Launch detached (survives logout):
#   GPU=0 CONFIG=pi05_red_blue_v2_subgoal EXP=v2_1gpu \
#     setsid nohup scripts/train_avalon.sh >/dev/null 2>&1 &
# Log:         $RUNS/logs/<CONFIG>__<EXP>.log   (+ _gpu.csv, VRAM every 5 s)
# Checkpoints: $RUNS/checkpoints/<CONFIG>/<EXP>/<step>
# wandb:       $RUNS/wandb (offline; `wandb sync` later)
# Resume:      same CONFIG/EXP with RESUME=1.
# Extra train.py flags: EXTRA_ARGS="--keep-period 1000 --seed 7"
set -uo pipefail
WORK=${WORK:-$HOME/vla-memory}
RUNS=${RUNS:-/data/mhartmann/vla-runs}   # /home is too small for 30 checkpoints
CONFIG=${CONFIG:?set CONFIG}
EXP=${EXP:?set EXP}
GPU=${GPU:?set GPU, e.g. GPU=0 or GPU=0,1}
RESUME=${RESUME:-0}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOG_INTERVAL=${LOG_INTERVAL:-10}
FSDP=${FSDP:-1}   # 1 = no sharding (pure data parallel if NGPU > 1)

mkdir -p "$RUNS/logs" "$RUNS/checkpoints" "$RUNS/wandb"
LOG="$RUNS/logs/${CONFIG}__${EXP}.log"
GPULOG="$RUNS/logs/${CONFIG}__${EXP}_gpu.csv"
exec >>"$LOG" 2>&1

if [ "$RESUME" = "1" ]; then FLAG=--resume; else FLAG=--overwrite; fi
EXTRA="--fsdp-devices $FSDP --log-interval $LOG_INTERVAL --checkpoint-base-dir $RUNS/checkpoints"
[ "$GRAD_ACCUM" != "1" ] && EXTRA="$EXTRA --grad-accum-steps $GRAD_ACCUM"
EXTRA="$EXTRA ${EXTRA_ARGS:-}"

# Norm stats are keyed by config name and repo_id; training without them
# silently learns on unnormalised actions.
case "$CONFIG" in
  pi05_red_blue_v2_subgoal) REPO=hartmann/red_blue_v2_abs_subgoal_train;;      # v2 data (physical grasp, cameras v2)
  pi05_red_blue_v2_monolith) REPO=hartmann/red_blue_v2_abs_train;;
  pi05_red_blue_v2_history) REPO=hartmann/red_blue_v2_abs_history_train;;   # v2 history-prompt arm (relabel_history.py)
  pi05_red_blue_v2_subgoal_recap) REPO=hartmann/red_blue_v2_recap_subgoal_train;;   # RECAP-lite: ckpt 20000 + own rollouts
  *) REPO=${REPO:?set REPO for config $CONFIG};;
esac
NORM="$WORK/openpi/assets/$CONFIG/$REPO/norm_stats.json"

echo "===== $(date -Is)  host=$(hostname)  pid=$$"
echo "config=$CONFIG exp=$EXP flag=$FLAG gpu=$GPU fsdp=$FSDP grad_accum=$GRAD_ACCUM"
echo "train.py args: $EXTRA"
if [ ! -s "$NORM" ]; then echo "[run] FATAL: no norm stats at $NORM"; exit 1; fi
echo "[run] norm stats: $NORM  md5 $(md5sum <"$NORM" | cut -c1-12)"
echo "[run] code: openpi $(git -C "$WORK/openpi" rev-parse --short HEAD)+local  train.py md5 $(md5sum <"$WORK/openpi/scripts/train.py" | cut -c1-12)  config.py md5 $(md5sum <"$WORK/openpi/src/openpi/training/config.py" | cut -c1-12)"
nvidia-smi -i "$GPU" --query-gpu=index,pci.bus_id,serial,name,memory.total,memory.used --format=csv,noheader
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" != "0" ]; then echo "[run] FATAL: GPU $GPU already has $BUSY process(es)"; exit 1; fi

# Pin CPU and host memory to the GPU's NUMA node (the data loader runs in-process).
BUS=$(nvidia-smi -i "${GPU%%,*}" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f' | sed 's/^0000//')
NODE=$(cat "/sys/bus/pci/devices/$BUS/numa_node" 2>/dev/null || echo -1)
NUMA=""
[ "$NODE" -ge 0 ] 2>/dev/null && NUMA="numactl --cpunodebind=$NODE --preferred=$NODE"
echo "[run] NUMA node $NODE"

( while true; do
    nvidia-smi -i "$GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >>"$GPULOG"
    sleep 5
  done ) &
GPUMON=$!
trap 'kill $GPUMON 2>/dev/null' EXIT

# Same env as the TCML sbatch, except PREALLOCATE: the container disables it
# (it was built to share a GPU with Isaac Sim), and without it pi05 LoRA at
# batch 16 OOMs on a 3090 at step 0 -- init grows the pool to 16.8 GB, then the
# step's 7.6 GB temp buffer no longer fits under the cap. Preallocated (openpi's
# upstream default) it runs at 22.0/24.6 GB. MEM_FRACTION 0.9 per openpi's README.
# PYTHONUNBUFFERED: the "Step N: loss=..." lines go to stdout, which Python
# block-buffers into a file -- without it they reach the log ~800 steps late.
# CUDA_VISIBLE_DEVICES goes in through SINGULARITYENV_*: singularity's --env splits its argument on commas, so a
# multi-GPU list such as 0,1,2,3 arrived as "0" plus stray variables "1", "2", "3" (JAX then saw one device).
export CUDA_VISIBLE_DEVICES=$GPU SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU
$NUMA singularity exec --nv -B "$WORK:/workspace" -B /data/mhartmann:/data/mhartmann \
  --env PYTHONPATH=/workspace/openpi/src \
  --env OPENPI_DATA_HOME=/workspace/checkpoints \
  --env HF_LEROBOT_HOME=/workspace/data/hf \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.9}" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOCATE:-true}" \
  --env JAX_TRACEBACK_FILTERING=off \
  --env PYTHONUNBUFFERED=1 \
  --env NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" \
  --env OPENPI_DONATE="${OPENPI_DONATE:-1}" \
  --env OPENPI_WEIGHT_GUARD="${OPENPI_WEIGHT_GUARD:-10.0}" \
  --env XLA_FLAGS="${XLA_FLAGS:-}" \
  --env WANDB_MODE="${WANDB_MODE:-offline}" --env WANDB_DIR="$RUNS" \
  --pwd /workspace/openpi "$WORK/containers/openpi.sif" \
  /opt/openpi-venv/bin/python /workspace/openpi/scripts/train.py "$CONFIG" \
    --exp-name="$EXP" $FLAG $EXTRA
RC=$?
kill $GPUMON 2>/dev/null
python3 - "$GPULOG" <<'PY'
import collections, csv, sys
peak, tot, util = collections.defaultdict(int), {}, collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1])):
    if len(r) < 5: continue
    g, used = int(r[1]), int(r[2]); tot[g] = int(r[3]); util[g].append(int(r[4])); peak[g] = max(peak[g], used)
for g in sorted(peak):
    print(f"[run] GPU {g}: peak {peak[g]} / {tot[g]} MiB ({100 * peak[g] / tot[g]:.1f}%), mean util {sum(util[g]) / len(util[g]):.0f}%")
PY
case $RC in
  0) echo "[run] ===== $(date -Is) finished OK";;
  3) echo "[run] ===== $(date -Is) WEIGHT GUARD ABORT (rc=3): see the WEIGHT GUARD line above";;
  *) echo "[run] ===== $(date -Is) train.py FAILED rc=$RC";;
esac
exit $RC
