#!/bin/bash
# Start the planner proxy between an environment client and a policy server (reference configuration).
#   UPSTREAM_PORT=18000 LISTEN_PORT=19000 MEMORY=stores/ep0-49_s8.npz ./scripts/run_proxy.sh
# The environment connects to LISTEN_PORT instead of the policy server and sends the memory/* control keys
# (see vla_memory/planner_loop/proxy.py). Requires: numpy, pillow, requests, websockets, openpi-client (for the openpi protocol).
set -u
cd "$(dirname "$0")/.."
UPSTREAM_PORT=${UPSTREAM_PORT:?}; LISTEN_PORT=${LISTEN_PORT:?}
python -m vla_memory.planner_loop.proxy --listen-port "$LISTEN_PORT" --upstream-port "$UPSTREAM_PORT" --mode "${MODE:-vlm}" \
  --vlm-url "${VLM_URL:-http://127.0.0.1:8100/v1}" --vlm-model "${VLM_MODEL:-Qwen3.8-27B-INT4}" \
  ${TASK:+--task "$TASK"} ${MEMORY:+--memory "$MEMORY"} --k "${K:-4}" --votes "${VOTES:-1}" --min-calls "${MIN_CALLS:-2}" \
  ${LOG:+--log "$LOG"} ${EXTRA:-}
