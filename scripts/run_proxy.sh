#!/bin/bash
# Start the proxy between an environment client and a policy server (reference configuration: the planner loop, i.e. the
# VLM writer with the sub-goal method).
#   UPSTREAM_PORT=18000 LISTEN_PORT=19000 MEMORY=stores/ep0-49_s8.npz ./scripts/run_proxy.sh
#   UPSTREAM_PORT=18000 LISTEN_PORT=19000 WRITER=rules METHOD=history ./scripts/run_proxy.sh
# The environment connects to LISTEN_PORT instead of the policy server and sends the memory/* control keys
# (see vla_memory/proxy/relay.py). Requires: numpy, pillow, requests, websockets, openpi-client (for the openpi protocol).
set -u
cd "$(dirname "$0")/.."
UPSTREAM_PORT=${UPSTREAM_PORT:?}; LISTEN_PORT=${LISTEN_PORT:?}
WRITER=${WRITER:-vlm}; METHOD=${METHOD:-subgoal}
VLM_OPTIONS=()
if [ "$WRITER" = vlm ]; then
  VLM_OPTIONS=(--vlm-url "${VLM_URL:-http://127.0.0.1:8100/v1}" --vlm-model "${VLM_MODEL:-Qwen3.8-27B-INT4}"
               ${MEMORY:+--memory "$MEMORY"} --k "${K:-4}" --votes "${VOTES:-1}" --min-calls "${MIN_CALLS:-2}")
fi
python -m vla_memory.proxy --listen-port "$LISTEN_PORT" --upstream-port "$UPSTREAM_PORT" --writer "$WRITER" --method "$METHOD" \
  "${VLM_OPTIONS[@]}" ${TASK:+--task "$TASK"} ${LOG:+--log "$LOG"} ${EXTRA:-}
