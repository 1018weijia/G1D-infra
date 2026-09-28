#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=lib/load_robot_config.sh
source "$SCRIPT_DIR/lib/load_robot_config.sh"
g1d_load_robot_config

SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_POLICY_HOST="${REMOTE_POLICY_HOST:-}"
REMOTE_POLICY_PORT="${REMOTE_POLICY_PORT:-5555}"
LOCAL_POLICY_HOST="${LOCAL_POLICY_HOST:-$G1D_LOCAL_POLICY_HOST}"
LOCAL_POLICY_PORT="${LOCAL_POLICY_PORT:-$G1D_LOCAL_POLICY_PORT}"
POLICY_PROTOCOL="${POLICY_PROTOCOL:-$G1D_POLICY_PROTOCOL}"
IMAGE_HOST="${IMAGE_HOST:-$G1D_IMAGE_HOST}"
DDS_INTERFACE="${UNITREE_DDSINTERFACE:-${DDS_INTERFACE:-$G1D_DDS_INTERFACE}}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/infer_g1d.yaml}"
READY_POSE_CONFIG="${READY_POSE_CONFIG:-$G1D_READY_POSE_CONFIG}"
READY_POSE_SECONDS="${READY_POSE_SECONDS:-3.0}"
ALIGNMENT_TARGET_CONFIG="${ALIGNMENT_TARGET_CONFIG:-$G1D_ALIGNMENT_TARGETS}"
INSTRUCTION="${INSTRUCTION:-}"
ACTION_INTERP_FACTOR="${ACTION_INTERP_FACTOR:-1}"
EXEC_CHUNK_STEPS="${EXEC_CHUNK_STEPS:-0}"
INPUT_MODE="${INPUT_MODE:-controller}"
TUNNEL_WAIT_SECONDS="${TUNNEL_WAIT_SECONDS:-60}"
DRY_RUN=0

export PAD_JOINT_VALUES="${PAD_JOINT_VALUES-}"
export SWAP_WRISTS="${SWAP_WRISTS:-1}"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

source "$SCRIPT_DIR/lib/ssh_tunnel.sh"
cleanup() {
  policy_tunnel_stop
}
trap cleanup EXIT INT TERM

if [[ ! -r "$CONFIG_PATH" ]]; then
  echo "[launcher] Policy config is not readable: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -r "$READY_POSE_CONFIG" ]]; then
  echo "[launcher] Ready pose config is not readable: $READY_POSE_CONFIG" >&2
  exit 1
fi

if (( DRY_RUN == 1 )); then
  echo "[launcher] dry-run"
  echo "[launcher] tunnel: ${SSH_USER}@${SSH_HOST:-<unset>}:${SSH_PORT} -> ${REMOTE_POLICY_HOST:-<unset>}:${REMOTE_POLICY_PORT} via ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT}"
  echo "[launcher] protocol: $POLICY_PROTOCOL"
  echo "[launcher] input-mode: $INPUT_MODE"
  echo "[launcher] config: $CONFIG_PATH"
  echo "[launcher] ready pose: $READY_POSE_CONFIG (${READY_POSE_SECONDS}s)"
  echo "[launcher] alignment: $ALIGNMENT_TARGET_CONFIG"
  echo "[launcher] image:  $IMAGE_HOST"
  echo "[launcher] instruction: ${INSTRUCTION:-<unset>}"
  exit 0
fi

if [[ -z "$INSTRUCTION" ]]; then
  echo "[launcher] Set INSTRUCTION to the task language." >&2
  exit 1
fi

if [[ -z "$SSH_HOST" || -z "$SSH_KEY" || -z "$REMOTE_POLICY_HOST" ]]; then
  echo "[launcher] Set SSH_HOST, SSH_KEY, and REMOTE_POLICY_HOST (see GUIDE_DEPLOY.md)." >&2
  exit 1
fi

if [[ ! -r "$SSH_KEY" ]]; then
  echo "[launcher] SSH key is not readable: $SSH_KEY" >&2
  exit 1
fi

policy_tunnel_start

if ! bash "$SCRIPT_DIR/wait_policy.sh" "$LOCAL_POLICY_HOST" "$LOCAL_POLICY_PORT" "$POLICY_PROTOCOL"; then
  echo "[launcher] Policy probe failed; not starting policy_deploy.py." >&2
  exit 1
fi

cd "$REPO_ROOT"
export UNITREE_DDSINTERFACE="$DDS_INTERFACE"
python policy_deploy.py \
  --config-path "$CONFIG_PATH" \
  --ready-pose-config "$READY_POSE_CONFIG" \
  --ready-pose-seconds "$READY_POSE_SECONDS" \
  --alignment-target-config "$ALIGNMENT_TARGET_CONFIG" \
  --instruction "$INSTRUCTION" \
  --server-host "$LOCAL_POLICY_HOST" \
  --server-port "$LOCAL_POLICY_PORT" \
  --protocol "$POLICY_PROTOCOL" \
  --input-mode "$INPUT_MODE" \
  --action-interp-factor "$ACTION_INTERP_FACTOR" \
  --exec-chunk-steps "$EXEC_CHUNK_STEPS" \
  --img-server-ip "$IMAGE_HOST" \
  --network-interface "$DDS_INTERFACE" \
  --display-mode "${DISPLAY_MODE:-ego}" \
  --ego-pixel-overlay \
  --rollback-seconds "${ROLLBACK_SECONDS:-3.0}" \
  --policy-prefetch-steps "${POLICY_PREFETCH_STEPS:-4}" \
  "$@"
