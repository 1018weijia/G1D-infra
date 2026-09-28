#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=lib/load_robot_config.sh
source "$SCRIPT_DIR/lib/load_robot_config.sh"
g1d_load_robot_config

IMAGE_HOST="${IMAGE_HOST:-$G1D_IMAGE_HOST}"
DDS_INTERFACE="${UNITREE_DDSINTERFACE:-${DDS_INTERFACE:-$G1D_DDS_INTERFACE}}"
INPUT_MODE="${INPUT_MODE:-controller}"
TASK_DIR="${TASK_DIR:-$G1D_TASK_DIR}"
TASK_NAME="${TASK_NAME:-$G1D_TASK_NAME}"
TASK_GOAL="${TASK_GOAL:-$G1D_TASK_GOAL}"

cd "$REPO_ROOT"
export UNITREE_DDSINTERFACE="$DDS_INTERFACE"
python collect.py \
  --ee dex1_internal \
  --input-mode "$INPUT_MODE" \
  --img-server-ip "$IMAGE_HOST" \
  --network-interface "$DDS_INTERFACE" \
  --record \
  --task-dir "$TASK_DIR" \
  --task-name "$TASK_NAME" \
  --task-goal "$TASK_GOAL" \
  "$@"
