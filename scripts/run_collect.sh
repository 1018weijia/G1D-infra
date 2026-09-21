#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_HOST="${IMAGE_HOST:-192.168.123.164}"
DDS_INTERFACE="${UNITREE_DDSINTERFACE:-eth0}"
EE="${EE:-dex1_internal}"
INPUT_MODE="${INPUT_MODE:-controller}"
TASK_DIR="${TASK_DIR:-$HOME/unitree_eai_environment/data/}"
TASK_NAME="${TASK_NAME:-pick_place}"
TASK_GOAL="${TASK_GOAL:-pick and place}"

cd "$REPO_ROOT"
export UNITREE_DDSINTERFACE="$DDS_INTERFACE"
python collect.py \
  --ee "$EE" \
  --input-mode "$INPUT_MODE" \
  --img-server-ip "$IMAGE_HOST" \
  --network-interface "$DDS_INTERFACE" \
  --record \
  --task-dir "$TASK_DIR" \
  --task-name "$TASK_NAME" \
  --task-goal "$TASK_GOAL" \
  "$@"
