#!/usr/bin/env bash
# Replay a recorded episode on the robot.
#
#   ./scripts/run_replay.sh ~/unitree_eai_environment/data/pick_place/episode_0000
#   ./scripts/run_replay.sh <episode> --dry-run
#   ./scripts/run_replay.sh <episode> --speed 0.5
#
# The first argument is the episode directory or its data.json. Anything after
# it is passed straight to teleop.replay.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DDS_INTERFACE="${UNITREE_DDSINTERFACE:-eth0}"

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cd "$REPO_ROOT"
  python -m teleop.replay --help
  echo
  echo "Usage: $(basename "$0") <episode dir or data.json> [teleop.replay options]"
  exit 0
fi

EPISODE="$1"
shift

EPISODE="${EPISODE/#\~/${HOME}}"
if [[ ! -e "$EPISODE" ]]; then
  echo "[replay] no such episode: $EPISODE" >&2
  exit 1
fi

cd "$REPO_ROOT"
export UNITREE_DDSINTERFACE="$DDS_INTERFACE"
exec python -m teleop.replay \
  --data-json "$EPISODE" \
  --network-interface "$DDS_INTERFACE" \
  "$@"
