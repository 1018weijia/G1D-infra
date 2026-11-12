#!/usr/bin/env bash
# Restart teleimager before the camera preflight. Password is supplied to sudo -S.
set -Eeuo pipefail

SERVICE="${TELEIMAGER_SERVICE:-teleimager.service}"
SUDO_PASSWORD="${SUDO_PASSWORD:-123}"
SETTLE_SECONDS="${CAMERA_RESTART_SETTLE_SECONDS:-8}"

echo "[restart_cameras] restarting ${SERVICE}"
printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' systemctl restart "$SERVICE"

state="$(systemctl is-active "$SERVICE" 2>/dev/null || true)"
if [[ "$state" != "active" ]]; then
  echo "[restart_cameras] ${SERVICE} is ${state:-missing} after restart." >&2
  exit 1
fi

echo "[restart_cameras] ${SERVICE} is active, waiting ${SETTLE_SECONDS}s for cameras to open"
sleep "$SETTLE_SECONDS"
echo "[restart_cameras] ${SERVICE} restarted"
