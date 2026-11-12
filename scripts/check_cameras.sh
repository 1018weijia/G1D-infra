#!/usr/bin/env bash
# Fail unless teleimager is up and head + both wrist ZMQ streams are fresh.
# Frozen JPEGs (the publisher repeating one frame) count as not ready.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/load_robot_config.sh
source "$SCRIPT_DIR/lib/load_robot_config.sh"

if [[ -z "${IMAGE_HOST:-}" ]]; then
  g1d_load_robot_config
  IMAGE_HOST="$G1D_IMAGE_HOST"
fi

SERVICE="${TELEIMAGER_SERVICE:-teleimager.service}"
CONFIG_PORT="${CAM_CONFIG_PORT:-60000}"
SAMPLES="${CHECK_CAMERAS_SAMPLES:-3}"
GAP="${CHECK_CAMERAS_GAP:-0.4}"
RECV_TIMEOUT_MS="${CHECK_CAMERAS_TIMEOUT_MS:-2000}"

pick_python() {
  local cmd
  for cmd in python3 python "${HOME}/miniconda3/envs/tv/bin/python"; do
    if [[ -x "$(command -v "$cmd" 2>/dev/null || true)" || -x "$cmd" ]] \
      && "$cmd" -c "import zmq" >/dev/null 2>&1; then
      echo "$cmd"
      return 0
    fi
  done
  return 1
}

host_is_local() {
  local host="$1"
  case "$host" in
    localhost|127.0.0.1|::1) return 0 ;;
  esac
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -Fxq "$host"
}

service_pid() {
  systemctl show -p MainPID --value "$SERVICE" 2>/dev/null || true
}

echo "[check_cameras] image host ${IMAGE_HOST}"

PID_BEFORE=""
if host_is_local "$IMAGE_HOST"; then
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "[check_cameras] systemctl is not available." >&2
    exit 1
  fi
  state="$(systemctl is-active "$SERVICE" 2>/dev/null || true)"
  if [[ "$state" != "active" ]]; then
    echo "[check_cameras] ${SERVICE} is ${state:-missing}." >&2
    echo "[check_cameras] Restart with: sudo systemctl restart ${SERVICE}" >&2
    exit 1
  fi
  PID_BEFORE="$(service_pid)"
  if [[ -z "$PID_BEFORE" || "$PID_BEFORE" == "0" ]]; then
    echo "[check_cameras] ${SERVICE} is active but has no main process." >&2
    echo "[check_cameras] Restart with: sudo systemctl restart ${SERVICE}" >&2
    exit 1
  fi
  age="$(ps -o etimes= -p "$PID_BEFORE" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$age" && "$age" -lt 5 ]]; then
    echo "[check_cameras] ${SERVICE} just started (pid ${PID_BEFORE}, ${age}s). Waiting for cameras."
    deadline=$((SECONDS + 20))
    while (( SECONDS < deadline )); do
      sleep 1
      now="$(service_pid)"
      if [[ "$now" != "$PID_BEFORE" ]]; then
        echo "[check_cameras] ${SERVICE} restarted while coming up (pid ${PID_BEFORE} -> ${now:-dead})." >&2
        echo "[check_cameras] Restart with: sudo systemctl restart ${SERVICE}" >&2
        exit 1
      fi
      age="$(ps -o etimes= -p "$PID_BEFORE" 2>/dev/null | tr -d ' ' || true)"
      if [[ -n "$age" && "$age" -ge 5 ]]; then
        break
      fi
    done
  fi
  echo "[check_cameras] ${SERVICE} is active (pid ${PID_BEFORE})"
fi

PYTHON="$(pick_python || true)"
if [[ -z "$PYTHON" ]]; then
  echo "[check_cameras] no Python with pyzmq found." >&2
  exit 1
fi

set +e
"$PYTHON" - "$IMAGE_HOST" "$CONFIG_PORT" "$SAMPLES" "$GAP" "$RECV_TIMEOUT_MS" <<'PY'
import hashlib
import sys
import time

import zmq

host, config_port, samples, gap, timeout_ms = sys.argv[1:]
samples = int(samples)
gap = float(gap)
timeout_ms = int(timeout_ms)
required = ("head_camera", "left_wrist_camera", "right_wrist_camera")

ctx = zmq.Context()

def close_socket(sock):
    try:
        sock.close(0)
    except Exception:
        pass

def fetch_config():
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{config_port}")
    try:
        sock.send(b"GET_DATA")
        return sock.recv_json()
    finally:
        close_socket(sock)

def sample_hashes(port):
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    time.sleep(0.25)
    hashes = []
    try:
        for _ in range(samples):
            payload = sock.recv()
            if len(payload) < 3 or payload[0] != 0xFF or payload[1] != 0xD8:
                return "NOT_JPEG", hashes
            hashes.append(hashlib.md5(payload).hexdigest())
            time.sleep(gap)
    except zmq.Again:
        return "TIMEOUT", hashes
    finally:
        close_socket(sock)
    if len(hashes) < samples:
        return "TIMEOUT", hashes
    if len(set(hashes)) == 1:
        return "FROZEN", hashes
    return "OK", hashes

try:
    config = fetch_config()
except Exception as exc:
    print(f"CONFIG_FAIL {exc}")
    sys.exit(1)

if not isinstance(config, dict):
    print("CONFIG_FAIL camera config response was not an object")
    sys.exit(1)

detail = {
    "OK": "fresh frames",
    "FROZEN": "image is not updating",
    "TIMEOUT": "no frame",
    "NOT_JPEG": "payload is not a JPEG",
}
failed = False
for name in required:
    cam = config.get(name)
    if not isinstance(cam, dict):
        print(f"[check_cameras] {name}: missing from the running camera config")
        failed = True
        continue
    if not cam.get("enable_zmq"):
        print(f"[check_cameras] {name}: ZMQ publishing is disabled")
        failed = True
        continue
    port = cam.get("zmq_port")
    if not port:
        print(f"[check_cameras] {name}: no zmq_port in the running camera config")
        failed = True
        continue
    status, _hashes = sample_hashes(int(port))
    print(f"[check_cameras] {name} :{port} {detail.get(status, status)}")
    if status != "OK":
        failed = True

ctx.term()
sys.exit(1 if failed else 0)
PY
rc=$?
set -e

if (( rc != 0 )); then
  echo "[check_cameras] camera streams are not ready." >&2
  echo "[check_cameras] Restart with: sudo systemctl restart ${SERVICE}" >&2
  exit 1
fi

if [[ -n "$PID_BEFORE" ]]; then
  pid_after="$(service_pid)"
  if [[ "$pid_after" != "$PID_BEFORE" ]]; then
    echo "[check_cameras] ${SERVICE} restarted during the check (pid ${PID_BEFORE} -> ${pid_after:-dead})." >&2
    echo "[check_cameras] Restart with: sudo systemctl restart ${SERVICE}" >&2
    exit 1
  fi
fi

echo "[check_cameras] head and both wrist cameras are publishing fresh frames"
exit 0
