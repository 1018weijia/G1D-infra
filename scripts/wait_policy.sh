#!/usr/bin/env bash
# Wait until the local policy endpoint accepts connections (and optional ZMQ ping).
set -Eeuo pipefail

HOST="${1:-${LOCAL_POLICY_HOST:-127.0.0.1}}"
PORT="${2:-${LOCAL_POLICY_PORT:-15555}}"
PROTOCOL="${3:-${POLICY_PROTOCOL:-zmq}}"
TIMEOUT_SECONDS="${WAIT_POLICY_TIMEOUT:-30}"
INTERVAL_SECONDS="${WAIT_POLICY_INTERVAL:-0.5}"

deadline=$((SECONDS + TIMEOUT_SECONDS))

tcp_ok() {
  python3 - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

zmq_ping() {
  python3 - "$HOST" "$PORT" <<'PY'
import sys
try:
    import zmq
except ImportError:
    sys.exit(0)  # TCP already passed; skip ZMQ if unavailable
host, port = sys.argv[1], int(sys.argv[2])
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, 1000)
sock.setsockopt(zmq.SNDTIMEO, 1000)
sock.connect(f"tcp://{host}:{port}")
try:
    # Unknown methods should still prove a ZMQ peer is listening.
    sock.send_json({"cmd": "ping"})
    try:
        sock.recv()
    except zmq.Again:
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    sock.close(0)
PY
}

echo "[wait_policy] waiting for ${PROTOCOL}://${HOST}:${PORT} (timeout ${TIMEOUT_SECONDS}s)"
while (( SECONDS < deadline )); do
  if tcp_ok; then
    if [[ "${PROTOCOL}" == "zmq" ]]; then
      if zmq_ping; then
        echo "[wait_policy] policy endpoint is reachable (tcp+zmq)"
        exit 0
      fi
    else
      echo "[wait_policy] policy endpoint is reachable (tcp)"
      exit 0
    fi
  fi
  sleep "${INTERVAL_SECONDS}"
done

echo "[wait_policy] timed out waiting for ${HOST}:${PORT} (protocol=${PROTOCOL})." >&2
echo "[wait_policy] Tunnel may be up but the remote inference process is not listening." >&2
exit 1
