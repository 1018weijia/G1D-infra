#!/usr/bin/env bash

TUNNEL_PID=""
TUNNEL_OWNED=0

policy_tunnel_is_ready() {
  (exec 3<>"/dev/tcp/${LOCAL_POLICY_HOST}/${LOCAL_POLICY_PORT}") 2>/dev/null
}

policy_remote_is_listening() {
  ssh \
    -p "$SSH_PORT" \
    -i "$SSH_KEY" \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    "${SSH_USER}@${SSH_HOST}" \
    "timeout 3 bash -c 'echo >/dev/tcp/${REMOTE_POLICY_HOST}/${REMOTE_POLICY_PORT}'" \
    >/dev/null 2>&1
}

policy_require_remote() {
  if policy_remote_is_listening; then
    echo "[launcher] Policy server is listening on ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}"
    return 0
  fi
  echo "[launcher] Policy server is NOT listening on ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}." >&2
  echo "[launcher] Local ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT} being open only means the SSH tunnel exists." >&2
  echo "[launcher] Start the remote ZMQ/WebSocket inference session, then rerun." >&2
  return 1
}

policy_tunnel_start() {
  if ! policy_require_remote; then
    return 1
  fi

  if policy_tunnel_is_ready; then
    echo "[launcher] Reusing existing tunnel on ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT}"
    return 0
  fi

  echo "[launcher] Opening SSH tunnel ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT} -> ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}"
  ssh \
    -p "$SSH_PORT" \
    -i "$SSH_KEY" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -N \
    -L "${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT}:${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}" \
    "${SSH_USER}@${SSH_HOST}" &
  TUNNEL_PID=$!
  TUNNEL_OWNED=1

  local deadline=$((SECONDS + TUNNEL_WAIT_SECONDS))
  until policy_tunnel_is_ready; do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      wait "$TUNNEL_PID" || true
      echo "[launcher] SSH tunnel exited before becoming ready." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[launcher] Timed out waiting for SSH tunnel." >&2
      return 1
    fi
    sleep 0.2
  done
}

policy_tunnel_stop() {
  if (( TUNNEL_OWNED == 1 )) && [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
}
