#!/usr/bin/env bash

TUNNEL_PID=""
TUNNEL_OWNED=0

# Slow jump hosts often need 30s+ for a second SSH handshake. One session both
# verifies the policy port and holds the LocalForward so we only pay that cost once.
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-30}"

policy_tunnel_is_ready() {
  (exec 3<>"/dev/tcp/${LOCAL_POLICY_HOST}/${LOCAL_POLICY_PORT}") 2>/dev/null
}

policy_ssh_base() {
  ssh \
    -p "$SSH_PORT" \
    -i "$SSH_KEY" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "$@"
}

policy_tunnel_start() {
  if policy_tunnel_is_ready; then
    echo "[launcher] Reusing existing tunnel on ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT}"
    return 0
  fi

  echo "[launcher] Opening SSH tunnel ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT} -> ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}"
  echo "[launcher] (single SSH: check remote policy port, then hold the forward)"

  # Exit 41 = authenticated but policy port not accepting. Keeps the LocalForward
  # alive with sleep only after the remote check passes.
  policy_ssh_base \
    -o ExitOnForwardFailure=yes \
    -L "${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT}:${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}" \
    "${SSH_USER}@${SSH_HOST}" \
    "timeout 3 bash -c 'echo >/dev/tcp/${REMOTE_POLICY_HOST}/${REMOTE_POLICY_PORT}' \
      || { echo \"[remote] policy not listening on ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}\" >&2; exit 41; }; \
     while true; do sleep 3600; done" &
  TUNNEL_PID=$!
  TUNNEL_OWNED=1

  local started=$SECONDS
  local deadline=$((started + TUNNEL_WAIT_SECONDS))
  local next_log=$((started + 5))
  until policy_tunnel_is_ready; do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      local rc=0
      wait "$TUNNEL_PID" || rc=$?
      if (( rc == 41 )); then
        echo "[launcher] Policy server is NOT listening on ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}." >&2
        echo "[launcher] Start the remote ZMQ/WebSocket inference session, then rerun." >&2
      else
        echo "[launcher] SSH tunnel exited before becoming ready (exit ${rc})." >&2
        echo "[launcher] Check SSH_HOST/SSH_PORT/SSH_KEY and jump-host reachability." >&2
      fi
      TUNNEL_OWNED=0
      TUNNEL_PID=""
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[launcher] Timed out waiting for SSH tunnel after ${TUNNEL_WAIT_SECONDS}s." >&2
      echo "[launcher] Jump-host auth can take 30s+; raise TUNNEL_WAIT_SECONDS (e.g. 90) and retry." >&2
      policy_tunnel_stop
      return 1
    fi
    if (( SECONDS >= next_log )); then
      echo "[launcher] still waiting for tunnel... $((SECONDS - started))s / ${TUNNEL_WAIT_SECONDS}s"
      next_log=$((SECONDS + 5))
    fi
    sleep 0.2
  done

  # Local listen is up after auth; give the remote tcp check a moment to finish.
  sleep 0.5
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    local rc=0
    wait "$TUNNEL_PID" || rc=$?
    if (( rc == 41 )); then
      echo "[launcher] Policy server is NOT listening on ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}." >&2
      echo "[launcher] Start the remote ZMQ/WebSocket inference session, then rerun." >&2
    else
      echo "[launcher] SSH tunnel dropped right after bind (exit ${rc})." >&2
    fi
    TUNNEL_OWNED=0
    TUNNEL_PID=""
    return 1
  fi

  echo "[launcher] Policy server is listening on ${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}"
  echo "[launcher] Tunnel ready on ${LOCAL_POLICY_HOST}:${LOCAL_POLICY_PORT}"
}

policy_tunnel_stop() {
  if (( TUNNEL_OWNED == 1 )) && [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  TUNNEL_OWNED=0
  TUNNEL_PID=""
}
