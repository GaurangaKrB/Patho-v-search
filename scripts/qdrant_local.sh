#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDRANT_HOME="${QDRANT_HOME:-$ROOT_DIR/.qdrant}"
QDRANT_BIN_DIR="$QDRANT_HOME/bin"
QDRANT_BIN="$QDRANT_BIN_DIR/qdrant"
QDRANT_LOG="${QDRANT_LOG:-$QDRANT_HOME/qdrant.log}"
QDRANT_PID_FILE="${QDRANT_PID_FILE:-$QDRANT_HOME/qdrant.pid}"
QDRANT_STORAGE_PATH="${QDRANT_STORAGE_PATH:-$ROOT_DIR/data/qdrant_storage}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
QDRANT_VERSION="${QDRANT_VERSION:-latest}"

latest_release_tag() {
  curl -fsSL https://api.github.com/repos/qdrant/qdrant/releases/latest \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'
}

resolve_qdrant_version() {
  if [[ "$QDRANT_VERSION" == "latest" ]]; then
    latest_release_tag
  else
    printf '%s\n' "$QDRANT_VERSION"
  fi
}

download_qdrant() {
  mkdir -p "$QDRANT_BIN_DIR"
  local version
  version="$(resolve_qdrant_version)"
  local url="https://github.com/qdrant/qdrant/releases/download/${version}/qdrant-x86_64-unknown-linux-gnu.tar.gz"

  local archive="$QDRANT_HOME/qdrant-${version}.tar.gz"
  curl -fL "$url" -o "$archive"
  tar -xzf "$archive" -C "$QDRANT_BIN_DIR"
  chmod +x "$QDRANT_BIN"
  rm -f "$archive"

  echo "Installed Qdrant ${version} to ${QDRANT_BIN}"
}

ensure_qdrant_installed() {
  if [[ ! -x "$QDRANT_BIN" ]]; then
    download_qdrant
  fi
}

is_running() {
  [[ -f "$QDRANT_PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$QDRANT_PID_FILE")"
  kill -0 "$pid" 2>/dev/null
}

start_qdrant() {
  ensure_qdrant_installed
  mkdir -p "$QDRANT_HOME" "$QDRANT_STORAGE_PATH"
  mkdir -p "$QDRANT_STORAGE_PATH/.deleted"

  if is_running; then
    echo "Qdrant already running (pid=$(cat "$QDRANT_PID_FILE"))."
    return 0
  fi

  (
    cd "$QDRANT_HOME"
    if command -v setsid >/dev/null 2>&1; then
      env \
        QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE_PATH" \
        QDRANT__SERVICE__HTTP_PORT="$QDRANT_HTTP_PORT" \
        QDRANT__SERVICE__GRPC_PORT="$QDRANT_GRPC_PORT" \
        setsid "$QDRANT_BIN" >"$QDRANT_LOG" 2>&1 < /dev/null &
    else
      env \
        QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE_PATH" \
        QDRANT__SERVICE__HTTP_PORT="$QDRANT_HTTP_PORT" \
        QDRANT__SERVICE__GRPC_PORT="$QDRANT_GRPC_PORT" \
        nohup "$QDRANT_BIN" >"$QDRANT_LOG" 2>&1 &
    fi
    echo $! >"$QDRANT_PID_FILE"
  )

  sleep 1
  if is_running; then
    echo "Qdrant started (pid=$(cat "$QDRANT_PID_FILE"), http=${QDRANT_HTTP_PORT}, grpc=${QDRANT_GRPC_PORT})."
  else
    echo "Qdrant failed to start. Recent logs:"
    tail -n 60 "$QDRANT_LOG" || true
    return 1
  fi
}

stop_qdrant() {
  if ! [[ -f "$QDRANT_PID_FILE" ]]; then
    echo "Qdrant is not running."
    return 0
  fi

  local pid
  pid="$(cat "$QDRANT_PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    sleep 1
  fi
  rm -f "$QDRANT_PID_FILE"
  echo "Qdrant stopped."
}

status_qdrant() {
  if is_running; then
    local pid
    pid="$(cat "$QDRANT_PID_FILE")"
    echo "Qdrant is running (pid=${pid})."
    curl -fsS "http://127.0.0.1:${QDRANT_HTTP_PORT}/healthz"
    echo " (healthz)"
    curl -fsS "http://127.0.0.1:${QDRANT_HTTP_PORT}/readyz"
    echo " (readyz)"
  else
    echo "Qdrant is not running."
    return 1
  fi
}

logs_qdrant() {
  local lines="${2:-80}"
  tail -n "$lines" "$QDRANT_LOG"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <install|start|stop|status|logs> [args]

Commands:
  install            Download qdrant binary into .qdrant/bin
  start              Start local qdrant server in background
  stop               Stop local qdrant server
  status             Show process status and health endpoints
  logs [N]           Show last N lines from qdrant log (default: 80)

Env (optional):
  QDRANT_VERSION      Default: latest
  QDRANT_STORAGE_PATH Default: $ROOT_DIR/data/qdrant_storage
  QDRANT_HTTP_PORT    Default: 6333
  QDRANT_GRPC_PORT    Default: 6334
EOF
}

cmd="${1:-}"
case "$cmd" in
  install)
    download_qdrant
    ;;
  start)
    start_qdrant
    ;;
  stop)
    stop_qdrant
    ;;
  status)
    status_qdrant
    ;;
  logs)
    logs_qdrant "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
