#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"

VOLUME_MOUNT="/data"
PROJECT_KEY="patho-v-search"
START_QDRANT=0

usage() {
  cat <<'EOF'
Usage: vast_setup_volume.sh [options]

Prepare a Vast volume layout and symlink project data directories so large
embedding files and Qdrant storage stay on persistent volume storage.

Options:
  --volume-mount PATH   Volume mount path inside instance (default: /data)
  --project-key NAME    Folder name under volume mount (default: patho-v-search)
  --repo-root PATH      Repo root to operate on (default: current repo root)
  --start-qdrant        Start Qdrant after setup using volume-backed storage
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volume-mount)
      VOLUME_MOUNT="$2"
      shift 2
      ;;
    --project-key)
      PROJECT_KEY="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --start-qdrant)
      START_QDRANT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repo root does not exist: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -d "$VOLUME_MOUNT" ]]; then
  echo "Volume mount not found: $VOLUME_MOUNT" >&2
  echo "Attach a Vast volume first, then re-run this script." >&2
  exit 1
fi

VOLUME_ROOT="${VOLUME_MOUNT%/}/${PROJECT_KEY}"
VOLUME_RAW="${VOLUME_ROOT}/raw_embeddings"
VOLUME_SLIDES="${VOLUME_RAW}/slides_embeddings"
VOLUME_PATCHES="${VOLUME_RAW}/patch_embeddings"
VOLUME_QDRANT="${VOLUME_ROOT}/qdrant_storage"

REPO_DATA_DIR="${REPO_ROOT}/data"
REPO_SLIDES="${REPO_DATA_DIR}/slides_embeddings"
REPO_PATCHES="${REPO_DATA_DIR}/patch_embeddings"
REPO_QDRANT="${REPO_DATA_DIR}/qdrant_storage"

mkdir -p "$REPO_DATA_DIR"
mkdir -p "$VOLUME_SLIDES" "$VOLUME_PATCHES" "$VOLUME_QDRANT"

timestamp="$(date +%Y%m%d-%H%M%S)"

link_path() {
  local src="$1"
  local dst="$2"

  if [[ -L "$dst" ]]; then
    local current
    current="$(readlink -f "$dst" || true)"
    if [[ "$current" == "$(readlink -f "$src")" ]]; then
      echo "[ok] $dst -> $src"
      return 0
    fi
    local backup="${dst}.bak.${timestamp}"
    mv "$dst" "$backup"
    echo "[moved] existing symlink to $backup"
  elif [[ -e "$dst" ]]; then
    local backup="${dst}.bak.${timestamp}"
    mv "$dst" "$backup"
    echo "[moved] existing path to $backup"
  fi

  ln -s "$src" "$dst"
  echo "[linked] $dst -> $src"
}

link_path "$VOLUME_SLIDES" "$REPO_SLIDES"
link_path "$VOLUME_PATCHES" "$REPO_PATCHES"
link_path "$VOLUME_QDRANT" "$REPO_QDRANT"

echo
echo "Volume layout ready:"
echo "  Raw slides:   $VOLUME_SLIDES"
echo "  Raw patches:  $VOLUME_PATCHES"
echo "  Qdrant store: $VOLUME_QDRANT"
echo
echo "Next:"
echo "  1) Put raw embeddings in:"
echo "     $VOLUME_SLIDES"
echo "     $VOLUME_PATCHES"
echo "  2) Ingest:"
echo "     cd $REPO_ROOT && source .venv/bin/activate && python ingest.py --slides_dir data/slides_embeddings --patches_dir data/patch_embeddings"

if [[ "$START_QDRANT" -eq 1 ]]; then
  if [[ ! -x "${REPO_ROOT}/scripts/qdrant_local.sh" ]]; then
    echo "Cannot start Qdrant: scripts/qdrant_local.sh not found." >&2
    exit 1
  fi
  echo "  3) Starting Qdrant now with QDRANT_STORAGE_PATH=$VOLUME_QDRANT"
  (
    cd "$REPO_ROOT"
    QDRANT_STORAGE_PATH="$VOLUME_QDRANT" bash scripts/qdrant_local.sh start
  )
fi
