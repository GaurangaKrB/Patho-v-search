#!/usr/bin/env bash
set -euo pipefail

HF_URL_DEFAULT="https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/resolve/main/GigaPath_PANDA_embeddings.zip"
DEST_ROOT="/data/patho-v-search/raw_embeddings"
ZIP_PATH=""
DOWNLOAD_ONLY=0

usage() {
  cat <<'EOF'
Usage: download_panda_embeddings.sh [options]

Download and extract Prov-GigaPath PANDA embeddings from Hugging Face.
Supports resume for interrupted downloads.

Options:
  --url URL          Source zip URL (default: official HF resolve URL)
  --dest-root PATH   Destination root (default: /data/patho-v-search/raw_embeddings)
  --zip-path PATH    Explicit zip output path (default: <dest-root>/GigaPath_PANDA_embeddings.zip)
  --download-only    Download zip only, skip extraction
  -h, --help         Show help

Note:
  If you have a Hugging Face "blob" URL, replace "/blob/" with "/resolve/".
EOF
}

HF_URL="$HF_URL_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      HF_URL="$2"
      shift 2
      ;;
    --dest-root)
      DEST_ROOT="$2"
      shift 2
      ;;
    --zip-path)
      ZIP_PATH="$2"
      shift 2
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
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

if [[ "$HF_URL" == *"/blob/"* ]]; then
  HF_URL="${HF_URL/\/blob\//\/resolve\/}"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing dependency: curl" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"
if [[ -z "$ZIP_PATH" ]]; then
  ZIP_PATH="${DEST_ROOT%/}/GigaPath_PANDA_embeddings.zip"
fi

echo "URL:      $HF_URL"
echo "ZIP:      $ZIP_PATH"
echo "DEST DIR: $DEST_ROOT"
echo
echo "Downloading (resume enabled)..."
curl -fL -C - "$HF_URL" -o "$ZIP_PATH"

if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
  echo "Download complete (extraction skipped)."
  exit 0
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Missing dependency: unzip" >&2
  echo "Install it, then extract manually:" >&2
  echo "  unzip -o \"$ZIP_PATH\" -d \"$DEST_ROOT\"" >&2
  exit 1
fi

echo "Extracting..."
unzip -o "$ZIP_PATH" -d "$DEST_ROOT"

h5_source_dir="${DEST_ROOT%/}/GigaPath_PANDA_embeddings/h5_files"
patches_dir="${DEST_ROOT%/}/patch_embeddings"
if [[ -d "$h5_source_dir" ]]; then
  mkdir -p "$patches_dir"
  if [[ ! -e "${patches_dir}/h5_files" ]]; then
    ln -s "$h5_source_dir" "${patches_dir}/h5_files"
    echo "Linked ${patches_dir}/h5_files -> ${h5_source_dir}"
  fi
fi

echo
echo "Extraction complete. Quick inventory:"
find "$DEST_ROOT" -maxdepth 3 -type d | sed -n '1,40p'
echo
echo "Embedding file count by expected folders:"
slides_count="$(find -L "$DEST_ROOT/slides_embeddings" -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
patches_pt_count="$(find -L "$DEST_ROOT/patch_embeddings" -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
patches_h5_count="$(find -L "$DEST_ROOT/patch_embeddings" -type f -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')"
echo "  slides_embeddings (.pt): $slides_count"
echo "  patch_embeddings (.pt):  $patches_pt_count"
echo "  patch_embeddings (.h5):  $patches_h5_count"
