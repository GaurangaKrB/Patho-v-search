#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
BACKEND_API_URL="${BACKEND_API_URL:-http://127.0.0.1:8000}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
REQUIRE_INGEST="${REQUIRE_INGEST:-1}"

QDRANT_URL="${QDRANT_URL%/}"
BACKEND_API_URL="${BACKEND_API_URL%/}"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

print_ok() {
  echo "[PASS] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

print_warn() {
  echo "[WARN] $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

print_fail() {
  echo "[FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

header() {
  echo
  echo "== $1 =="
}

curl_qdrant() {
  local path="$1"
  local url="${QDRANT_URL}${path}"
  if [[ -n "$QDRANT_API_KEY" ]]; then
    curl -fsS -m 10 -H "api-key: ${QDRANT_API_KEY}" "$url"
  else
    curl -fsS -m 10 "$url"
  fi
}

header "Config"
echo "QDRANT_URL=${QDRANT_URL}"
echo "BACKEND_API_URL=${BACKEND_API_URL}"
echo "REQUIRE_INGEST=${REQUIRE_INGEST}"

header "Qdrant"
if ready_text="$(curl_qdrant /readyz 2>/dev/null)"; then
  print_ok "Qdrant reachable (${ready_text})"
else
  print_fail "Cannot reach Qdrant at ${QDRANT_URL} (/readyz)"
fi

collections_json=""
if collections_json="$(curl_qdrant /collections 2>/dev/null)"; then
  print_ok "Qdrant collections endpoint reachable"
else
  print_fail "Cannot read Qdrant collections at ${QDRANT_URL}/collections"
fi

slides_points=-1
patches_points=-1
collection_names=""

if [[ -n "$collections_json" ]]; then
  collection_names="$(
    python3 -c '
import json,sys
data=json.load(sys.stdin)
for c in data.get("result", {}).get("collections", []):
    print(c.get("name", ""))
' <<<"$collections_json" 2>/dev/null || true
  )"

  if grep -qx "Slides" <<<"$collection_names"; then
    print_ok "Slides collection exists"
    slides_points="$(
      curl_qdrant /collections/Slides \
        | python3 -c '
import json,sys
data=json.load(sys.stdin)
print(data.get("result", {}).get("points_count", 0))
' 2>/dev/null || echo "-1"
    )"
    if [[ "$slides_points" =~ ^[0-9]+$ ]]; then
      print_ok "Slides points_count=${slides_points}"
    else
      print_fail "Could not read Slides points_count"
    fi
  else
    print_fail "Slides collection missing"
  fi

  if grep -qx "Patches" <<<"$collection_names"; then
    print_ok "Patches collection exists"
    patches_points="$(
      curl_qdrant /collections/Patches \
        | python3 -c '
import json,sys
data=json.load(sys.stdin)
print(data.get("result", {}).get("points_count", 0))
' 2>/dev/null || echo "-1"
    )"
    if [[ "$patches_points" =~ ^[0-9]+$ ]]; then
      print_ok "Patches points_count=${patches_points}"
    else
      print_fail "Could not read Patches points_count"
    fi
  else
    print_fail "Patches collection missing"
  fi
fi

header "Backend"
if backend_health_json="$(curl -fsS -m 10 "${BACKEND_API_URL}/health" 2>/dev/null)"; then
  backend_ok="$(
    python3 -c '
import json,sys
data=json.load(sys.stdin)
print("true" if data.get("ok") is True else "false")
' <<<"$backend_health_json" 2>/dev/null || echo "false"
  )"
  if [[ "$backend_ok" == "true" ]]; then
    print_ok "Backend reachable and healthy"
  else
    print_fail "Backend /health responded but ok != true"
  fi
else
  print_fail "Cannot reach backend at ${BACKEND_API_URL}/health"
fi

header "Embedding Files"
slides_pt_files="$(find -L "${ROOT_DIR}/data/slides_embeddings" -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
slides_h5_files="$(find -L "${ROOT_DIR}/data/slides_embeddings" -type f -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')"
patches_pt_files="$(find -L "${ROOT_DIR}/data/patch_embeddings" -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
patches_h5_files="$(find -L "${ROOT_DIR}/data/patch_embeddings" -type f -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')"
slides_files=$((slides_pt_files + slides_h5_files))
patches_files=$((patches_pt_files + patches_h5_files))
echo "slides files:  pt=${slides_pt_files} h5=${slides_h5_files} total=${slides_files}"
echo "patches files: pt=${patches_pt_files} h5=${patches_h5_files} total=${patches_files}"

if [[ "$slides_files" -gt 0 ]]; then
  print_ok "Slides embedding files found"
else
  print_warn "No slide embeddings found in data/slides_embeddings"
fi

if [[ "$patches_files" -gt 0 ]]; then
  print_ok "Patch embedding files found"
else
  print_warn "No patch embeddings found in data/patch_embeddings"
fi

if [[ "$REQUIRE_INGEST" == "1" ]]; then
  if [[ "$slides_points" =~ ^[0-9]+$ ]] && [[ "$patches_points" =~ ^[0-9]+$ ]] \
    && [[ "$slides_points" -gt 0 ]] && [[ "$patches_points" -gt 0 ]]; then
    print_ok "Ingestion looks complete (non-zero points in both collections)"
  else
    print_fail "Ingestion incomplete (set REQUIRE_INGEST=0 to ignore this check)"
  fi
else
  print_warn "Skipping strict ingestion requirement (REQUIRE_INGEST=0)"
fi

header "Summary"
echo "pass=${PASS_COUNT} warn=${WARN_COUNT} fail=${FAIL_COUNT}"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
