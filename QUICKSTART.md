# Quickstart Guide

Everything you need to run Patho-v-search from scratch.

---

## 1. Clone & Install

```bash
git clone https://github.com/GaurangaKrB/Patho-v-search.git
cd Patho-v-search

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Run Tests (No Server Needed)

This spins up an in-memory Qdrant, generates synthetic data, ingests it, and validates all features:

```bash
python3 tests/test_e2e.py
```

You should see **36/36 PASS** covering:
- HNSW on-disk config
- Patch search (same-slide exclusion, label filtering)
- Slide search (self-exclusion)
- All 4 aggregation strategies (`mean`, `max`, `cls_token`, `attention_weighted`)
- Backend API endpoints

---

## 3. Generate Synthetic Demo Data

```bash
python3 -c "
import os, torch, numpy as np
os.makedirs('data/patch_embeddings', exist_ok=True)
for i in range(5):
    rng = np.random.RandomState(seed=i+100)
    n_patches = rng.randint(8, 20)
    data = {
        'slide_id': f'demo_slide_{i}',
        'patch_embeddings': torch.from_numpy(rng.randn(n_patches, 1536).astype('f')),
        'coords': torch.from_numpy(np.stack([np.arange(n_patches)*256, np.zeros(n_patches)], 1).astype('i')),
        'label': ['tumor', 'normal', 'stroma', 'necrosis', 'tumor'][i],
    }
    torch.save(data, f'data/patch_embeddings/demo_slide_{i}.pt')
    print(f'  Created demo_slide_{i}.pt  ({n_patches} patches, label={data[\"label\"]})')
"
```

---

## 4. Ingest Into Qdrant

### Option A: Local embedded Qdrant (no Docker, simplest)

```bash
mkdir -p data/empty_slides

python3 ingest.py \
  --qdrant_path data/qdrant_local \
  --patches_dir data/patch_embeddings \
  --slides_dir data/empty_slides \
  --recreate \
  --slide_agg_strategy attention_weighted
```

### Option B: Qdrant server via Docker

```bash
docker compose up -d          # starts Qdrant on localhost:6333

python3 ingest.py \
  --qdrant_url http://localhost:6333 \
  --patches_dir data/patch_embeddings \
  --slides_dir data/empty_slides \
  --recreate \
  --slide_agg_strategy attention_weighted
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--qdrant_path PATH` | Use embedded Qdrant (no server). Overrides `--qdrant_url` |
| `--qdrant_url URL` | Qdrant server URL (default: `http://localhost:6333`) |
| `--slide_agg_strategy` | `mean` / `max` / `cls_token` / `attention_weighted` |
| `--recreate` | Drop and recreate collections before ingesting |
| `--limit_patch_files N` | Only ingest first N slide files |
| `--limit_patches_per_file N` | Cap patches per file |
| `--no-patches-on-disk` | Keep patch vectors in RAM instead of disk |

---

## 5. Start the Backend API

```bash
# If using local embedded Qdrant:
QDRANT_PATH=data/qdrant_local uvicorn app.backend:app --host 0.0.0.0 --port 8000

# If using Qdrant server:
uvicorn app.backend:app --host 0.0.0.0 --port 8000
```

Verify: open http://localhost:8000/health → `{"ok": true, ...}`

---

## 6. Start the Streamlit Frontend

In a **second terminal**:

```bash
source .venv/bin/activate

# If using local embedded Qdrant:
QDRANT_PATH=data/qdrant_local streamlit run app/frontend.py --server.port 8501

# If using Qdrant server:
streamlit run app/frontend.py --server.port 8501
```

Open http://localhost:8501 in your browser.

---

## 7. Use the UI

1. Click **"Refresh slides"** → should show your ingested slides
2. **Slide Search** tab → pick a query slide, set Top-K, click "Search similar slides"
3. **Patch Search** tab → pick a slide + patch index, click "Search similar patches"

### What to look for

- Query slide is **excluded** from its own results
- Patch results come from **different slides** (same-slide exclusion)
- `filters_applied` in JSON response confirms filtering is active
- Each result has `slide_id`, `score`, `diagnosis`, and patch coordinates

---

## 8. cURL Examples (API directly)

```bash
# List slides
curl http://localhost:8000/slides | python3 -m json.tool

# Slide search
curl -X POST http://localhost:8000/search/slides \
  -H "Content-Type: application/json" \
  -d '{"slide_id": "demo_slide_0", "top_k": 5}'

# Patch search (same-slide excluded, label filter)
curl -X POST http://localhost:8000/search/patches \
  -H "Content-Type: application/json" \
  -d '{"slide_id": "demo_slide_0", "patch_idx": 0, "top_k": 20, "exclude_same_slide": true, "filter_label": "normal"}'
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `QDRANT_PATH` | Path to local embedded Qdrant storage (no server needed) |
| `QDRANT_URL` | Qdrant server URL (default `http://localhost:6333`) |
| `QDRANT_API_KEY` | API key for Qdrant Cloud |
