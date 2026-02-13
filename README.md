# Patho-v-search

Pathology embedding search (slides + patches) using Qdrant, FastAPI, and Streamlit.

## Run without Docker (local Qdrant binary)

```bash
cd /root/Patho-v-search
source .venv/bin/activate

# install and run local qdrant in background
bash scripts/qdrant_local.sh install
bash scripts/qdrant_local.sh start
bash scripts/qdrant_local.sh status
```

Qdrant data is stored in `data/qdrant_storage`.

## Dedicated Vast GPU plan (persistent embeddings on volume)

On Vast, attach a persistent volume and keep embeddings there so you do not
re-download them on every new instance.

Inside the instance:

```bash
cd /workspace/Patho-v-search
source .venv/bin/activate
bash scripts/vast_setup_volume.sh --start-qdrant
```

This wires:
- `data/slides_embeddings` -> `/data/patho-v-search/raw_embeddings/slides_embeddings`
- `data/patch_embeddings` -> `/data/patho-v-search/raw_embeddings/patch_embeddings`
- `data/qdrant_storage` -> `/data/patho-v-search/qdrant_storage`

Copy embeddings once to the volume (from your local machine):

```bash
vastai copy local:/path/to/slides_embeddings C.<INSTANCE_ID>:/data/patho-v-search/raw_embeddings/
vastai copy local:/path/to/patch_embeddings C.<INSTANCE_ID>:/data/patho-v-search/raw_embeddings/
```

Or download directly on the instance from Hugging Face:

```bash
bash scripts/download_panda_embeddings.sh --dest-root /data/patho-v-search/raw_embeddings
```

If you only have the HF page-style link:

`https://huggingface.co/.../blob/main/...zip`

replace `blob` with `resolve`:

`https://huggingface.co/.../resolve/main/...zip`

Then ingest:

```bash
python ingest.py --slides_dir data/slides_embeddings --patches_dir data/patch_embeddings
```

For a quick sample run on this dataset (`.h5` patch embeddings), use:

```bash
python ingest.py --recreate --no-patches-on-disk --limit_patch_files 5 --limit_patches_per_file 128
```

On future instance restarts (same attached volume), you only need:

```bash
bash scripts/vast_setup_volume.sh --start-qdrant
```

## Use remote Qdrant instead

Set these environment variables before running `ingest.py` or the backend:

```bash
export QDRANT_URL="https://<your-cluster-url>:6333"
export QDRANT_API_KEY="<your-api-key>"
```

## Run API + UI

```bash
# backend
uvicorn app.backend:app --host 0.0.0.0 --port 8000 --reload

# frontend (new terminal)
streamlit run app/frontend.py
```

Optional for frontend:

```bash
export BACKEND_API_URL="http://localhost:8000"
```

## Ingest embeddings

`ingest.py` supports:
- slide embeddings from `.pt`
- patch embeddings from `.pt` and `.h5` (Prov-GigaPath format)
- automatic slide vector derivation from patch embeddings when slide `.pt` files are absent

```bash
python ingest.py \
  --slides_dir data/slides_embeddings \
  --patches_dir data/patch_embeddings
```

## Smoke test (one command)

```bash
bash scripts/smoke_test.sh
```

`/health` on backend is part of the smoke test, so start backend first:

```bash
uvicorn app.backend:app --host 0.0.0.0 --port 8000 --reload
```

If you only want infra checks (and want to ignore empty ingestion), run:

```bash
REQUIRE_INGEST=0 bash scripts/smoke_test.sh
```
