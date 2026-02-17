import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import xxhash
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

try:
    import h5py
except ImportError:
    h5py = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DIM = 1536
SLIDES_COLLECTION = "Slides"
PATCHES_COLLECTION = "Patches"

VALID_AGG_STRATEGIES = ("mean", "max", "cls_token", "attention_weighted")


def stable_int_id(s: str) -> int:
    # Deterministic int ID so re-ingest overwrites instead of duplicating.
    return xxhash.xxh64(s).intdigest() & 0x7FFFFFFFFFFFFFFF


def to_float32_1d(x: Any) -> np.ndarray:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    arr = arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape={arr.shape}")
    return arr


def to_float32_2d(x: Any) -> np.ndarray:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    arr = arr.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")
    return arr


def infer_slide_id(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def list_files_with_extensions(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    exts = tuple(e.lower() for e in exts)
    files = []
    for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(exts):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def load_slide_pt(path: str) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """
    Returns (slide_id, slide_vec[1536], payload).
    Accepts:
      - tensor (1536,)
      - dict with a vector in common keys
    """
    obj = torch.load(path, map_location="cpu")

    slide_id = None
    payload: Dict[str, Any] = {"source_file": os.path.basename(path)}

    if isinstance(obj, dict):
        slide_id = obj.get("slide_id") or obj.get("wsi_id") or obj.get("id")
        # Try common keys for the slide-level embedding
        candidate_keys = [
            "slide_embedding", "embedding", "emb", "vector",
            "global_embedding", "slide_vec", "features"
        ]
        vec = None
        for k in candidate_keys:
            if k in obj:
                vec = obj[k]
                break
        if vec is None:
            raise KeyError(
                f"Could not find slide vector in {path}. "
                f"Available keys: {list(obj.keys())[:50]}"
            )

        slide_vec = to_float32_1d(vec)

        # Optional payload fields if present
        for k in ["patient_id", "diagnosis", "dataset", "label", "wsi_path"]:
            if k in obj and obj[k] is not None:
                payload[k] = obj[k]

    else:
        slide_vec = to_float32_1d(obj)

    if slide_id is None:
        slide_id = infer_slide_id(path)

    if slide_vec.shape[0] != DIM:
        raise ValueError(f"Slide vector dim mismatch in {path}: got {slide_vec.shape[0]}, expected {DIM}")

    payload["slide_id"] = slide_id
    return slide_id, slide_vec, payload


def load_patch_pt(path: str) -> Tuple[str, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns (slide_id, patch_vecs[N,1536], coords[N,2] or None, payload_template).
    Accepts:
      - tensor (N,1536)
      - dict with patch vectors in common keys
    """
    obj = torch.load(path, map_location="cpu")

    slide_id = None
    payload_template: Dict[str, Any] = {"source_file": os.path.basename(path)}

    coords = None

    if isinstance(obj, dict):
        slide_id = obj.get("slide_id") or obj.get("wsi_id") or obj.get("id")

        candidate_keys = [
            "patch_embeddings", "embeddings", "tile_embeddings", "patch_embeds",
            "patch_features", "features"
        ]
        mat = None
        for k in candidate_keys:
            if k in obj:
                mat = obj[k]
                break
        if mat is None:
            raise KeyError(
                f"Could not find patch matrix in {path}. "
                f"Available keys: {list(obj.keys())[:50]}"
            )

        patch_vecs = to_float32_2d(mat)

        for ck in ["coords", "coordinates", "xy", "patch_coords"]:
            if ck in obj and obj[ck] is not None:
                coords = obj[ck]
                break
        if coords is not None:
            coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
            coords = coords.astype(np.int32)
            if coords.ndim != 2 or coords.shape[1] != 2:
                coords = None  # keep ingestion robust

        # Optional payload propagation (useful for filtering later)
        for k in ["patient_id", "diagnosis", "dataset", "label"]:
            if k in obj and obj[k] is not None:
                payload_template[k] = obj[k]

    else:
        patch_vecs = to_float32_2d(obj)

    if slide_id is None:
        slide_id = infer_slide_id(path)

    if patch_vecs.shape[1] != DIM:
        raise ValueError(f"Patch dim mismatch in {path}: got {patch_vecs.shape[1]}, expected {DIM}")

    payload_template["slide_id"] = slide_id
    return slide_id, patch_vecs, coords, payload_template


def load_patch_h5(path: str) -> Tuple[str, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns (slide_id, patch_vecs[N,1536], coords[N,2] or None, payload_template)
    for HDF5 tiles exported by Prov-GigaPath.
    """
    if h5py is None:
        raise RuntimeError(
            "h5py is required to ingest .h5 files. Install with: pip install h5py"
        )

    slide_id = infer_slide_id(path)
    payload_template: Dict[str, Any] = {
        "source_file": os.path.basename(path),
        "slide_id": slide_id,
        "dataset": "prov-gigapath",
    }
    coords = None

    with h5py.File(path, "r") as f:
        mat = None
        for k in ("features", "embeddings", "patch_embeddings"):
            if k in f:
                mat = f[k]
                break
        if mat is None:
            for k in f.keys():
                ds = f[k]
                if getattr(ds, "ndim", None) == 2:
                    mat = ds
                    break
        if mat is None:
            raise KeyError(f"Could not find 2D embeddings dataset in {path}. Keys: {list(f.keys())[:20]}")

        patch_vecs = to_float32_2d(np.asarray(mat))

        for ck in ("coords", "coordinates", "xy", "patch_coords"):
            if ck in f:
                coords = np.asarray(f[ck]).astype(np.int32)
                break

        # Read metadata from HDF5 attributes (e.g. diagnosis, label, provider)
        for attr_key in ("diagnosis", "label", "patient_id", "provider",
                         "isup_grade", "dataset"):
            if attr_key in f.attrs:
                val = f.attrs[attr_key]
                # h5py may return bytes; decode to str
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                # numpy scalars aren't JSON-serializable; cast to Python native
                elif isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                payload_template[attr_key] = val
        # Also propagate diagnosis → label if label is missing
        if "label" not in payload_template and "diagnosis" in payload_template:
            payload_template["label"] = payload_template["diagnosis"]

    if patch_vecs.shape[1] != DIM:
        raise ValueError(f"Patch dim mismatch in {path}: got {patch_vecs.shape[1]}, expected {DIM}")

    if coords is not None and (coords.ndim != 2 or coords.shape[1] != 2):
        coords = None

    return slide_id, patch_vecs, coords, payload_template


def aggregate_patches(patch_vecs: np.ndarray, strategy: str) -> np.ndarray:
    """
    Aggregate N patch vectors into a single slide-level vector.

    Strategies:
      - mean:               Simple average (baseline).
      - max:                Element-wise max across patches.
      - cls_token:          Use the first patch vector (index 0) as a proxy CLS token.
      - attention_weighted: Compute per-patch L2-norm as pseudo-attention weight,
                            apply softmax, then weighted average.
    """
    if strategy == "mean":
        return patch_vecs.mean(axis=0).astype(np.float32)

    if strategy == "max":
        return patch_vecs.max(axis=0).astype(np.float32)

    if strategy == "cls_token":
        return patch_vecs[0].astype(np.float32)

    if strategy == "attention_weighted":
        # Per-patch L2 norm as pseudo importance score
        norms = np.linalg.norm(patch_vecs, axis=1)  # (N,)
        # Numerical stability for softmax
        norms = norms - norms.max()
        weights = np.exp(norms)
        weights = weights / weights.sum()
        return (weights[:, None] * patch_vecs).sum(axis=0).astype(np.float32)

    raise ValueError(f"Unknown aggregation strategy: {strategy!r}. Valid: {VALID_AGG_STRATEGIES}")


def ensure_collections(client: QdrantClient, recreate: bool = False, patches_on_disk: bool = True) -> None:
    existing = {c.name for c in client.get_collections().collections}

    def create_slides():
        client.recreate_collection(
            collection_name=SLIDES_COLLECTION,
            vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE, on_disk=False),
        )
        client.create_payload_index(SLIDES_COLLECTION, "slide_id", qm.PayloadSchemaType.KEYWORD)

    def create_patches():
        client.recreate_collection(
            collection_name=PATCHES_COLLECTION,
            vectors_config=qm.VectorParams(
                size=DIM,
                distance=qm.Distance.COSINE,
                on_disk=patches_on_disk,
            ),
            hnsw_config=qm.HnswConfigDiff(on_disk=True),
        )
        client.create_payload_index(PATCHES_COLLECTION, "slide_id", qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(PATCHES_COLLECTION, "label", qm.PayloadSchemaType.KEYWORD)

    if recreate:
        create_slides()
        create_patches()
        return

    if SLIDES_COLLECTION not in existing:
        create_slides()
    if PATCHES_COLLECTION not in existing:
        create_patches()


def ingest_slides(client: QdrantClient, slides_dir: str, limit: Optional[int] = None) -> int:
    files = list_files_with_extensions(slides_dir, (".pt",))
    if limit:
        files = files[:limit]

    if not files:
        print(f"No slide .pt files found in {slides_dir}")
        return 0

    points = []
    for path in tqdm(files, desc="Slides"):
        slide_id, slide_vec, payload = load_slide_pt(path)

        pid = stable_int_id(f"slide:{slide_id}")
        points.append(qm.PointStruct(id=pid, vector=slide_vec.tolist(), payload=payload))

        if len(points) >= 256:
            client.upsert(SLIDES_COLLECTION, points=points, wait=True)
            points = []

    if points:
        client.upsert(SLIDES_COLLECTION, points=points, wait=True)

    return len(files)


def ingest_patches(
    client: QdrantClient,
    patches_dir: str,
    batch_size: int = 512,
    limit_files: Optional[int] = None,
    limit_patches_per_file: Optional[int] = None,
    derive_slides_from_patches: bool = True,
    slide_agg_strategy: str = "mean",
) -> Tuple[int, int]:
    files = list_files_with_extensions(patches_dir, (".pt", ".h5"))
    if limit_files:
        files = files[:limit_files]

    if not files:
        print(f"No patch .pt/.h5 files found in {patches_dir}")
        return 0, 0

    if derive_slides_from_patches:
        log.warning(
            "Deriving slide vectors with '%s' pooling. "
            "For best quality, use Prov-GigaPath's LongNet slide encoder.",
            slide_agg_strategy,
        )

    total_files = 0
    derived_slide_count = 0
    slide_points: List[qm.PointStruct] = []

    for path in tqdm(files, desc="Patch files"):
        if path.endswith(".h5"):
            slide_id, patch_vecs, coords, payload_template = load_patch_h5(path)
        else:
            slide_id, patch_vecs, coords, payload_template = load_patch_pt(path)

        n = patch_vecs.shape[0]
        if limit_patches_per_file:
            n = min(n, limit_patches_per_file)
        if n <= 0:
            continue

        if derive_slides_from_patches:
            slide_vec = aggregate_patches(patch_vecs[:n], slide_agg_strategy)
            slide_payload = dict(payload_template)
            slide_payload["slide_id"] = slide_id
            slide_payload["derived_from"] = "patch_embeddings"
            slide_payload["agg_strategy"] = slide_agg_strategy
            slide_payload["n_patches"] = int(n)
            slide_pid = stable_int_id(f"slide:{slide_id}")
            slide_points.append(
                qm.PointStruct(
                    id=slide_pid,
                    vector=slide_vec.tolist(),
                    payload=slide_payload,
                )
            )
            derived_slide_count += 1
            if len(slide_points) >= 256:
                client.upsert(SLIDES_COLLECTION, points=slide_points, wait=True)
                slide_points = []

        # Batch-convert numpy slice to list-of-lists in a single C-level call
        # instead of calling .tolist() per vector inside the loop.
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vecs_as_lists = patch_vecs[start:end].tolist()  # one C-level conversion
            batch_points: List[qm.PointStruct] = []

            for local_idx, i in enumerate(range(start, end)):
                pid = stable_int_id(f"patch:{slide_id}:{i}")
                payload = dict(payload_template)
                payload["patch_idx"] = i

                if coords is not None and i < coords.shape[0]:
                    payload["x"] = int(coords[i, 0])
                    payload["y"] = int(coords[i, 1])

                batch_points.append(
                    qm.PointStruct(id=pid, vector=vecs_as_lists[local_idx], payload=payload)
                )

            client.upsert(PATCHES_COLLECTION, points=batch_points, wait=True)

        total_files += 1

    if slide_points:
        client.upsert(SLIDES_COLLECTION, points=slide_points, wait=True)

    return total_files, derived_slide_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant_url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant_api_key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument(
        "--qdrant_path",
        default=os.getenv("QDRANT_PATH"),
        help="Local folder for embedded Qdrant storage (no server needed). "
             "If set, --qdrant_url is ignored. Env: QDRANT_PATH",
    )
    ap.add_argument("--slides_dir", default="data/slides_embeddings")
    ap.add_argument("--patches_dir", default="data/patch_embeddings")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument("--limit_slides", type=int, default=None)
    ap.add_argument("--limit_patch_files", type=int, default=None)
    ap.add_argument("--limit_patches_per_file", type=int, default=None)
    ap.add_argument("--disable_derive_slides_from_patches", action="store_true")
    ap.add_argument(
        "--slide_agg_strategy",
        choices=VALID_AGG_STRATEGIES,
        default="mean",
        help="Strategy for deriving slide vectors from patch embeddings. "
             "Choices: mean, max, cls_token, attention_weighted. Default: mean.",
    )
    ap.add_argument(
        "--patches-on-disk",
        dest="patches_on_disk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = ap.parse_args()

    if args.qdrant_path:
        log.info("Using local embedded Qdrant at: %s", args.qdrant_path)
        client = QdrantClient(path=args.qdrant_path)
    else:
        qdrant_client_kwargs = {"url": args.qdrant_url}
        if args.qdrant_api_key:
            qdrant_client_kwargs["api_key"] = args.qdrant_api_key
        client = QdrantClient(**qdrant_client_kwargs)
    ensure_collections(client, recreate=args.recreate, patches_on_disk=args.patches_on_disk)

    n_slides = ingest_slides(client, args.slides_dir, limit=args.limit_slides)
    derive_slides_from_patches = (n_slides == 0) and (not args.disable_derive_slides_from_patches)
    n_patch_files, n_derived_slides = ingest_patches(
        client,
        args.patches_dir,
        batch_size=args.batch_size,
        limit_files=args.limit_patch_files,
        limit_patches_per_file=args.limit_patches_per_file,
        derive_slides_from_patches=derive_slides_from_patches,
        slide_agg_strategy=args.slide_agg_strategy,
    )

    print(
        "Done. "
        f"Slides ingested files={n_slides}, "
        f"patch files ingested={n_patch_files}, "
        f"slides_derived_from_patches={n_derived_slides}"
    )


if __name__ == "__main__":
    main()
