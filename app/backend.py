import os
from typing import Any, Dict, List, Optional

import xxhash
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH")  # local folder for embedded Qdrant (no server)
SLIDES_COLLECTION = "Slides"
PATCHES_COLLECTION = "Patches"


def stable_int_id(s: str) -> int:
    return xxhash.xxh64(s).intdigest() & 0x7FFFFFFFFFFFFFFF


if QDRANT_PATH:
    client = QdrantClient(path=QDRANT_PATH)
else:
    qdrant_client_kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        qdrant_client_kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(**qdrant_client_kwargs)

app = FastAPI(title="Patho-v-search Backend", version="0.1.0")


class SlideSearchRequest(BaseModel):
    slide_id: str
    top_k: int = Field(default=5, ge=1, le=50)


class PatchSearchRequest(BaseModel):
    slide_id: str
    patch_idx: int = Field(default=0, ge=0)
    top_k: int = Field(default=50, ge=1, le=500)
    exclude_same_slide: bool = Field(
        default=True,
        description="If True, exclude patches from the same WSI as the query patch.",
    )
    filter_label: Optional[str] = Field(
        default=None,
        description="If set, only return patches whose 'label' payload matches this value.",
    )


def get_slide_vector(slide_id: str) -> List[float]:
    pid = stable_int_id(f"slide:{slide_id}")
    pts = client.retrieve(
        collection_name=SLIDES_COLLECTION,
        ids=[pid],
        with_vectors=True,
        with_payload=True,
    )
    if not pts:
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_id}")
    vec = pts[0].vector
    if vec is None:
        raise HTTPException(status_code=500, detail="Slide vector missing in DB")
    return vec


def get_patch_vector(slide_id: str, patch_idx: int) -> List[float]:
    pid = stable_int_id(f"patch:{slide_id}:{patch_idx}")
    pts = client.retrieve(
        collection_name=PATCHES_COLLECTION,
        ids=[pid],
        with_vectors=True,
        with_payload=True,
    )
    if not pts:
        raise HTTPException(status_code=404, detail=f"Patch not found: slide_id={slide_id}, patch_idx={patch_idx}")
    vec = pts[0].vector
    if vec is None:
        raise HTTPException(status_code=500, detail="Patch vector missing in DB")
    return vec


def vector_search(
    collection_name: str,
    query_vector: List[float],
    limit: int,
    query_filter: Optional[qm.Filter] = None,
):
    # qdrant-client >=1.10 uses query_points; older versions expose search.
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return resp.points

    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "qdrant_url": QDRANT_URL}


@app.get("/slides")
def list_slides(limit: int = 200) -> Dict[str, Any]:
    limit = max(1, min(limit, 5000))
    points, _ = client.scroll(
        collection_name=SLIDES_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    slide_ids = []
    for p in points:
        sid = p.payload.get("slide_id") if p.payload else None
        if sid:
            slide_ids.append(sid)
    slide_ids = sorted(set(slide_ids))
    return {"count": len(slide_ids), "slide_ids": slide_ids}


@app.post("/search/slides")
def search_similar_slides(req: SlideSearchRequest) -> Dict[str, Any]:
    query_vec = get_slide_vector(req.slide_id)

    flt = qm.Filter(
        must_not=[qm.FieldCondition(key="slide_id", match=qm.MatchValue(value=req.slide_id))]
    )

    hits = vector_search(
        collection_name=SLIDES_COLLECTION,
        query_vector=query_vec,
        limit=req.top_k,
        query_filter=flt,
    )

    results = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "slide_id": payload.get("slide_id"),
                "score": float(h.score),
                "diagnosis": payload.get("diagnosis") or payload.get("label"),
                "patient_id": payload.get("patient_id"),
                "dataset": payload.get("dataset"),
            }
        )

    return {"query_slide_id": req.slide_id, "results": results}


@app.post("/search/patches")
def search_similar_patches(req: PatchSearchRequest) -> Dict[str, Any]:
    query_vec = get_patch_vector(req.slide_id, req.patch_idx)

    # Build filter: exclude same slide, optionally require specific label
    must_not_conditions = []
    must_conditions = []

    if req.exclude_same_slide:
        must_not_conditions.append(
            qm.FieldCondition(key="slide_id", match=qm.MatchValue(value=req.slide_id))
        )

    if req.filter_label is not None:
        must_conditions.append(
            qm.FieldCondition(key="label", match=qm.MatchValue(value=req.filter_label))
        )

    query_filter = None
    if must_not_conditions or must_conditions:
        query_filter = qm.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

    hits = vector_search(
        collection_name=PATCHES_COLLECTION,
        query_vector=query_vec,
        limit=req.top_k,
        query_filter=query_filter,
    )

    results = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "slide_id": payload.get("slide_id"),
                "patch_idx": payload.get("patch_idx"),
                "x": payload.get("x"),
                "y": payload.get("y"),
                "score": float(h.score),
                "diagnosis": payload.get("diagnosis") or payload.get("label"),
            }
        )

    return {
        "query": {"slide_id": req.slide_id, "patch_idx": req.patch_idx},
        "filters_applied": {
            "exclude_same_slide": req.exclude_same_slide,
            "filter_label": req.filter_label,
        },
        "results": results,
    }
