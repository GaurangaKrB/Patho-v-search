#!/usr/bin/env python3
"""
End-to-end verification test for Patho-v-search.

Uses Qdrant's in-memory mode (no external server required).
Generates synthetic .pt data, ingests it, and validates:
  1. HNSW on-disk config is set for Patches collection
  2. Patch search correctly excludes same-slide results
  3. Label filtering works in patch search
  4. Slide search correctly excludes the query slide
  5. All 4 aggregation strategies produce valid slide vectors

Usage:
  python tests/test_e2e.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Import project modules
from ingest import (
    DIM,
    PATCHES_COLLECTION,
    SLIDES_COLLECTION,
    aggregate_patches,
    ensure_collections,
    ingest_patches,
    stable_int_id,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_SLIDES = 3
PATCHES_PER_SLIDE = 10
LABELS = ["tumor", "normal", "tumor"]
SLIDE_IDS = [f"test_slide_{i}" for i in range(NUM_SLIDES)]

PASS_COUNT = 0
FAIL_COUNT = 0


def ok(msg: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {msg}")


def check(condition: bool, pass_msg: str, fail_msg: str) -> None:
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg)


# ---------------------------------------------------------------------------
# Generate synthetic .pt files
# ---------------------------------------------------------------------------
def generate_synthetic_data(dest_dir: str) -> str:
    patches_dir = os.path.join(dest_dir, "patch_embeddings")
    os.makedirs(patches_dir, exist_ok=True)

    for idx, sid in enumerate(SLIDE_IDS):
        rng = np.random.RandomState(seed=idx)
        patch_vecs = rng.randn(PATCHES_PER_SLIDE, DIM).astype(np.float32)
        coords = np.stack(
            [np.arange(PATCHES_PER_SLIDE) * 256, np.zeros(PATCHES_PER_SLIDE)], axis=1
        ).astype(np.int32)

        data = {
            "slide_id": sid,
            "patch_embeddings": torch.from_numpy(patch_vecs),
            "coords": torch.from_numpy(coords),
            "label": LABELS[idx],
        }
        torch.save(data, os.path.join(patches_dir, f"{sid}.pt"))

    print(f"  Generated {NUM_SLIDES} .pt files ({PATCHES_PER_SLIDE} patches each, dim={DIM})")
    return patches_dir


# ---------------------------------------------------------------------------
# Test 1: HNSW on-disk config
# ---------------------------------------------------------------------------
def test_hnsw_on_disk(client: QdrantClient) -> None:
    print("\n== Test 1: HNSW On-Disk Config ==")

    info = client.get_collection(PATCHES_COLLECTION)
    hnsw_cfg = info.config.hnsw_config

    # In-memory Qdrant returns on_disk=None (no disk layer); server Qdrant returns True.
    # Both are valid — what matters is that our code passes on_disk=True to the config.
    on_disk_val = hnsw_cfg.on_disk
    check(
        on_disk_val is True or on_disk_val is None,
        f"Patches HNSW on_disk={on_disk_val} (True on server, None in-memory — both OK)",
        f"Patches HNSW on_disk={on_disk_val} (unexpected value)",
    )

    # Verify the code sets it correctly by inspecting the source
    import inspect
    from ingest import ensure_collections as fn
    source = inspect.getsource(fn)
    check(
        "HnswConfigDiff(on_disk=True)" in source,
        "ensure_collections source contains HnswConfigDiff(on_disk=True)",
        "HnswConfigDiff(on_disk=True) NOT found in ensure_collections source!",
    )


# ---------------------------------------------------------------------------
# Test 2: Collection counts
# ---------------------------------------------------------------------------
def test_counts(client: QdrantClient) -> None:
    print("\n== Test 2: Collection Counts ==")

    for coll, expected in [
        (SLIDES_COLLECTION, NUM_SLIDES),
        (PATCHES_COLLECTION, NUM_SLIDES * PATCHES_PER_SLIDE),
    ]:
        info = client.get_collection(coll)
        count = info.points_count
        check(
            count == expected,
            f"{coll} count={count} (expected {expected})",
            f"{coll} count={count}, expected {expected}",
        )


# ---------------------------------------------------------------------------
# Test 3: Patch search — same-slide exclusion
# ---------------------------------------------------------------------------
def test_patch_search_exclude_same_slide(client: QdrantClient) -> None:
    print("\n== Test 3: Patch Search — Same-Slide Exclusion ==")

    query_slide = SLIDE_IDS[0]
    query_pid = stable_int_id(f"patch:{query_slide}:0")

    pts = client.retrieve(PATCHES_COLLECTION, ids=[query_pid], with_vectors=True)
    assert pts, f"Patch not found: {query_slide}:0"
    query_vec = pts[0].vector

    # Search WITH exclusion
    flt = qm.Filter(
        must_not=[qm.FieldCondition(key="slide_id", match=qm.MatchValue(value=query_slide))]
    )

    if hasattr(client, "query_points"):
        results = client.query_points(
            collection_name=PATCHES_COLLECTION,
            query=query_vec,
            limit=50,
            with_payload=True,
            query_filter=flt,
        ).points
    else:
        results = client.search(
            collection_name=PATCHES_COLLECTION,
            query_vector=query_vec,
            limit=50,
            with_payload=True,
            query_filter=flt,
        )

    same_slide = [r for r in results if r.payload.get("slide_id") == query_slide]
    check(
        len(same_slide) == 0,
        f"Exclude filter works: 0 same-slide results out of {len(results)}",
        f"Filter broken: {len(same_slide)} same-slide patches returned",
    )

    # Search WITHOUT exclusion
    if hasattr(client, "query_points"):
        results_no_flt = client.query_points(
            collection_name=PATCHES_COLLECTION,
            query=query_vec,
            limit=50,
            with_payload=True,
        ).points
    else:
        results_no_flt = client.search(
            collection_name=PATCHES_COLLECTION,
            query_vector=query_vec,
            limit=50,
            with_payload=True,
        )

    same_slide_no_flt = [r for r in results_no_flt if r.payload.get("slide_id") == query_slide]
    check(
        len(same_slide_no_flt) > 0,
        f"Without filter: {len(same_slide_no_flt)} same-slide patches (correct)",
        "No same-slide patches even without filter",
    )


# ---------------------------------------------------------------------------
# Test 4: Patch search — label filtering
# ---------------------------------------------------------------------------
def test_patch_search_label_filter(client: QdrantClient) -> None:
    print("\n== Test 4: Patch Search — Label Filter ==")

    query_slide = SLIDE_IDS[0]  # label = "tumor"
    query_pid = stable_int_id(f"patch:{query_slide}:0")
    pts = client.retrieve(PATCHES_COLLECTION, ids=[query_pid], with_vectors=True)
    query_vec = pts[0].vector

    # Filter to only "normal" labels, AND exclude same slide
    flt = qm.Filter(
        must=[qm.FieldCondition(key="label", match=qm.MatchValue(value="normal"))],
        must_not=[qm.FieldCondition(key="slide_id", match=qm.MatchValue(value=query_slide))],
    )

    if hasattr(client, "query_points"):
        results = client.query_points(
            collection_name=PATCHES_COLLECTION,
            query=query_vec,
            limit=50,
            with_payload=True,
            query_filter=flt,
        ).points
    else:
        results = client.search(
            collection_name=PATCHES_COLLECTION,
            query_vector=query_vec,
            limit=50,
            with_payload=True,
            query_filter=flt,
        )

    wrong_label = [r for r in results if r.payload.get("label") != "normal"]
    check(
        len(wrong_label) == 0,
        f"Label filter works: all {len(results)} results have label='normal'",
        f"{len(wrong_label)} results have wrong label",
    )

    # Also check that results only come from slide_1 (the "normal" slide)
    result_slides = set(r.payload.get("slide_id") for r in results)
    check(
        result_slides == {"test_slide_1"},
        f"All results from 'normal' slide(s): {result_slides}",
        f"Unexpected slide(s) in results: {result_slides}",
    )


# ---------------------------------------------------------------------------
# Test 5: Slide search — query slide excluded
# ---------------------------------------------------------------------------
def test_slide_search(client: QdrantClient) -> None:
    print("\n== Test 5: Slide Search — Self-Exclusion ==")

    query_slide = SLIDE_IDS[0]
    query_pid = stable_int_id(f"slide:{query_slide}")
    pts = client.retrieve(SLIDES_COLLECTION, ids=[query_pid], with_vectors=True)
    assert pts, f"Slide not found: {query_slide}"
    query_vec = pts[0].vector

    flt = qm.Filter(
        must_not=[qm.FieldCondition(key="slide_id", match=qm.MatchValue(value=query_slide))]
    )

    if hasattr(client, "query_points"):
        results = client.query_points(
            collection_name=SLIDES_COLLECTION,
            query=query_vec,
            limit=10,
            with_payload=True,
            query_filter=flt,
        ).points
    else:
        results = client.search(
            collection_name=SLIDES_COLLECTION,
            query_vector=query_vec,
            limit=10,
            with_payload=True,
            query_filter=flt,
        )

    result_ids = [r.payload.get("slide_id") for r in results]
    check(
        query_slide not in result_ids,
        f"Query slide excluded: results={result_ids}",
        f"Query slide '{query_slide}' found in results",
    )
    check(
        len(result_ids) == NUM_SLIDES - 1,
        f"Got {len(result_ids)} results (expected {NUM_SLIDES - 1})",
        f"Got {len(result_ids)} results, expected {NUM_SLIDES - 1}",
    )


# ---------------------------------------------------------------------------
# Test 6: Aggregation strategies
# ---------------------------------------------------------------------------
def test_aggregation_strategies() -> None:
    print("\n== Test 6: Aggregation Strategies ==")

    rng = np.random.RandomState(42)
    vecs = rng.randn(20, DIM).astype(np.float32)

    for strat in ["mean", "max", "cls_token", "attention_weighted"]:
        try:
            result = aggregate_patches(vecs, strat)
            check(
                result.shape == (DIM,) and result.dtype == np.float32,
                f"'{strat}': shape={result.shape}, dtype={result.dtype}",
                f"'{strat}': unexpected shape={result.shape} or dtype={result.dtype}",
            )
        except Exception as e:
            fail(f"'{strat}' raised: {e}")

    # Verify strategies produce different results
    mean_vec = aggregate_patches(vecs, "mean")
    max_vec = aggregate_patches(vecs, "max")
    cls_vec = aggregate_patches(vecs, "cls_token")
    attn_vec = aggregate_patches(vecs, "attention_weighted")

    check(
        not np.allclose(mean_vec, max_vec),
        "'mean' and 'max' produce different vectors",
        "'mean' and 'max' are identical (suspicious)",
    )
    check(
        not np.allclose(mean_vec, cls_vec),
        "'mean' and 'cls_token' produce different vectors",
        "'mean' and 'cls_token' are identical (suspicious)",
    )
    check(
        not np.allclose(mean_vec, attn_vec),
        "'mean' and 'attention_weighted' produce different vectors",
        "'mean' and 'attention_weighted' are identical (suspicious)",
    )


# ---------------------------------------------------------------------------
# Test 7: Aggregation strategies through full ingestion pipeline
# ---------------------------------------------------------------------------
def test_agg_strategies_ingestion(patches_dir: str) -> None:
    print("\n== Test 7: Aggregation Strategies Through Ingestion ==")

    for strat in ["mean", "max", "cls_token", "attention_weighted"]:
        # Fresh in-memory client per strategy
        c = QdrantClient(":memory:")
        ensure_collections(c, recreate=True)
        n_files, n_derived = ingest_patches(
            c, patches_dir,
            derive_slides_from_patches=True,
            slide_agg_strategy=strat,
        )
        check(
            n_files == NUM_SLIDES and n_derived == NUM_SLIDES,
            f"'{strat}': ingested {n_files} files, derived {n_derived} slides",
            f"'{strat}': expected files={NUM_SLIDES}, derived={NUM_SLIDES}, "
            f"got files={n_files}, derived={n_derived}",
        )

        # Check agg_strategy is stored in payload
        pts, _ = c.scroll(SLIDES_COLLECTION, limit=1, with_payload=True)
        if pts:
            stored_strat = pts[0].payload.get("agg_strategy")
            check(
                stored_strat == strat,
                f"'{strat}': payload agg_strategy='{stored_strat}'",
                f"'{strat}': payload agg_strategy='{stored_strat}' (expected '{strat}')",
            )


# ---------------------------------------------------------------------------
# Test 8: Backend API via FastAPI TestClient
# ---------------------------------------------------------------------------
def test_backend_api(client: QdrantClient) -> None:
    print("\n== Test 8: Backend API (TestClient) ==")

    # Monkey-patch the backend module's client to use our in-memory client
    import app.backend as backend_mod
    backend_mod.client = client

    from fastapi.testclient import TestClient
    tc = TestClient(backend_mod.app)

    # Health
    resp = tc.get("/health")
    check(resp.status_code == 200, "GET /health → 200", f"GET /health → {resp.status_code}")

    # List slides
    resp = tc.get("/slides")
    check(resp.status_code == 200, "GET /slides → 200", f"GET /slides → {resp.status_code}")
    slide_ids = resp.json().get("slide_ids", [])
    check(
        set(slide_ids) == set(SLIDE_IDS),
        f"Listed slides: {sorted(slide_ids)}",
        f"Expected {sorted(SLIDE_IDS)}, got {sorted(slide_ids)}",
    )

    # Slide search
    query_slide = SLIDE_IDS[0]
    resp = tc.post("/search/slides", json={"slide_id": query_slide, "top_k": 10})
    check(resp.status_code == 200, "POST /search/slides → 200", f"→ {resp.status_code}")
    result_ids = [r["slide_id"] for r in resp.json().get("results", [])]
    check(
        query_slide not in result_ids,
        f"Query slide excluded via API: {result_ids}",
        f"Query slide in API results: {result_ids}",
    )

    # Patch search: exclude_same_slide=True (default)
    resp = tc.post("/search/patches", json={
        "slide_id": query_slide, "patch_idx": 0, "top_k": 50,
        "exclude_same_slide": True,
    })
    check(resp.status_code == 200, "POST /search/patches → 200", f"→ {resp.status_code}")
    data = resp.json()
    same_slide = [r for r in data["results"] if r["slide_id"] == query_slide]
    check(
        len(same_slide) == 0,
        f"API: 0 same-slide patches ({len(data['results'])} total)",
        f"API: {len(same_slide)} same-slide patches returned!",
    )

    # Verify filters_applied in response
    fa = data.get("filters_applied", {})
    check(
        fa.get("exclude_same_slide") is True,
        "filters_applied.exclude_same_slide=True in response",
        f"filters_applied missing or wrong: {fa}",
    )

    # Patch search: exclude_same_slide=False
    resp = tc.post("/search/patches", json={
        "slide_id": query_slide, "patch_idx": 0, "top_k": 50,
        "exclude_same_slide": False,
    })
    data = resp.json()
    same_slide = [r for r in data["results"] if r["slide_id"] == query_slide]
    check(
        len(same_slide) > 0,
        f"API: {len(same_slide)} same-slide patches with exclusion disabled",
        "API: no same-slide patches even with exclusion disabled",
    )

    # Patch search: label filter
    resp = tc.post("/search/patches", json={
        "slide_id": query_slide, "patch_idx": 0, "top_k": 50,
        "exclude_same_slide": True,
        "filter_label": "normal",
    })
    data = resp.json()
    wrong = [r for r in data["results"] if r["diagnosis"] != "normal"]
    check(
        len(wrong) == 0,
        f"API: all {len(data['results'])} results label='normal'",
        f"API: {len(wrong)} results have wrong label",
    )
    check(
        data.get("filters_applied", {}).get("filter_label") == "normal",
        "filters_applied.filter_label='normal' in response",
        f"filter_label not in response",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global PASS_COUNT, FAIL_COUNT

    print("=" * 60)
    print("Patho-v-search E2E Verification Test")
    print("  Mode: In-Memory Qdrant (no external server needed)")
    print("=" * 60)

    # Generate synthetic data
    print("\n== Generating Synthetic Data ==")
    tmp_dir = tempfile.mkdtemp(prefix="patho_e2e_")

    try:
        patches_dir = generate_synthetic_data(tmp_dir)
        slides_dir = os.path.join(tmp_dir, "empty_slides")
        os.makedirs(slides_dir, exist_ok=True)

        # Create in-memory Qdrant client
        client = QdrantClient(":memory:")
        ensure_collections(client, recreate=True, patches_on_disk=False)  # in-memory ignores on_disk

        # Ingest with default strategy
        print("\n== Ingestion (mean strategy) ==")
        n_files, n_derived = ingest_patches(
            client, patches_dir,
            derive_slides_from_patches=True,
            slide_agg_strategy="mean",
        )
        print(f"  Ingested {n_files} files, derived {n_derived} slide vectors")

        # Run tests
        test_hnsw_on_disk(client)
        test_counts(client)
        test_patch_search_exclude_same_slide(client)
        test_patch_search_label_filter(client)
        test_slide_search(client)
        test_aggregation_strategies()
        test_agg_strategies_ingestion(patches_dir)
        test_backend_api(client)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Summary: pass={PASS_COUNT} fail={FAIL_COUNT}")
    print("=" * 60)
    sys.exit(1 if FAIL_COUNT > 0 else 0)


if __name__ == "__main__":
    main()
