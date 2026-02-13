import os

import requests
import streamlit as st


API = os.getenv("BACKEND_API_URL", "http://localhost:8000")


st.set_page_config(page_title="Patho-v-search", layout="wide")
st.title("Patho-v-search (V1)")
st.caption("Slide similarity and patch similarity using Qdrant embeddings.")


def get_json(path: str, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def post_json(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


colA, colB = st.columns([1, 2])

with colA:
    st.subheader("Backend status")
    try:
        health = get_json("/health")
        st.success(f"Backend OK: {health}")
    except Exception as e:
        st.error(f"Backend not reachable at {API}. Error: {e}")

    st.subheader("Load slides list")
    limit = st.number_input("Max slides to list", min_value=1, max_value=5000, value=200, step=50)
    slides_resp = {"slide_ids": []}
    if st.button("Refresh slides"):
        try:
            slides_resp = get_json("/slides", params={"limit": int(limit)})
            st.write(f"Found {slides_resp.get('count', 0)} slides")
        except Exception as e:
            st.error(f"Failed to list slides: {e}")

    slide_ids = slides_resp.get("slide_ids", [])
    if not slide_ids:
        st.info("Click 'Refresh slides' after ingestion.")
    else:
        st.session_state["slide_ids"] = slide_ids

slide_ids = st.session_state.get("slide_ids", [])

tab1, tab2 = st.tabs(["Slide Search", "Patch Search"])

with tab1:
    st.subheader("Slide-level similarity")
    if not slide_ids:
        st.warning("No slides loaded. Refresh slides list.")
    else:
        slide_id = st.selectbox("Query slide", slide_ids)
        top_k = st.slider("Top-K", min_value=1, max_value=50, value=5)

        if st.button("Search similar slides"):
            try:
                resp = post_json("/search/slides", {"slide_id": slide_id, "top_k": int(top_k)})
                st.write(resp)
            except Exception as e:
                st.error(f"Slide search failed: {e}")

with tab2:
    st.subheader("Patch-level similarity")
    if not slide_ids:
        st.warning("No slides loaded. Refresh slides list.")
    else:
        slide_id = st.selectbox("Source slide (for patch)", slide_ids, key="patch_slide")
        patch_idx = st.number_input("Patch index", min_value=0, value=0, step=1)
        top_k = st.slider("Top-K patches", min_value=1, max_value=500, value=50)

        if st.button("Search similar patches"):
            try:
                resp = post_json(
                    "/search/patches",
                    {"slide_id": slide_id, "patch_idx": int(patch_idx), "top_k": int(top_k)},
                )
                st.write(resp)
            except Exception as e:
                st.error(f"Patch search failed: {e}")
