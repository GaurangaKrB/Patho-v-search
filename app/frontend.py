import os
import glob

import requests
import streamlit as st
from PIL import Image


API = os.getenv("BACKEND_API_URL", "http://localhost:8000")
IMAGES_DIR = os.getenv("IMAGES_DIR", "data/panda_real/images")
THUMBS_DIR = os.path.join(IMAGES_DIR, "thumbnails")
PATCHES_DIR = os.path.join(IMAGES_DIR, "patches")

# Uniform display size for all thumbnails
THUMB_DISPLAY_SIZE = (300, 300)
PATCH_DISPLAY_SIZE = (256, 256)


st.set_page_config(page_title="Patho-v-search", layout="wide")
st.title("🔬 Patho-v-search")
st.caption("Visual similarity search for pathology slides and patches.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_json(path: str, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def post_json(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()





def get_thumbnail_uniform(slide_id: str):
    """Return a uniformly-sized PIL Image thumbnail, or None."""
    path = os.path.join(THUMBS_DIR, f"{slide_id}_thumb.png")
    if not os.path.isfile(path):
        return None
    img = Image.open(path).convert("RGB")
    # Create uniform-size canvas (white background) and paste centered
    canvas = Image.new("RGB", THUMB_DISPLAY_SIZE, (255, 255, 255))
    img.thumbnail(THUMB_DISPLAY_SIZE, Image.LANCZOS)
    offset_x = (THUMB_DISPLAY_SIZE[0] - img.width) // 2
    offset_y = (THUMB_DISPLAY_SIZE[1] - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def get_patch_image(slide_id: str, patch_idx: int):
    """Find and return a patch image matching the index, or None."""
    pattern = os.path.join(PATCHES_DIR, slide_id, f"patch_{patch_idx:04d}_*.png")
    matches = glob.glob(pattern)
    if matches:
        return Image.open(matches[0]).convert("RGB")
    return None


def find_closest_available_patch(slide_id: str, target_idx: int):
    """Find the closest extracted patch image to the target index.
    
    Returns (patch_idx, PIL.Image) or (None, None).
    """
    patch_dir = os.path.join(PATCHES_DIR, slide_id)
    if not os.path.isdir(patch_dir):
        return None, None
    
    available = []
    for f in os.listdir(patch_dir):
        if f.startswith("patch_") and f.endswith(".png"):
            try:
                idx = int(f.split("_")[1])
                available.append((idx, os.path.join(patch_dir, f)))
            except ValueError:
                pass
    
    if not available:
        return None, None
    
    # Find closest
    available.sort(key=lambda x: abs(x[0] - target_idx))
    best_idx, best_path = available[0]
    return best_idx, Image.open(best_path).convert("RGB")


def get_available_patch_indices(slide_id: str, exclude_background: bool = False):
    """Return sorted list of (index, filename) for available patches."""
    patch_dir = os.path.join(PATCHES_DIR, slide_id)
    if not os.path.isdir(patch_dir):
        return []

    # Load background metadata if filtering
    bg_set = set()
    if exclude_background:
        import json
        meta_path = os.path.join("data", "patch_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
                for k, v in meta.items():
                    if v.get("is_background", False):
                        bg_set.add(k)
            except Exception:
                pass

    result = []
    for f in sorted(os.listdir(patch_dir)):
        if f.startswith("patch_") and f.endswith(".png"):
            try:
                idx = int(f.split("_")[1])
                # Skip background patches
                if exclude_background and f"{slide_id}_{idx}" in bg_set:
                    continue
                # Extract x,y from filename
                parts = f.replace(".png", "").split("_")
                x = parts[2] if len(parts) > 2 else ""
                y = parts[3] if len(parts) > 3 else ""
                result.append((idx, x, y))
            except (ValueError, IndexError):
                pass
    return sorted(result)





# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Backend")
    try:
        health = get_json("/health")
        st.success("Backend OK ✅")
    except Exception as e:
        st.error(f"Backend unreachable at {API}")

    has_images = os.path.isdir(THUMBS_DIR) and len(os.listdir(THUMBS_DIR)) > 0
    n_thumbs = len(os.listdir(THUMBS_DIR)) if has_images else 0
    n_patches = sum(
        len([f for f in os.listdir(os.path.join(PATCHES_DIR, d)) if f.endswith(".png")])
        for d in os.listdir(PATCHES_DIR) if os.path.isdir(os.path.join(PATCHES_DIR, d))
    ) if os.path.isdir(PATCHES_DIR) else 0
    
    if has_images:
        st.success(f"📁 {n_thumbs} thumbnails, {n_patches} patches ✅")
    else:
        st.warning("No images found.")

    st.divider()
    st.subheader("Load Slides")
    limit = st.number_input("Max slides", min_value=1, max_value=5000, value=200, step=50)
    if st.button("🔄 Refresh slides"):
        try:
            resp = get_json("/slides", params={"limit": int(limit)})
            st.session_state["slide_ids"] = resp.get("slide_ids", [])
            st.write(f"Found **{resp.get('count', 0)}** slides")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()
    st.subheader("ℹ️ About Images")
    st.caption(
        "**Thumbnails** = bird's-eye view of the whole glass slide.\n\n"
        "**Patches** = 256×256 pixel crops from the slide. "
        f"We have extracted {n_patches} patch images. "
        "Patch search queries ALL embedded patches but can only display images for extracted ones."
    )

slide_ids = st.session_state.get("slide_ids", [])
if not slide_ids:
    st.info("👈 Click **Refresh slides** in the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🖼️ Slide Search (WSI-level)", "🔬 Patch Search (tile-level)"])

# ===========================================================================
# TAB 1: Slide Search with uniform thumbnails
# ===========================================================================
with tab1:
    st.subheader("Slide-level similarity")
    st.caption("Find whole slides that are most similar to a query slide.")

    col_q, col_opts = st.columns([2, 1])
    with col_q:
        slide_id = st.selectbox("Query slide", slide_ids)
    with col_opts:
        top_k = st.slider("Top-K results", min_value=1, max_value=50, value=5)

    # Show query slide info and thumbnail
    col_img, col_info = st.columns([1, 2])
    with col_img:
        query_thumb = get_thumbnail_uniform(slide_id)
        if query_thumb:
            st.image(query_thumb, caption="Query slide", use_container_width=False, width=250)
    with col_info:
        st.markdown(f"**Query:** `{slide_id}`")

    if st.button("🔍 Search similar slides"):
        try:
            resp = post_json("/search/slides", {"slide_id": slide_id, "top_k": int(top_k)})
            results = resp.get("results", [])

            if results:
                st.markdown(f"### Top {len(results)} similar slides")

                # Show results in uniform grid
                cols_per_row = min(5, len(results))
                for row_start in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = row_start + j
                        if idx >= len(results):
                            break
                        r = results[idx]
                        r_id = r.get("slide_id", "?")
                        r_score = r.get("score", 0)

                        with col:
                            thumb = get_thumbnail_uniform(r_id)
                            if thumb:
                                st.image(thumb, use_container_width=True)
                            else:
                                st.info("No thumbnail")
                            st.markdown(f"**#{idx+1}** Score: **{r_score:.4f}**")
                            st.caption(f"`{r_id[:20]}…`")
            else:
                st.warning("No results returned.")
        except Exception as e:
            st.error(f"Slide search failed: {e}")

# ===========================================================================
# TAB 2: Patch Search with actual patch images
# ===========================================================================
with tab2:
    st.subheader("Patch-level similarity")
    st.caption(
        "Find individual 256×256 tissue patches that look most similar. "
        "Select a patch WITH an extracted image to see visual comparisons."
    )

    col_q, col_p, col_k = st.columns([2, 1, 1])
    with col_q:
        patch_slide_id = st.selectbox("Source slide", slide_ids, key="patch_slide")
    with col_p:
        avail = get_available_patch_indices(patch_slide_id, exclude_background=True)
        if avail:
            # Format nicely: "Patch 50 (x3648, y5424)"
            options = [f"Patch {idx} ({x},{y})" for idx, x, y in avail]
            selected = st.selectbox("Pick a patch (with image)", options,
                help="These are patches with extracted images available.")
            patch_idx = avail[options.index(selected)][0]
        else:
            patch_idx = st.number_input("Patch index", min_value=0, value=0, step=1)
            st.caption("⚠️ No extracted images for this slide.")
    with col_k:
        patch_top_k = st.slider("Top-K", min_value=1, max_value=100, value=20, key="pk")

    col_toggles = st.columns(3)
    with col_toggles[0]:
        exclude_same = st.checkbox("Exclude query's WSI", value=True, help="Don't show patches from the same slide as the query.")
    with col_toggles[1]:
        exclude_bg = st.checkbox("Hide background patches", value=True, help="Filter out patches that are mostly white or black.")
    with col_toggles[2]:
        only_images = st.checkbox("Only patches with images", value=True, help="Only return results that have extracted patch PNGs available.")

    # Show query patch image
    query_patch = get_patch_image(patch_slide_id, patch_idx)

    col_pi, col_pinf = st.columns([1, 2])
    with col_pi:
        if query_patch:
            st.image(query_patch, caption=f"Query patch #{patch_idx}", width=256)
        else:
            st.warning(f"No image for patch {patch_idx}. Choose a patch from the dropdown.")
    with col_pinf:
        st.markdown(f"**Slide:** `{patch_slide_id}`")
        st.markdown(f"**Patch index:** {patch_idx}")

    if st.button("🔍 Search similar patches"):
        try:
            resp = post_json(
                "/search/patches",
                {
                    "slide_id": patch_slide_id,
                    "patch_idx": int(patch_idx),
                    "top_k": int(patch_top_k),
                    "exclude_same_slide": exclude_same,
                    "exclude_background": exclude_bg,
                    "only_with_images": only_images,
                },
            )
            results = resp.get("results", [])

            if results:
                # Separate results into those with images vs without
                with_img = []
                without_img = []
                for r in results:
                    r_slide = r.get("slide_id", "?")
                    r_pidx = r.get("patch_idx", -1)
                    patch_img = get_patch_image(r_slide, r_pidx)
                    if patch_img:
                        with_img.append((r, patch_img))
                    else:
                        without_img.append(r)

                st.markdown(f"### Results: {len(results)} similar patches")
                st.caption(
                    f"🟪 **{len(with_img)}** have extracted images | "
                    f"📝 **{len(without_img)}** show metadata only (no image extracted)"
                )

                # Show patches WITH images first (visual comparison)
                if with_img:
                    st.markdown("#### 🟪 Patches with images")
                    cols_per_row = 4
                    for row_start in range(0, len(with_img), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = row_start + j
                            if idx >= len(with_img):
                                break
                            r, patch_img = with_img[idx]
                            r_slide = r.get("slide_id", "?")
                            r_pidx = r.get("patch_idx", -1)
                            r_score = r.get("score", 0)

                            with col:
                                st.image(patch_img, use_container_width=True)
                                st.markdown(f"**Score: {r_score:.4f}**")
                                st.caption(f"`{r_slide[:16]}…` p{r_pidx}")

                # Show remaining results as text table
                if without_img:
                    st.markdown("#### 📝 Additional results (no extracted image)")
                    table_data = []
                    for r in without_img[:20]:  # Limit to 20 rows
                        r_slide = r.get("slide_id", "?")
                        r_pidx = r.get("patch_idx", -1)
                        r_score = r.get("score", 0)
                        r_x = r.get("x", "?")
                        r_y = r.get("y", "?")
                        table_data.append({
                            "Score": f"{r_score:.4f}",
                            "Slide": f"{r_slide[:20]}…",
                            "Patch": r_pidx,
                            "Coords": f"({r_x}, {r_y})",
                        })
                    st.dataframe(table_data, use_container_width=True)
            else:
                st.warning("No results returned.")
        except Exception as e:
            st.error(f"Patch search failed: {e}")
