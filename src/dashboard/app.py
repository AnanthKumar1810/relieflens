from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.dashboard.core import (
    corrected_mask,
    mask_to_pixel_geojson,
    registration_sanity_check,
    sha256_file,
    utc_timestamp,
)
from src.dashboard.inference import CLASS_NAMES, checkpoint_metadata, load_checkpoint, overlay_damage, predict_pair

st.set_page_config(page_title="ReliefLens", layout="wide")
st.title("ReliefLens — Calibrated Disaster-Damage Triage")
st.caption("Decision support for human review. Not an autonomous emergency-response system.")

with st.sidebar:
    st.header("Review")
    reviewer = st.text_input("Analyst / reviewer name", value="")
    checkpoint = st.text_input("Checkpoint", value="artifacts/best.pt")
    device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = st.selectbox("Device", device_options, index=0)
    uploaded_pre = st.file_uploader("Before-disaster image", type=["png", "jpg", "jpeg"], key="pre")
    uploaded_post = st.file_uploader("After-disaster image", type=["png", "jpg", "jpeg"], key="post")
    max_shift = st.number_input("Registration warning threshold (pixels)", min_value=1.0, value=12.0, step=1.0)
    min_response = st.number_input("Registration phase-response threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    low_confidence = st.number_input("Low-confidence warning threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    run = st.button("Run triage", type="primary")


@st.cache_resource
def cached_model(path: str, selected_device: str):
    return load_checkpoint(path, selected_device)


if run:
    if not uploaded_pre or not uploaded_post:
        st.error("Upload both a before and after image.")
        st.stop()
    if not reviewer.strip():
        st.error("Enter an analyst/reviewer name before exporting an audit record.")
        st.stop()
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint}. Train the model first with the README command.")
        st.stop()

    pre = np.array(Image.open(io.BytesIO(uploaded_pre.read())).convert("RGB"))
    post = np.array(Image.open(io.BytesIO(uploaded_post.read())).convert("RGB"))
    if pre.shape != post.shape:
        st.error(f"Before/after images must have identical dimensions: got {pre.shape} vs {post.shape}")
        st.stop()

    registration = registration_sanity_check(
        pre,
        post,
        max_shift_pixels=float(max_shift),
        min_response=float(min_response),
    )
    if registration["ok"]:
        st.success(
            f"Registration sanity check passed: shift={registration['shift_magnitude_pixels']:.2f}px, "
            f"phase response={registration['response']:.3f}"
        )
    else:
        st.warning(
            f"Registration sanity check is weak: estimated shift={registration['shift_magnitude_pixels']:.2f}px, "
            f"phase response={registration['response']:.3f}. Treat change predictions cautiously."
        )

    model, cfg = cached_model(str(checkpoint_path), device)
    metadata = checkpoint_metadata(checkpoint_path, device="cpu")
    temperature = float(metadata["temperature"])
    result = predict_pair(
        model,
        pre,
        post,
        device=device,
        normalize=cfg.get("image", {}).get("normalize", "imagenet"),
        temperature=temperature,
        abstain_confidence_threshold=float(low_confidence),
    )

    review_timestamp = utc_timestamp()
    checkpoint_hash = sha256_file(checkpoint_path)

    col1, col2, col3 = st.columns(3)
    col1.image(pre, caption="Before", use_container_width=True)
    col2.image(post, caption="After", use_container_width=True)
    col3.image(overlay_damage(post, result["damage_mask"]), caption="Predicted damage", use_container_width=True)

    st.subheader("Model provenance")
    provenance = pd.DataFrame(
        [
            {
                "analyst": reviewer,
                "reviewed_at_utc": review_timestamp,
                "model_type": metadata["model_type"],
                "checkpoint_epoch": metadata["epoch"],
                "checkpoint_sha256": checkpoint_hash,
                "calibration_temperature": result["temperature"],
                "validation_damage_macro_f1": metadata["damage_macro_f1"],
            }
        ]
    )
    st.dataframe(provenance, hide_index=True, use_container_width=True)

    counts = np.bincount(result["damage_mask"].ravel(), minlength=5)
    summary = pd.DataFrame({"severity": [CLASS_NAMES[i] for i in range(5)], "pixels": counts})
    st.subheader("Triage summary")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.metric("Review priority", f"{result['priority']:.4f}")
    st.metric("Calibration temperature", f"{result['temperature']:.3f}")

    max_conf = float(result["mean_confidence"])
    if result["abstain"] or max_conf < float(low_confidence):
        st.warning(f"ReliefLens recommends human review / abstention: mean confidence={max_conf:.3f}, normalized entropy={result['normalized_entropy']:.3f}.")
    else:
        st.info(f"Mean pixel confidence: {max_conf:.3f}; normalized entropy: {result['normalized_entropy']:.3f}. Confidence is a calibrated score, not a guarantee.")

    st.subheader("Building-level summary")
    building_df = pd.DataFrame(result["buildings"])
    if building_df.empty:
        building_df = pd.DataFrame(columns=["building_id", "area_pixels", "centroid_x", "centroid_y", "damage_class", "confidence"])
    building_df["corrected_damage_class"] = building_df.get("damage_class", pd.Series(dtype=str))
    st.dataframe(building_df, hide_index=True, use_container_width=True)

    st.subheader("Human correction")
    override = st.selectbox("Override entire tile severity", ["no override", "no damage", "minor", "major", "destroyed"])
    building_id = None
    building_override_class = None
    if not building_df.empty:
        ids = ["no building correction"] + [str(int(v)) for v in building_df["building_id"]]
        selected = st.selectbox("Select one connected building for correction", ids)
        if selected != "no building correction":
            building_id = int(selected)
            building_override_class = st.selectbox(
                "Correct selected building to",
                ["no damage", "minor", "major", "destroyed"],
                key="building_override_class",
            )
            building_df.loc[building_df["building_id"] == building_id, "corrected_damage_class"] = building_override_class

    corrected = corrected_mask(
        result["building_mask"],
        result["damage_mask"],
        tile_override=None if override == "no override" else override,
        building_override_id=building_id,
        building_override_class=building_override_class,
    )
    if override != "no override" or building_id is not None:
        st.image(overlay_damage(post, corrected), caption="Corrected mask", use_container_width=True)

    correction_payload = {
        "review": {
            "analyst": reviewer,
            "reviewed_at_utc": review_timestamp,
            "before_filename": uploaded_pre.name,
            "after_filename": uploaded_post.name,
            "image_shape": list(post.shape),
        },
        "model": {
            "model_type": metadata["model_type"],
            "checkpoint_epoch": metadata["epoch"],
            "checkpoint_sha256": checkpoint_hash,
            "calibration_temperature": result["temperature"],
            "validation_damage_macro_f1": metadata["damage_macro_f1"],
        },
        "registration": registration,
        "triage_priority": result["priority"],
        "uncertainty": {"mean_confidence": result["mean_confidence"], "normalized_entropy": result["normalized_entropy"], "abstain": result["abstain"]},
        "tile_override": None if override == "no override" else override,
        "building_override": {
            "building_id": building_id,
            "class": building_override_class,
        },
        "buildings": building_df.to_dict(orient="records"),
    }

    st.download_button(
        "Export audit/correction JSON",
        data=json.dumps(correction_payload, indent=2),
        file_name="relieflens_correction.json",
        mime="application/json",
    )

    corrected_png = io.BytesIO()
    Image.fromarray(corrected, mode="L").save(corrected_png, format="PNG")
    corrected_png.seek(0)
    st.download_button(
        "Export corrected mask PNG",
        data=corrected_png,
        file_name="relieflens_corrected_mask.png",
        mime="image/png",
    )

    geojson = mask_to_pixel_geojson(corrected, CLASS_NAMES)
    st.download_button(
        "Export corrected pixel-space GeoJSON",
        data=json.dumps(geojson, indent=2),
        file_name="relieflens_corrected_pixel_mask.geojson",
        mime="application/geo+json",
    )

    csv_data = building_df.to_csv(index=False)
    st.download_button(
        "Export building CSV",
        data=csv_data,
        file_name="relieflens_buildings.csv",
        mime="text/csv",
    )

    conf = result["damage_probs"].max(axis=0).ravel()
    st.subheader("Confidence")
    hist, edges = np.histogram(conf, bins=np.linspace(0.0, 1.0, 11))
    centers = (edges[:-1] + edges[1:]) / 2.0
    st.bar_chart(pd.DataFrame({"confidence": centers, "pixels": hist}).set_index("confidence"))

    if st.checkbox("Show raw class mask"):
        st.image(result["damage_mask"], caption="0 background · 1 no damage · 2 minor · 3 major · 4 destroyed", use_container_width=True)
