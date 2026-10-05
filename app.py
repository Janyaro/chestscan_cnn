from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from inference import load_model, overlay, predict_with_cam, to_cache_array

st.set_page_config(page_title="ChestScan-CNN", page_icon="🫁", layout="wide")
BASE = Path(__file__).parent


@st.cache_resource
def get_model():
    p = BASE / "best_model.pt"
    return load_model(str(p)) if p.exists() else None


st.title("🫁 ChestScan-CNN")
st.caption("Pneumonia detection from paediatric chest X-rays: a research demo")
st.error("**Not a medical device.** Do not use it to make decisions about a patient. The model was trained "
         "on children aged 1 to 5 from one hospital, and its score depends partly on image size and source.")

tab1, tab2 = st.tabs(["Check an X-ray", "Model report"])

# ---------------- Tab 1 ----------------
with tab1:
    model = get_model()
    if model is None:
        st.warning("`best_model.pt` was not found next to `app.py`. Train the model or copy the file here.")
    else:
        up = st.file_uploader("Upload a chest X-ray (PNG or JPEG)", type=["png", "jpg", "jpeg"])
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                        help="Scores at or above this value are flagged as 'pneumonia likely'.")
        if up is not None:
            img = Image.open(up)
            if min(img.size) < 441:
                st.warning(f"This image is small ({img.size[0]}x{img.size[1]}). Training images were at least "
                           "441 pixels wide, so the score may be unreliable.")
            p, cam, crop = predict_with_cam(model, to_cache_array(img))
            c1, c2, c3 = st.columns(3)
            c1.image(crop, caption="What the model sees (centre crop)", clamp=True)
            c2.image(overlay(crop, cam), caption="Grad-CAM: evidence for pneumonia (no colour = none)", clamp=True)
            c3.metric("Pneumonia score", f"{p:.2f}")
            if p >= thr:
                c3.error("Flagged: pneumonia likely")
            else:
                c3.success("Not flagged")
            st.caption("The score is not a calibrated probability. Grad-CAM is coarse (7x7) and shows where the "
                       "network found evidence, not a diagnosis.")
        else:
            st.info("Upload a chest X-ray to see the score and the Grad-CAM map.")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Held-out test set (official test split, 624 images from 428 patients)")
    st.dataframe(pd.DataFrame({
        "Metric": ["ROC-AUC", "Sensitivity (pneumonia caught)", "Specificity (normal cleared)", "Accuracy"],
        "Value": [0.968, 0.990, 0.761, 0.904],
        "95% CI (patient bootstrap)": ["0.953 to 0.981", "0.978 to 1.000", "0.703 to 0.815", "n/a"]}),
        hide_index=True, width="stretch")
    st.write("Confusion matrix at threshold 0.5: 178 normal cleared, **56 normal wrongly flagged**, "
             "4 pneumonia missed, 386 pneumonia caught.")
    st.write("Validation split (744 images): ROC-AUC 0.998, sensitivity 0.940, specificity 1.000. "
             "The test split is harder, so the validation numbers should not be quoted as performance.")

    st.subheader("The size shortcut")
    st.markdown("In this dataset the image size alone predicts the class: a model that sees **only width, "
                "height and aspect ratio** reaches ROC-AUC **0.931 on validation and 0.895 on test**. "
                "The CNN's test AUC of 0.968 must be read against that baseline.")
    st.dataframe(pd.DataFrame({
        "Normal test images by width": ["Narrow (984 to 1611 px)", "Middle (1612 to 1949 px)", "Wide (1952 to 2752 px)"],
        "Images": [78, 78, 78],
        "Wrongly flagged": ["51%", "15%", "5%"]}), hide_index=True, width="stretch")
    st.markdown("- Inside each class, the score still correlates with image width (Spearman -0.40 for normal, "
                "-0.42 for pneumonia).\n"
                "- Test ROC-AUC inside groups of similar image shape: 0.989, 0.986, 0.936, 0.925 (aspect-ratio "
                "quartiles, narrowest to widest). Inside the width range 1265 to 1640 px (72 normal, 84 pneumonia) "
                "it is 0.890.\n"
                "- So the model learned signal beyond size, but part of the headline performance, and most of "
                "the false alarms, is tied to image size or scale.")

    st.subheader("Where does the model look?")
    img_path = BASE / "gradcam_examples.png"
    if img_path.exists():
        st.image(str(img_path), caption="Random test examples (not cherry-picked). Normal images that are "
                                        "cleared show no evidence by design.")
    st.markdown("Share of the Grad-CAM map in the outer border (49% of the area): **0.40** for correctly flagged "
                "pneumonia and **0.43** for false positives. Attention is not concentrated at the edges, but "
                "the map is coarse and cannot rule out subtle cues such as image sharpness.")

    st.subheader("Limitations")
    st.markdown("- One dataset from one hospital, paediatric patients only (ages 1 to 5).\n"
                "- Specificity on the test split is only 0.76. A threshold chosen on validation (sensitivity at least "
                "0.99) gave test specificity of 0.44, so validation does not transfer to test.\n"
                "- Binary label only (no bacterial or viral distinction), no radiologist review of the labels here.\n"
                "- Not tested on adults or on images from other sources.")
