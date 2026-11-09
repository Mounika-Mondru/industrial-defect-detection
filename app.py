import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, #dceafe 0%, transparent 35%),
        linear-gradient(135deg, #f5f7ff 0%, #eaf1ff 55%, #f8faff 100%);
    color: #172554;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    text-align: center;
    padding: 25px 10px 28px 10px;
}

.hero h1 {
    color: #173b75;
    font-size: 2.5rem;
    font-weight: 850;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.hero p {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 0;
}

.section-title {
    color: #234779;
    font-size: 1.25rem;
    font-weight: 750;
    margin-bottom: 15px;
}

.upload-area {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(148,163,184,0.35);
    border-radius: 18px;
    padding: 20px;
    min-height: 150px;
}

.result-area {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(148,163,184,0.35);
    border-radius: 18px;
    padding: 20px;
    min-height: 150px;
}

.stButton > button {
    width: 100%;
    height: 3.1rem;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #6d5dfc);
    color: white;
    font-size: 1rem;
    font-weight: 750;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #5845e8);
    color: white;
    transform: translateY(-2px);
}

.normal-result {
    background: #dcfce7;
    border: 1px solid #86efac;
    color: #166534;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 800;
}

.defect-result {
    background: #fee2e2;
    border: 1px solid #fca5a5;
    color: #991b1b;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 800;
}

.info-card {
    background: rgba(255,255,255,0.6);
    border: 1px solid #dbe5f5;
    border-radius: 14px;
    padding: 15px;
    text-align: center;
    color: #475569;
    font-size: 0.9rem;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    padding-top: 25px;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.45);
    border-radius: 12px;
    padding: 8px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.65);
    border: 1px solid #dbe5f5;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINED CNN ----------------
MODEL_PATH = "/tmp/industrial_defect_cnn.keras"

MODEL_URL = (
    "https://github.com/Mounika-Mondru/"
    "industrial-defect-detection/releases/download/"
    "v1.0/industrial_defect_cnn.keras"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Loading trained CNN model..."):
            response = requests.get(MODEL_URL, timeout=300)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as file:
                file.write(response.content)

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🔍 CNN-Based Industrial Defect Detection</h1>
    <p>Smart visual inspection of metal nuts using Deep Learning</p>
</div>
""", unsafe_allow_html=True)


# ---------------- PROJECT INFORMATION ----------------
info1, info2, info3 = st.columns(3)

with info1:
    st.markdown(
        '<div class="info-card">🧠 <b>Model</b><br>TensorFlow / Keras</div>',
        unsafe_allow_html=True
    )

with info2:
    st.markdown(
        '<div class="info-card">📊 <b>Dataset</b><br>MVTec AD – Metal Nut</div>',
        unsafe_allow_html=True
    )

with info3:
    st.markdown(
        '<div class="info-card">🖼️ <b>Input Size</b><br>128 × 128 RGB</div>',
        unsafe_allow_html=True
    )

st.write("")


# ---------------- IMAGE INPUT AND RESULT ----------------
left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<div class="section-title">📤 1. Upload Input Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a metal-nut image",
        type=["jpg", "jpeg", "png"]
    )

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Metal Nut",
            width=300
        )
    else:
        st.info("Upload a metal-nut image to start inspection.")


with right:
    st.markdown(
        '<div class="section-title">🔬 2. Prediction Result</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:
        st.info("Your inspection result will appear here.")

    else:
        if st.button("🔍 Inspect Image", type="primary"):
            try:
                model = load_model()

                # Preprocessing
                resized_image = image.resize((128, 128))
                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )
                input_batch = np.expand_dims(image_array, axis=0)

                # Prediction
                prediction = model.predict(input_batch, verbose=0)
                defect_probability = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

                # Classification threshold
                threshold = 0.25
                is_defective = defect_probability >= threshold

                st.markdown("**Output Image**")

                st.image(
                    image,
                    caption="Inspected Image",
                    width=300
                )

                st.markdown("**Prediction**")

                if is_defective:
                    st.markdown(
                        '<div class="defect-result">⚠️ DEFECTIVE</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="normal-result">✅ NORMAL</div>',
                        unsafe_allow_html=True
                    )

                st.write("")

                st.metric(
                    label="Predicted Defect Probability",
                    value=f"{defect_probability * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, defect_probability))
                )

                st.caption(
                    "Classification threshold: 25%. "
                    "This is an academic prototype and not a "
                    "guarantee of industrial quality."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <hr>
    <b>Industrial Defect Detection System</b><br>
    Academic Project | CNN-Based Image Classification<br>
    Developed using Python, TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)
```
