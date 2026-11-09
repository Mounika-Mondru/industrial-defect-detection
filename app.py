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
    padding-top: 0rem;
    padding-bottom: 2rem;
}

.hero {
    text-align: center;
    padding: 5px 10px 20px 10px;
    margin-top: 0;
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
    background: #1559a6;
    color: white;
    font-size: 1rem;
    font-weight: 750;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #0e437f;
    color: white;
    transform: translateY(-2px);
}

.normal-result {
    background: #e8f7ee;
    border-left: 6px solid #1b9b59;
    padding: 1rem;
    border-radius: 10px;
    color: #145c38;
    font-size: 1.2rem;
    font-weight: 700;
}

.defect-result {
    background: #fff0ed;
    border-left: 6px solid #d94b3d;
    padding: 1rem;
    border-radius: 10px;
    color: #8e241c;
    font-size: 1.2rem;
    font-weight: 700;
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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 About the Project")

    st.write(
        "This application uses a trained Convolutional Neural Network "
        "to classify a metal-nut image as Normal or Defective."
    )

    st.divider()

    st.subheader("🧠 Model Information")
    st.write("**Model:** TensorFlow / Keras")
    st.write("**Dataset:** MVTec AD – Metal Nut")
    st.write("**Input Size:** 128 × 128 RGB")

    st.divider()

    st.subheader("🔍 Detection Details")
    st.write("**Task:** Normal / Defective Classification")
    st.write("**Decision Threshold:** 25%")
    st.write("**Image Format:** JPG, JPEG, PNG")

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
            width=500
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
                threshold = 0.50
                is_defective = defect_probability >= threshold
                normal_probability = 1 - defect_probability

                st.markdown("**Output Image**")

                st.image(
                    image,
                    caption="Inspected Image",
                    width=500
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
                st.metric(
                    label="Predicted Normal Probability",
                    value=f"{normal_probability * 100:.2f}%"
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
