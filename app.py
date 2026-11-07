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

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef4fb 0%, #f8fafc 55%, #e8f0fa 100%);
        color: #172033;
    }
    .main-title {
        text-align: center;
        color: #12345a;
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #53657d;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .panel {
    background: transparent;
    padding: 0.5rem;
    border: none;
    box-shadow: none;
    }
    .result-normal {
        background: #e8f7ee;
        border-left: 6px solid #1b9b59;
        padding: 1rem;
        border-radius: 10px;
        color: #145c38;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .result-defect {
        background: #fff0ed;
        border-left: 6px solid #d94b3d;
        padding: 1rem;
        border-radius: 10px;
        color: #8e241c;
        font-size: 1.2rem;
        font-weight: 700;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background: #1559a6;
        color: white;
        font-weight: 700;
        border: none;
    }
    div.stButton > button:hover {
        background: #0e437f;
        color: white;
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
        with st.spinner("Downloading trained CNN model..."):
            response = requests.get(MODEL_URL, timeout=300)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as file:
                file.write(response.content)

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">CNN-Based Industrial Defect Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Visual inspection of metal nuts using a trained CNN model</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About the Project")
    st.write(
        "This application uses a trained Convolutional Neural Network "
        "to classify a metal-nut image as Normal or Defective."
    )
    st.info(
        "The model expects an RGB image and resizes it to 128 × 128 pixels."
    )
    st.caption("Model: TensorFlow / Keras")
    st.caption("Dataset: MVTec AD – metal_nut")

# ---------------- IMAGE INPUT ----------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("1. Upload Input Image")
    uploaded_file = st.file_uploader(
        "Choose a metal-nut image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Input Image", width=300)
    else:
        st.write("Upload an image to begin inspection.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("2. Prediction Result")

    if uploaded_file is None:
        st.write("Your prediction will appear here after uploading an image.")
    else:
        if st.button("🔍 Inspect Image", type="primary"):
            try:
                model = load_model()

                # IMPORTANT:
                # The uploaded CNN already contains a Rescaling(1./255) layer.
                # Therefore, pass pixel values in the 0–255 range here.
                resized_image = image.resize((128, 128))
                image_array = np.asarray(resized_image, dtype=np.float32)
                input_batch = np.expand_dims(image_array, axis=0)

                prediction = model.predict(input_batch, verbose=0)
                defect_probability = float(np.asarray(prediction).reshape(-1)[0])

                # Same decision threshold used for the reported CNN result.
                threshold = 0.25
                is_defective = defect_probability >= threshold

                st.markdown("**Output Image**")
                st.image(image, caption="Image passed to the CNN", width=300)
                st.markdown("**Prediction**")
                if is_defective:
                    st.markdown(
                        '<div class="result-defect">⚠️ DEFECTIVE</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-normal">✅ NORMAL</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("**Defect Probability**")
                st.metric(
                    label="Predicted probability of defect",
                    value=f"{defect_probability * 100:.2f}%"
                )
                st.progress(max(0.0, min(1.0, defect_probability)))

                st.caption(
                    "Decision threshold: 25%. This is a model prediction, "
                    "not a guarantee of industrial quality."
                )

            except FileNotFoundError:
                st.error(
                    "Model file not found. Place industrial_defect_cnn.keras "
                    "in the same folder as app.py."
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Academic prototype | CNN image-level classification | "
    "Not validated for production-line quality control"
)
