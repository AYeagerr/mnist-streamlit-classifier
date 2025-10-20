# app.py — MNIST Digit Classifier (Logistic Regression, ANN, CNN)
# Expects:
#   models/logistic_regression.joblib
#   models/ann_dense.h5
#   models/cnn.h5
# Run: streamlit run app.py

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
from tensorflow import keras

# -----------------------
# Utility: Safe rerun for any Streamlit version
# -----------------------
def do_rerun():
    """Rerun the app safely on all Streamlit versions."""
    try:
        st.rerun()  # new API
    except Exception:
        st.experimental_rerun()

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️", layout="centered")
st.title("✍️ MNIST Digit Classifier")
st.write("Draw a digit (0–9), choose a model, and see the prediction with confidence.")

# -----------------------
# Session state (for clear canvas functionality)
# -----------------------
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

# -----------------------
# Load models
# -----------------------
@st.cache_resource
def load_models():
    models = {"Logistic": None, "ANN": None, "CNN": None}
    errors = []

    try:
        models["Logistic"] = joblib.load("models/logistic_regression.joblib")
    except Exception as e:
        errors.append(f"Logistic model not loaded: {e}")

    try:
        models["ANN"] = keras.models.load_model("models/ann_dense.h5")
    except Exception as e:
        errors.append(f"ANN model not loaded: {e}")

    try:
        models["CNN"] = keras.models.load_model("models/cnn.h5")
    except Exception as e:
        errors.append(f"CNN model not loaded: {e}")

    return models, errors

models, load_errors = load_models()
if load_errors:
    for msg in load_errors:
        st.warning(msg)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose a model:", ["Logistic", "ANN", "CNN"])
stroke_width = st.sidebar.slider("Brush size", min_value=6, max_value=32, value=14, step=2)
invert_colors = st.sidebar.checkbox("Invert colors (use if your strokes look black on white)", value=False)
show_topk = st.sidebar.slider("Show top-k classes", min_value=1, max_value=10, value=3, step=1)

st.write("**Draw here (white on black by default):**")

# -----------------------
# Drawing canvas
# -----------------------
canvas_res = 256
canvas = st_canvas(
    fill_color="#000000",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=canvas_res,
    width=canvas_res,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state['canvas_key']}",  # refreshes when clearing
)

# -----------------------
# Image preprocessing
# -----------------------
def preprocess(image: Image.Image, invert: bool = False):
    """Convert RGBA canvas -> 28x28 grayscale [0..1] like MNIST."""
    img = image.convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    if invert:
        arr = 1.0 - arr
    return arr

# -----------------------
# Buttons
# -----------------------
col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear Canvas")

if clear_btn:
    st.session_state["canvas_key"] += 1  # reset canvas
    do_rerun()

# -----------------------
# Prediction helpers
# -----------------------
def softmax(x: np.ndarray):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def predict(models, model_name: str, arr28: np.ndarray):
    """Return predicted class and probabilities."""
    if models.get(model_name) is None:
        raise RuntimeError(f"{model_name} model is not loaded. Check ./models/")

    if model_name == "Logistic":
        # Logistic Regression outputs class labels directly, not probabilities
        x = arr28.reshape(1, 28 * 28)
        pred = models["Logistic"].predict(x)[0]  # Logistic Regression outputs the predicted class label
        probs = None  # No probabilities for Logistic Regression
    elif model_name == "ANN":
        x = arr28.reshape(1, 28 * 28)
        probs = models["ANN"].predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
    else:  # CNN
        x = arr28.reshape(1, 28, 28, 1)
        probs = models["CNN"].predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))

    return pred, probs

# -----------------------
# Main prediction logic
# -----------------------
arr28_to_use = None
if canvas.image_data is not None:
    pil_img = Image.fromarray(canvas.image_data.astype("uint8"))
    arr28_to_use = preprocess(pil_img, invert=invert_colors)

if run_btn:
    if arr28_to_use is None:
        st.error("Please draw a digit first.")
    else:
        try:
            pred, probs = predict(models, model_choice, arr28_to_use)
        except Exception as e:
            st.error(str(e))
        else:
            st.write("**Model input preview (28×28)**")
            st.image((arr28_to_use * 255).astype("uint8"), width=140, caption="Grayscale 28×28")

            st.subheader(f"Prediction: **{pred}**")
          
            if probs is not None:
                st.write(f"Confidence: **{np.max(probs):.2%}**")
                top_idx = np.argsort(probs)[::-1][:show_topk]
                st.write(f"Top-{show_topk} classes:")
                for i in top_idx:
                    st.write(f"- {i}: {probs[i]:.2%}")
            else:
                st.write("No confidence for Logistic Regression (it outputs class labels directly).")

            if model_choice == "Logistic":
                st.caption("Note: Logistic model ‘confidence’ is not applicable. It only outputs predicted class labels.")

st.caption("Tip: If your digit looks inverted (dark stroke on white), toggle **Invert colors** in the sidebar.")