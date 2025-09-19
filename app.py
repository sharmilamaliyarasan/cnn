import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load trained model
model = load_model("digit_cnn_model.h5")

# --- Page Configuration ---
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    /* Main card container */
    .container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
        max-width: 500px;
        margin: auto;
    }
    .title {
        color: #4B0082;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        color: #555;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stFileUploader>div>div>label {
        background: linear-gradient(90deg, #4B0082 0%, #8A2BE2 100%);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s ease;
    }
    .stFileUploader>div>div>label:hover {
        background: linear-gradient(90deg, #8A2BE2 0%, #4B0082 100%);
    }
    .predicted {
        font-size: 3rem;
        color: #4B0082;
        text-align: center;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Container ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">✏️ Handwritten Digit Recognizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a digit image (0-9) and see the prediction instantly.</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('L')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img_resized = img.resize((28,28))
    img_array = ImageOps.invert(img_resized)
    img_array = np.array(img_array).reshape(1,28,28,1)/255.0
    
    # Predict
    pred = np.argmax(model.predict(img_array), axis=1)[0]
    
    # Show prediction
    st.markdown(f'<p class="predicted">Predicted Digit: {pred}</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
