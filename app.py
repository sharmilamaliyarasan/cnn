import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64

model = load_model("digit_cnn_model.h5")

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="centered"
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("C:/Users/HP/cnn/images.jpeg")

st.markdown("""
    <style>
    .container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
        max-width: 500px;
        margin: auto;
        animation: fadeIn 2s ease;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .title {
        color: #fff;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.3rem;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.7);
    }
    .subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.6);
    }
    .stFileUploader>div>div>label {
        background: linear-gradient(90deg, #4B0082 0%, #8A2BE2 100%);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.3s ease, background 0.3s ease;
    }
    .stFileUploader>div>div>label:hover {
        background: linear-gradient(90deg, #8A2BE2 0%, #4B0082 100%);
        transform: scale(1.05);
    }
    .uploaded-img {
        display: block;
        margin: 20px auto;
        border-radius: 15px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
        animation: popIn 1s ease;
    }
    @keyframes popIn {
        0% {transform: scale(0.7); opacity: 0;}
        100% {transform: scale(1); opacity: 1;}
    }
    .predicted {
        font-size: 3rem;
        color: #fff;
        text-align: center;
        font-weight: bold;
        margin-top: 1.5rem;
        text-shadow: 0px 0px 15px #8A2BE2, 0px 0px 25px #4B0082;
        animation: glow 1.5s infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0px 0px 10px #8A2BE2, 0px 0px 20px #4B0082; }
        to { text-shadow: 0px 0px 20px #8A2BE2, 0px 0px 40px #4B0082; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">✏️ Handwritten Digit Recognizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a digit image (0-9) and see the prediction instantly.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file).convert('L')
    st.image(img, caption='Uploaded Image', use_column_width=True, output_format="PNG")
   
    img_resized = img.resize((28,28))
    img_array = ImageOps.invert(img_resized)
    img_array = np.array(img_array).reshape(1,28,28,1)/255.0
    
    pred = np.argmax(model.predict(img_array), axis=1)[0]
    
    st.markdown(f'<p class="predicted">Predicted Digit: {pred}</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
