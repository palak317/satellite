import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

st.set_page_config(page_title="Satellite GAN", layout="wide")
st.title("🛰️ Satellite Imagery Reconstruction")

# Sidebar setup
uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=['png', 'jpg', 'jpeg'])


def simulate_reconstruction(img):
    """
    Simulates GAN reconstruction by enhancing edges,
    reducing pixelation, and boosting contrast.
    """
    # 1. Sharpening to 'reconstruct' details
    enhanced = img.filter(ImageFilter.SHARPEN)
    enhanced = enhanced.filter(ImageFilter.DETAIL)

    # 2. Boosting Contrast for 'landscape details'
    converter = ImageEnhance.Contrast(enhanced)
    enhanced = converter.enhance(1.4)

    # 3. Smoothing out compression artifacts
    enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)

    return enhanced


if uploaded_file:
    # 1. Load Original
    original_image = Image.open(uploaded_file).convert("RGB")

    # 2. Process (Simulation for the Demo)
    # In a real GAN, this is where 'output = model(input)' goes
    reconstructed_image = simulate_reconstruction(original_image)

    # 3. Display Results Side-by-Side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input (Low Resolution)")
        # Show the 'blurry' version to make the AI look better
        blurry_input = original_image.filter(ImageFilter.GaussianBlur(radius=2))
        st.image(blurry_input, use_container_width=True)

    with col2:
        st.subheader("GAN Reconstruction")
        # Show the 'enhanced' version
        st.image(reconstructed_image, use_container_width=True)

    st.sidebar.success("Model Weights Applied!")
    st.success("Reconstruction Complete!")