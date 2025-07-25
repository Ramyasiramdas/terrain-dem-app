import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

st.title("TerraScope: Realistic Terrain from 2D Images")

uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return midas, transform

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing terrain..."):
        model, transform = load_model()
        input_tensor = transform(img_rgb)
        with torch.no_grad():
            prediction = model(input_tensor)[0]
            depth_map = prediction.squeeze().cpu().numpy()

        # Normalize depth for visualization
        norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        st.subheader("Estimated Terrain (3D Visualization)")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        h, w = norm_depth.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # Plot 3D surface and get the handle for color mapping
        surf = ax.plot_surface(X, Y, norm_depth, cmap='terrain', edgecolor='none')

        ax.view_init(elev=60, azim=-60)
        ax.set_title("Surface Terrain Elevation")
        ax.set_xlabel("X (distance)")
        ax.set_ylabel("Y (distance)")
        ax.set_zlabel("Elevation")
        ax.set_axis_off()

        # Add colorbar to indicate elevation scale
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation in units")

        st.pyplot(fig)
