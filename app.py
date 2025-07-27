import streamlit as st
import torch
import cv2
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

st.set_page_config(page_title="TerraScope", layout="centered")
st.title("TerraScope: Interactive 3D Terrain from 2D Image")

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

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Ensure shape is [1, 3, H, W]

        with torch.no_grad():
            prediction = model(input_tensor)[0]
            depth_map = prediction.squeeze().cpu().numpy()

        # Smooth for more realistic appearance
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

        # Plot interactive 3D terrain
        h, w = depth_map.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        fig = go.Figure(data=[go.Surface(z=depth_map, x=X, y=Y, colorscale='Earth', showscale=True)])

        fig.update_layout(
            title="Interactive 3D Terrain",
            margin=dict(l=10, r=10, t=40, b=10),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Elevation',
                aspectratio=dict(x=1, y=1, z=0.4),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Terrain Summary
        st.subheader("Terrain Summary")
        elevation_min = np.min(depth_map)
        elevation_max = np.max(depth_map)
        elevation_mean = np.mean(depth_map)
        elevation_std = np.std(depth_map)
        elevation_range = elevation_max - elevation_min

        if elevation_range < 10:
            terrain_type = "Flat terrain"
        elif elevation_range < 50:
            terrain_type = "Hilly terrain"
        else:
            terrain_type = "Mountainous terrain"

        st.markdown(f"""
        - **Terrain Type**: {terrain_type}  
        - **Min Elevation**: {elevation_min:.2f}  
        - **Max Elevation**: {elevation_max:.2f}  
        - **Mean Elevation**: {elevation_mean:.2f}  
        - **Elevation Std Dev**: {elevation_std:.2f}
        """)
