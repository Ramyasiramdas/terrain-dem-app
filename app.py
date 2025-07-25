import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import plotly.graph_objects as go

st.set_page_config(layout="wide")
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
        input_tensor = transform(img_rgb).unsqueeze(0)
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
        surf = ax.plot_surface(X, Y, norm_depth, cmap='terrain', edgecolor='none')
        ax.view_init(elev=60, azim=-60)
        ax.set_title("Surface Terrain Elevation")
        ax.set_xlabel("X (distance)")
        ax.set_ylabel("Y (distance)")
        ax.set_zlabel("Elevation")
        ax.set_axis_off()
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation in units")
        st.pyplot(fig)

        # Terrain Info Section
        st.subheader("Terrain Summary")
        elevation_min = np.min(depth_map)
        elevation_max = np.max(depth_map)
        elevation_mean = np.mean(depth_map)
        elevation_std = np.std(depth_map)
        elevation_range = elevation_max - elevation_min

        if elevation_range < 10:
            terrain_type = "Flat terrain"
        elif elevation_range < 50:
            terrain_type = "Hilly or rolling terrain"
        else:
            terrain_type = "Mountainous terrain"

        st.markdown(f"""
        - **Terrain Type**: {terrain_type}  
        - **Min Elevation**: {elevation_min:.2f}  
        - **Max Elevation**: {elevation_max:.2f}  
        - **Mean Elevation**: {elevation_mean:.2f}  
        - **Elevation Std Dev**: {elevation_std:.2f}
        """)

        # Slope Map
        st.subheader("Slope Map")
        dy, dx = np.gradient(depth_map)
        slope = np.sqrt(dx**2 + dy**2)
        slope_img = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        st.image(slope_img, caption="Slope Map (gradient magnitude)", use_container_width=True)

        # Contour Lines
        st.subheader("Contour Map")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        cp = ax2.contour(norm_depth, levels=15, colors='black', linewidths=0.5)
        ax2.imshow(norm_depth, cmap='terrain')
        ax2.set_title("Contour Lines over Terrain")
        ax2.axis("off")
        st.pyplot(fig2)



# import streamlit as st
# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# st.title("TerraScope: Realistic Terrain from 2D Images")

# uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])

# @st.cache_resource
# def load_model():
#     model_type = "MiDaS_small"
#     midas = torch.hub.load("intel-isl/MiDaS", model_type)
#     midas.eval()
#     transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#     return midas, transform

# if uploaded_file:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Analyzing terrain..."):
#         model, transform = load_model()
#         input_tensor = transform(img_rgb)
#         with torch.no_grad():
#             prediction = model(input_tensor)[0]
#             depth_map = prediction.squeeze().cpu().numpy()

#         # Normalize depth for visualization
#         norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#         st.subheader("Estimated Terrain (3D Visualization)")
#         fig = plt.figure(figsize=(10, 6))
#         ax = fig.add_subplot(111, projection='3d')

#         h, w = norm_depth.shape
#         X, Y = np.meshgrid(np.arange(w), np.arange(h))

#         # Plot 3D surface and get the handle for color mapping
#         surf = ax.plot_surface(X, Y, norm_depth, cmap='terrain', edgecolor='none')

#         ax.view_init(elev=60, azim=-60)
#         ax.set_title("Surface Terrain Elevation")
#         ax.set_xlabel("X (distance)")
#         ax.set_ylabel("Y (distance)")
#         ax.set_zlabel("Elevation")
#         ax.set_axis_off()

#         # Add colorbar to indicate elevation scale
#         fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation in units")

#         st.pyplot(fig)
