import os
import urllib.request
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import math

# ----------------------------
# Build SRCNN model with proper layer naming
# ----------------------------
def build_srcnn_model():
    input_img = Input(shape=(256, 256, 3), name='input_layer')
    l1 = Conv2D(64, 9, padding='same', activation='relu', name='conv2d_1')(input_img)
    l2 = Conv2D(32, 3, padding='same', activation='relu', name='conv2d_2')(l1)
    l3 = Conv2D(16, 1, padding='same', activation='relu', name='conv2d_3')(l2)
    l4 = Conv2D(3, 5, padding='same', activation='relu', name='conv2d_4')(l3)
    return Model(inputs=input_img, outputs=l4)

# ----------------------------
# Download models
# ----------------------------
srcnn_url = "https://huggingface.co/Dimsralf/model/resolve/main/srcnn_model.keras?download=true"
unet_url = "https://huggingface.co/Dimsralf/model/resolve/main/unet_model.keras?download=true"

@st.cache_resource
def download_models():
    if not os.path.exists("srcnn_model.keras"):
        urllib.request.urlretrieve(srcnn_url, "srcnn_model.keras")
    if not os.path.exists("unet_model.keras"):
        urllib.request.urlretrieve(unet_url, "unet_model.keras")

download_models()

# ----------------------------
# Load models with proper weight loading
# ----------------------------
try:
    # First build the SRCNN model architecture
    model_srcnn = build_srcnn_model()
    # Then load the weights separately
    model_srcnn.load_weights("srcnn_model.keras")
    # Compile the model (not necessary for inference but good practice)
    model_srcnn.compile(optimizer='adam', loss='mse')
except Exception as e:
    st.error(f"Failed to load SRCNN model: {str(e)}")
    st.stop()

try:
    # U-Net: Load with custom objects
    custom_objects = {
        'LeakyReLU': tf.keras.layers.LeakyReLU,
        'BatchNormalization': tf.keras.layers.BatchNormalization,
        'Dropout': tf.keras.layers.Dropout,
        'concatenate': tf.keras.layers.concatenate,
    }
    model_unet = tf.keras.models.load_model("unet_model.keras", 
                                          custom_objects=custom_objects, 
                                          compile=False)
except Exception as e:
    st.error(f"Failed to load U-Net model: {str(e)}")
    st.stop()

# ----------------------------
# Rest of your code remains the same
# ----------------------------
def add_blur(image, blur_level):
    if blur_level == 0:
        return image
    return cv2.GaussianBlur(image, (2 * blur_level + 1, 2 * blur_level + 1), 0)

def predict_image(model, image):
    input_img = image.astype(np.float32)
    pred = model.predict(np.expand_dims(input_img, axis=0), verbose=0)
    return np.clip(pred[0], 0.0, 1.0)

def calculate_metrics(original, restored):
    original = np.clip(original, 0, 1)
    restored = np.clip(restored, 0, 1)
    mse_val = mean_squared_error(original, restored)
    rmse_val = math.sqrt(mse_val)
    psnr_val = peak_signal_noise_ratio(original, restored, data_range=1.0)
    try:
        ssim_val = structural_similarity(original, restored, channel_axis=-1, data_range=1.0)
    except ValueError:
        ssim_val = 0.0
    return mse_val, rmse_val, psnr_val, ssim_val

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Restorasi Citra")
st.title("🖼️ Restorasi Citra dengan SRCNN & U-Net")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])

with col1:
    uploaded_file = st.file_uploader("Upload Citra", type=["jpg", "png", "jpeg"])
    blur_level = st.slider("Tingkat Blur", min_value=0, max_value=10, value=5)
    if st.button("Proses"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((256, 256))
            image_np = np.array(image) / 255.0

            blurred_image = add_blur((image_np * 255).astype(np.uint8), blur_level)
            blurred_image = blurred_image.astype(np.float32) / 255.0

            output_srcnn = predict_image(model_srcnn, blurred_image)
            output_unet = predict_image(model_unet, blurred_image)

            st.session_state['original'] = image_np
            st.session_state['blurred'] = blurred_image
            st.session_state['srcnn'] = output_srcnn
            st.session_state['unet'] = output_unet

if 'original' in st.session_state:
    col1.image(st.session_state['blurred'], caption="Citra Blur", use_container_width=True)
    col2.image(st.session_state['original'], caption="Before", use_container_width=True)
    col3.image(st.session_state['srcnn'], caption="SRCNN", use_container_width=True)
    col4.image(st.session_state['unet'], caption="U-NET", use_container_width=True)

    def render_metrics(col, title, target):
        mse, rmse, psnr_val, ssim_val = calculate_metrics(st.session_state['original'], target)
        with col:
            st.markdown(f"**Parameter - {title}**")
            st.markdown(f"MSE  : `{mse:.4f}`")
            st.markdown(f"RMSE : `{rmse:.4f}`")
            st.markdown(f"PSNR : `{psnr_val:.2f}`")
            st.markdown(f"SSIM : `{ssim_val:.4f}`")

    st.markdown("---")
    col5, col6, col7 = st.columns(3)
    render_metrics(col5, "Before", st.session_state['blurred'])
    render_metrics(col6, "SRCNN", st.session_state['srcnn'])
    render_metrics(col7, "U-NET", st.session_state['unet'])