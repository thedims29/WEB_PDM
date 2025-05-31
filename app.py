import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import math

# ----------------------------
# Build model SRCNN
# ----------------------------
def build_srcnn():
    input_img = Input(shape=(256, 256, 3), name='input_layer')
    l1 = Conv2D(64, 9, padding='same', activation='relu', name='conv2d_1')(input_img)
    l2 = Conv2D(32, 3, padding='same', activation='relu', name='conv2d_2')(l1)
    l3 = Conv2D(16, 1, padding='same', activation='relu', name='conv2d_3')(l2)
    l4 = Conv2D(3, 5, padding='same', activation='relu', name='conv2d_4')(l3)
    return Model(inputs=input_img, outputs=l4)

# ----------------------------
# Load model dan weights
# ----------------------------
try:
    model_srcnn = build_srcnn()
    model_srcnn.load_weights("srcnn_model.h5")
except Exception as e:
    st.error(f"Gagal memuat model SRCNN: {str(e)}")
    st.stop()

try:
    model_srcnn = tf.keras.models.load_model("srcnn_model.keras", compile=False)
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
# Fungsi bantu
# ----------------------------
def add_blur(image, blur_level):
    if blur_level == 0:
        return image
    # Pastikan gambar uint8 untuk cv2.GaussianBlur
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image
    blurred = cv2.GaussianBlur(image_uint8, (2 * blur_level + 1, 2 * blur_level + 1), 0)
    # Kembalikan ke float32 0-1
    return blurred.astype(np.float32) / 255.0

def predict_image(model, image):
    input_img = image.astype(np.float32)
    pred = model.predict(np.expand_dims(input_img, axis=0), verbose=0)
    pred = np.clip(pred[0], 0.0, 1.0)
    return pred

def calculate_metrics(original, restored):
    original = np.clip(original, 0, 1)
    restored = np.clip(restored, 0, 1)
    mse_val = mean_squared_error(original, restored)
    rmse_val = math.sqrt(mse_val)
    psnr_val = peak_signal_noise_ratio(original, restored, data_range=1.0)
    try:
        ssim_val = structural_similarity(original, restored, channel_axis=-1, data_range=1.0)
    except:
        ssim_val = 0.0
    return mse_val, rmse_val, psnr_val, ssim_val

def prepare_for_display(img):
    # img diasumsikan float32 0-1, convert ke uint8 0-255
    img_disp = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img_disp

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Restorasi Citra dengan SRCNN")
st.title("🖼️ Restorasi Citra dengan SRCNN")

col1, col2, col3 = st.columns(3)

with col1:
    uploaded_file = st.file_uploader("Upload Citra (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])
    blur_level = st.slider("Tingkat Blur", min_value=0, max_value=10, value=5)
    if st.button("Proses"):
        if uploaded_file is not None:
            # Load dan resize image ke 256x256
            image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
            image_np = np.array(image).astype(np.float32) / 255.0

            blurred_image = add_blur(image_np, blur_level)

            output_srcnn = predict_image(model_srcnn, blurred_image)

            st.session_state['original'] = image_np
            st.session_state['blurred'] = blurred_image
            st.session_state['srcnn'] = output_srcnn

# ----------------------------
# Tampilkan hasil dan metrik
# ----------------------------
if 'original' in st.session_state:
    img_blurred = prepare_for_display(st.session_state['blurred'])
    img_original = prepare_for_display(st.session_state['original'])
    img_srcnn = prepare_for_display(st.session_state['srcnn'])

    # Debug tambahan untuk memastikan data valid
    st.write("DEBUG shapes and dtypes:")
    st.write(f"Blurred shape: {img_blurred.shape}, dtype: {img_blurred.dtype}")
    st.write(f"Original shape: {img_original.shape}, dtype: {img_original.dtype}")
    st.write(f"SRCNN shape: {img_srcnn.shape}, dtype: {img_srcnn.dtype}")

    st.write("min/max blurred:", img_blurred.min(), img_blurred.max())
    st.write("min/max original:", img_original.min(), img_original.max())
    st.write("min/max srcnn:", img_srcnn.min(), img_srcnn.max())

    # Pastikan contiguous array
    img_blurred = np.ascontiguousarray(img_blurred)
    img_original = np.ascontiguousarray(img_original)
    img_srcnn = np.ascontiguousarray(img_srcnn)

    # Tampilkan gambar dengan parameter yang kompatibel
    col1.image(img_blurred, caption="Citra Blur", use_column_width=True)
    col2.image(img_original, caption="Citra Asli", use_column_width=True)
    col3.image(img_srcnn, caption="Hasil SRCNN", use_column_width=True)

    def render_metrics(col, title, target_img_uint8):
        # Convert uint8 0-255 ke float 0-1 utk metrik
        target = target_img_uint8.astype(np.float32) / 255.0
        mse, rmse, psnr_val, ssim_val = calculate_metrics(st.session_state['original'], target)
        with col:
            st.markdown(f"**Parameter - {title}**")
            st.markdown(f"MSE  : `{mse:.4f}`")
            st.markdown(f"RMSE : `{rmse:.4f}`")
            st.markdown(f"PSNR : `{psnr_val:.2f}`")
            st.markdown(f"SSIM : `{ssim_val:.4f}`")

    st.markdown("---")
    col4, col5 = st.columns(2)
    render_metrics(col4, "Blurred", img_blurred)
    render_metrics(col5, "SRCNN", img_srcnn)
