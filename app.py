
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Rice Classification", page_icon="🌾")

st.title("🌾 Rice Classification")
st.write("CNN ve VGG16 modelleri ile pirinç türü tahmini")

class_names = [
    "Arborio",
    "Basmati",
    "Ipsala",
    "Jasmine",
    "Karacadag"
]

@st.cache_resource
def load_models():
    cnn = load_model("riceModel.keras")
    vgg = load_model("riceModelTL.keras")
    return cnn, vgg

try:
    cnn_model, vgg_model = load_models()
except Exception as e:
    st.error(f"Modeller yüklenemedi: {e}")
    st.stop()

model_choice = st.selectbox(
    "Model seçiniz",
    ["CNN", "VGG16"]
)

uploaded_file = st.file_uploader(
    "Bir pirinç fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", width=300)

    img = image.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Tahmin Et"):
        model = cnn_model if model_choice == "CNN" else vgg_model

        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        st.success(f"Tahmin: {class_names[predicted_class]}")
        st.write(f"Güven oranı: %{confidence:.2f}")
