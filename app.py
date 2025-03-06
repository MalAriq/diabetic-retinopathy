import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/Model_8_augmentasi.h5')
    return model

model = load_model()

def preprocess_image(image, target_size):
    # Resize image to target size
    image = image.resize(target_size)
    # Convert to NumPy array and normalize
    image = np.array(image) / 255.0
    # Expand dimensions to match model input
    image = np.expand_dims(image, axis=0)
    return image


# Streamlit UI
st.title("Image Classification App")
st.header("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess and predict
    processed_image = preprocess_image(image, target_size=(224, 224)) 
    prediction = model.predict(processed_image)
    class_label = np.argmax(prediction, axis=1)
    
    st.write(f"Prediction: {class_label}")
