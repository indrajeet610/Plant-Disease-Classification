import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load your trained model
model = tf.keras.models.load_model('final_model.h5')

class_names = ['Potato__Late_blight', 'Potato__Early_blight', 'Potato__healthy']

st.title('Potato Leaf Disease Prediction')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize the image

    # Make a prediction
    with st.spinner('Predicting...'):
        predictions = model.predict(np.expand_dims(image, axis=0))

    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f'Prediction: {predicted_class}')
    st.write(f'Confidence: {confidence:.2%}')
