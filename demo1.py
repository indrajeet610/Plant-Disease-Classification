import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pyttsx3  # Import the pyttsx3 library for speech synthesis

# Define a custom CSS style for buttons
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the title and description
st.title('Potato Leaf Disease Prediction')
st.write('Upload an image of a potato leaf to predict the disease.')

# Initialize the text-to-speech engine
engine = pyttsx3.init()


def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')

    if file_uploaded is not None:
        image = Image.open(file_uploaded)

        st.write("Uploaded Image:")
        st.image(image, use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                result, confidence = predict_class(image)

            st.subheader('Prediction:')
            prediction_container = st.empty()
            with prediction_container:
                st.write(result)
                st.write(f'Confidence: {confidence}%')
                if 'blight' in result.lower():
                    st.warning('The leaves might be infected. Consider consulting an expert.')
                else:
                    st.success('The leaves appear healthy!')

            # Convert the prediction to speech and speak it
            speak_result(result)


def predict_class(image):
    classifier_model = keras.models.load_model('potatoes.h5', compile=False)
    shape = (256, 256, 3)
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape )])

    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)

    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * np.max(prediction[0]), 2)
    final_pred = class_name[np.argmax(prediction)]

    return final_pred


def speak_result(result):
    # Use pyttsx3 to speak the prediction result
    engine.say(f'The predicted disease is {result}')
    engine.runAndWait()


if __name__ == '__main__':
    main()

# Add a footer
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #333;
    color: white;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    Your Footer Content Goes Here
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
