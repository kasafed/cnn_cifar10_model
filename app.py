import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

#load the pre_trained model
model = tf.keras.models.load_model('cifar_model1.keras')

#define the class names for cifar-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it into one of the CIFAR-10 categories.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded, process it and make a prediction
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    #image (32,32,3) -> (1,32,32,3), which is the expected input shape for the model

    # Make predictions
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Prediction Confidence: {np.max(predictions):.2f}")