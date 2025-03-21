import streamlit as st
#st.cache_data.clear()  # Clear Streamlit's cache (new method)

import tensorflow as tf
import numpy as np
#import cv2
from PIL import Image

# STEP 3: Load the trained model
@st.cache_resource  # Cache the model to speed up loading
def load_model():
    # Return the fine-tuned model
    return tf.keras.models.load_model(r"C:\Users\Gehan Massoud\keratoconus_model_finetuned300.keras")

model = load_model()

# Define class labels
class_names = ["Normal", "Keratoconus"]

# STEP 4: Preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image)  # Convert PIL image to NumPy array
   # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR format
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# STEP 5: Make Predictions
def predict_keratoconus(image):
    img_array = preprocess_image(image)  # Preprocess uploaded image
    predictions = model.predict(img_array)  # Get model predictions
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
    return class_names[predicted_class]  # Return only the predicted class

# STEP 6: Build Streamlit UI
st.title("Miracle Keratoconus Detection AI")
st.write("Upload an eye scan image and let the AI analyze it.")

uploaded_file = st.file_uploader("Upload an eye scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open image using PIL
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        predicted_class = predict_keratoconus(image)
        st.success(f"**Prediction:** {predicted_class}")
