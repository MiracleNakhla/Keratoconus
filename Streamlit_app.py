import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model using a relative path
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("keratoconus_model_finetuned300.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Define class labels
class_names = ["Normal", "Keratoconus"]

# Preprocess the uploaded image using PIL for resizing
def preprocess_image(image):
    try:
        if image is None:
            return "Error: No image provided"
        # Resize the image using PIL (image is a PIL Image object)
        image = image.resize((224, 224))
        # Convert the resized image to a NumPy array
        image = np.array(image)
        # Normalize pixel values (scale between 0 and 1)
        image = image / 255.0
        # Add batch dimension (the model expects a batch of images)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        return f"Error in preprocessing image: {str(e)}"

# Make Predictions
def predict_keratoconus(image):
    if model is None:
        return "Model not loaded."
    try:
        img_array = preprocess_image(image)
        # If an error occurred during preprocessing, img_array will be a string
        if isinstance(img_array, str):
            return img_array
        # Get predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return class_names[predicted_class]
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Build Streamlit UI
st.title("KlearVue")
st.write("Upload an eye scan image and let KlearVue analyze it.")

uploaded_file = st.file_uploader("Upload an eye scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        prediction = predict_keratoconus(image)
        st.success(f"Prediction: {prediction}")
