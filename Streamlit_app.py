import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
def load_model():
    try:
        # Load your fine-tuned model (use the correct path to your model)
        model = tf.keras.models.load_model(r"C:\Users\Gehan Massoud\keratoconus_model_finetuned300.keras")
        return model
    except Exception as e:
        return f"Error loading model: {str(e)}"

model = load_model()

# Define class labels
class_names = ["Normal", "Keratoconus"]

# Preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert PIL image to NumPy array
        image = np.array(image)
        # Convert the image from RGB to BGR (since OpenCV uses BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize the image to 224x224 pixels (the expected input size for MobileNetV2)
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values (scale between 0 and 1)
        image = image / 255.0
        # Add batch dimension (the model expects a batch of images)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        return f"Error in preprocessing image: {str(e)}"

# Make Predictions
def predict_keratoconus(image):
    try:
        # Preprocess the image before feeding into the model
        img_array = preprocess_image(image)
        
        # If there was an error in preprocessing, return the error message
        if isinstance(img_array, str):  
            return img_array
        
        # Make prediction using the model
        predictions = model.predict(img_array)
        # Get the class with the highest probability
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Return the predicted class (either 'Normal' or 'Keratoconus')
        return class_names[predicted_class]
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Streamlit UI
st.title("Miracle Keratoconus Detection AI")
st.write("Upload an eye scan image and let the AI analyze it.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an eye scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # When the button is clicked, run the prediction
    if st.button("Analyze Image"):
        predicted_class = predict_keratoconus(image)
        st.success(f"**Prediction:** {predicted_class}")
