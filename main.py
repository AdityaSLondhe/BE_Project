import os
import json
import cv2

from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

import google.generativeai as genai

import requests

import gdown

file_id = "1W_6W0CNgW-2B4TcxA5nWjkwpLLs4eECj"
gdown_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_model/NewDataset.h5"

if not os.path.exists(model_path):
    gdown.download(gdown_url, model_path, quiet=False)


working_dir = os.path.dirname(os.path.abspath(__file__))
print (working_dir,"Directory is working")
model_path = working_dir + r'\trained_model\NewDataset.h5'
print(model_path)
print(working_dir, " Working DIR")

# Load the pre-trained model with error handling
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function detect largest leaf then crop it and removes background
def detect_and_crop_leaf(image: Image.Image):
    img_np = np.array(image.convert("RGB"))

    # Convert to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define green color range
    lower_green = np.array([25, 10, 10])
    upper_green = np.array([85, 255, 255])

    # Define yellow color range
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([34, 255, 255])

    # Define brown/dark range (tweak based on your leaf samples)
    lower_brown = np.array([5, 25, 25])
    upper_brown = np.array([25, 180, 180])

    # Create masks
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine all masks
    mask = cv2.bitwise_or(green_mask, yellow_mask)
    mask = cv2.bitwise_or(mask, brown_mask)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"Largest contour area: {area}")

    if area < 1000:
        print("Contour too small — skipping.")
        return image

    # Crop around largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Get image dimensions
    img_height, img_width = img_np.shape[:2]

    # Safely compute bounding box with padding
    x1 = max(x - 5, 0)
    y1 = max(y - 5, 0)
    x2 = min(x + w + 5, img_width)
    y2 = min(y + h + 5, img_height)

    # Crop the image and mask safely
    cropped_img = img_np[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Apply mask to remove background (set it to gray)
    result = np.zeros_like(cropped_img)
    gray_background = (100, 100, 100)
    result[:] = gray_background
    result[cropped_mask > 0] = cropped_img[cropped_mask > 0]

    result = np.ascontiguousarray(result.astype(np.uint8))
    return Image.fromarray(result)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # ---------------------------------------
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array
    # --------------------------------------------------

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class_index]  # Extract confidence score
    # -----------------------------------------------------------
    # st.success(f'Predicted class index: {predicted_class_index} with confidence: {confidence_score * 100:.2f}%')
    # -----------------------------------------------------------
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name, confidence_score


def predict_image_class_list(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)[0]  # shape: (num_classes,)

    # Get indices of top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]  # descending order

    # Map indices to class names and confidence scores
    top_predictions = [
        (class_indices.get(str(i), "Unknown"), predictions[i])
        for i in top_indices
    ]
    return top_predictions

# Function to check if the prediction is a tomato leaf
def is_tomato_leaf(predicted_class_name):
    return predicted_class_name.startswith("Tomato")

# Streamlit App
st.title('🍅🌿Tomato Leaf Disease Detection and Prescription System')

uploaded_image = st.file_uploader("Upload your image here ", type=["jpg", "jpeg", "png"])

# Function to get remedy for a given disease
def get_remedy(disease_name):
    try:
        # Configure the Gemini API key
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Initialize the model
        genmodel = genai.GenerativeModel()  # Replace with a supported model
        # Generate remedy using prompt
        response = genmodel.generate_content(
            f"Provide an effective agricultural remedy for {disease_name}. Include actionable steps. Keep the response concise and under 100 words and bulletpoints.")
        # Return generated text
        return response.text
    except Exception as e:
        return f"Error occurred while using Gemini: {str(e)}"

# Streamlit App Integration
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded image")

    with col2:
        if st.button('Classify', key="classify_button"):
            # Apply automatic leaf detection
            cropped_leaf_image = detect_and_crop_leaf(image)

            # Save cropped image temporarily
            cropped_leaf_image_path = "temp_cropped_leaf.png"
            cropped_leaf_image.save(cropped_leaf_image_path)

            # Predict using segmented image
            prediction1, confidence1 = predict_image_class(model, cropped_leaf_image_path, class_indices)
            prediction2, confidence2 = predict_image_class(model, uploaded_image, class_indices)

            print(prediction1,prediction2,confidence1,confidence2)

            with col1:
                st.image(cropped_leaf_image.resize((150, 150)), caption="Detected Leaf Image")

            prediction = prediction1
            if confidence1 < confidence2:
                prediction = prediction2

            if is_tomato_leaf(prediction):
                remedy = get_remedy(prediction)
                st.success(f'Prediction:\n {prediction}')
                st.info(f'Recommended Remedy:\n {remedy}')
            else:
                st.error("The cropped image is not a tomato leaf image.")
