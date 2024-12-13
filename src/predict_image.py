#-----------------------------------#
# Machine Learning Project
# LeavesLife: Plant Disease Detection
# Dates: 2024-11-27 - 2024-12-12
#
# Authors:
# - Mathias BENOIT
# - Adam CHABA
# - Eva MAROT
# - Sacha PORTAL
#
# File Description: 
# Function to predict the class of an image
#-----------------------------------#

# Importing required libraries
import io
import numpy as np
from PIL import Image
import tensorflow as tf


# Function to predict the class of an image
def predict_image(image_bytes, model_path=f"../models/1.3/trained_model_3.keras"):
    """
    Predict the class of an image using a trained model.
    
    Parameters:
        - image_bytes: bytes, image file content
        - model_path: str, path to the trained model
    """
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    # Add batch dimension
    image = np.expand_dims(image, axis=0) 
    # Normalization
    image = image / 255.0  

    # Predict the class of the image
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    probability = np.max(predictions)

    # Return the predicted class and the probability
    return predicted_class[0], probability
