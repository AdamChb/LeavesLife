import numpy as np
import tensorflow as tf
from PIL import Image
import io


def predict_image(image_bytes, model_path=f"../models/final_model/trained_final_model.keras"):
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)
    
    # Charger et prétraiter l'image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    image = image / 255.0  # Normaliser l'image

    # Prédire les probabilités des classes
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    probability = np.max(predictions)

    return predicted_class[0], probability
