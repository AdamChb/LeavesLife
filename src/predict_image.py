import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Seed used to trin the wanted model
seed = 0

def predict_image(image_bytes, model_path=f"../models/{seed}/trained_model_{seed}.keras"):
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

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "../Dataset/Apple___Apple_scab/29ab8216-ec38-4efd-9c77-21068fa899a4___FREC_Scab 3241.JPG"
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    predicted_class, probability = predict_image(image_bytes)
    print(f"Predicted class: {predicted_class}, Probability: {probability}")