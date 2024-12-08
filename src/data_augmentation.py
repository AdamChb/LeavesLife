# Importing required libraries
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import tensorflow as tf
import os


# Labels
HEALTHY = 0

# Constants
np.random.seed(0)
normalize = Normalizer()


def read_image(file_path):
    try:
        with Image.open(file_path) as image:
            return np.array(image)
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def save_image(image_data, file_path):
    array = image_data.numpy().astype(np.uint8) if isinstance(image_data, tf.Tensor) else image_data.astype(np.uint8)
    with Image.fromarray(image_data) as image:
        image.save(file_path)
        
def data_augmentation(file_path, folder_path):    
    # Read the image
    image_data = read_image(file_path)
    
    # Data augmentation by flipping the image
    flipped_image = tf.image.flip_up_down(image_data)
    
    # Data augmentation by normalizing the image
    # Reshape image data to 2D array
    pixels = image_data.reshape(-1, 3)
    # Scale pixel values
    normalized_image_data = normalize.fit_transform(pixels).reshape(image_data.shape)
    
    # Data augmentation by increasing saturation
    increased_saturation_image = tf.image.adjust_saturation(image_data, 3)
    
    # Data augmentation by grayscale
    grayscale_image = tf.image.rgb_to_grayscale(image_data)
    
    # Data augmentation by rotating and randomly adjusting contrast
    rotated_image = tf.image.rot90(image_data)
    contrast_image = tf.image.random_contrast(rotated_image, lower=0.2, upper=0.8)
    
    # Data augmentation by adding noise + adjusting saturation
    noisy_image = tf.image.random_jpeg_quality(image_data, min_jpeg_quality=5, max_jpeg_quality=20)
    noisy_image = tf.image.adjust_saturation(noisy_image, 2)
    
    # Data augmentation by cropping the image randomly, turning it then grayscaling it
    cropped_image = tf.image.random_crop(image_data, size=(128, 128, 3))
    cropped_image = tf.image.rot90(cropped_image, k=3)
    cropped_image = tf.image.rgb_to_grayscale(cropped_image)
    
    # Save the augmented images
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_image(flipped_image, f"{folder_path}/{base_name}_flipped_image.jpg")
    save_image(normalized_image_data, f"{folder_path}/{base_name}_normalized_image.jpg")
    save_image(increased_saturation_image, f"{folder_path}/{base_name}_increased_saturation_image.jpg")
    save_image(grayscale_image, f"{folder_path}/{base_name}_grayscale_image.jpg")
    save_image(contrast_image, f"{folder_path}/{base_name}_contrast_image.jpg")
    save_image(noisy_image, f"{folder_path}/{base_name}_noisy_image.jpg")
    save_image(cropped_image, f"{folder_path}/{base_name}_cropped_image.jpg")
    
def launch_data_augmentation(folder_paths): 
    for folder_path in folder_paths:
        folder_path = f"../Dataset/{folder_path}"
        list = os.listdir(folder_path)
        for file in list:
            data_augmentation(f"{folder_path}/{file}", folder_path)    
