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
# Data augmentation functions
#-----------------------------------#
 
# Importing required libraries
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from PIL import Image, UnidentifiedImageError


# Constants
HEALTHY = 0
np.random.seed(0)
normalize = Normalizer()

# Function to read an image
def read_image(file_path):
    """
    Read an image from a file path and return it as a numpy array

    Arguments:
        - file_path (String): Path of the image
    """
    
    try:
        # Open the image file
        with Image.open(file_path) as image:
            return np.array(image)
    except UnidentifiedImageError:
        # If the image file is not recognized
        print(f"Error: Cannot identify image file {file_path}")
        return None
    except Exception as e:
        # If an error occurs
        print(f"Error: {e}")
        return None
    
    
# Function to save an image
def save_image(image_data, file_path):
    """
    Save an image to a file
    
    Arguments:
        - image_data (numpy.ndarray or tf.Tensor): Image data
        - file_path (String): Path where the image will be saved
    """
    
    # Convert the image data to a numpy array if it is a tensor
    array = image_data.numpy().astype(np.uint8) if isinstance(image_data, tf.Tensor) else image_data.astype(np.uint8)
    # Save the image
    with Image.fromarray(image_data) as image:
        image.save(file_path)
        
        
# Function to perform data augmentation
def data_augmentation(file_path, folder_path):   
    """
    Perform data augmentation on an image and save the augmented images
    in a specified folder.
    
    Arguments:
        - file_path (String): Path of the image
        - folder_path (String): Path of the folder where the augmented images will be saved
    """
    
    # Read the image
    image_data = read_image(file_path)
    
    # Data augmentation by flipping the image
    flipped_image = tf.image.flip_up_down(image_data)
    
    # Data augmentation by normalizing the image
    pixels = image_data.reshape(-1, 3)
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
    

# Function to launch data augmentation    
def launch_data_augmentation(folder_paths): 
    """
    Launch data augmentation on all images in the specified folders
    
    Arguments:
        - folder_paths (List): List of folder paths where the images are stored
    """
    
    for folder_path in folder_paths:
        # Get the list of files in the folder
        folder_path = f"../Dataset/{folder_path}"
        list = os.listdir(folder_path)
        # Perform data augmentation on each image
        for file in list:
            data_augmentation(f"{folder_path}/{file}", folder_path)    
