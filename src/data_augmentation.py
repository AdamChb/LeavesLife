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
    contrast_image = tf.image.stateless_random_contrast(rotated_image, lower=0.2, upper=0.8, seed=(1, 2))
    
    # Data augmentation by adding noise + adjusting saturation
    noisy_image = tf.image.random_jpeg_quality(image_data, min_jpeg_quality=5, max_jpeg_quality=20, seed=0)
    noisy_image = tf.image.adjust_saturation(noisy_image, 2)
    
    # Data augmentation by cropping the image randomly, turning it then grayscaling it
    cropped_image = tf.image.random_crop(image_data, size=(128, 128, 3), seed=0)
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
        for file in os.listdir(folder_path):
            data_augmentation(f"{folder_path}/{file}", folder_path)
    



# image_path = 'test.JPG'
# image_data = read_image(image_path)

# print('Image Shape: {} \nRows: {}, \nColumns: {}, \nRGB values: {}'.format(image_data.shape, *image_data.shape))

# normalize = Normalizer()


# plt.figure(figsize=(12, 18))

# dim = (4, 2)

# # Base Image
# plt.subplot(dim[0], dim[1], 1)
# plt.title('Base Image')
# plt.imshow(image_data)

# # Data augmentation by flipping the image
# plt.subplot(dim[0], dim[1], 2)
# plt.title('Flipped Image')
# plt.imshow(tf.image.flip_up_down(image_data))

# # Data augmentation by normalizing the image
# # Reshape image data to 2D array
# pixels = image_data.reshape(-1, 3)
# # Scale pixel values
# normalized_image_data = normalize.fit_transform(pixels).reshape(image_data.shape)
# # Display the image
# plt.subplot(dim[0], dim[1], 3)
# plt.title('Normalized Image')
# plt.imshow(normalized_image_data)

# # Data augmentation by increasing saturation
# plt.subplot(dim[0], dim[1], 4)
# plt.title('Increased Saturation')
# plt.imshow(tf.image.adjust_saturation(image_data, 3))

# # Data augmentation by grayscale
# plt.subplot(dim[0], dim[1], 5)
# plt.title('Grayscale Image')
# plt.imshow(tf.image.rgb_to_grayscale(image_data))
# plt.colorbar()

# # Data augmentation by rotating and randomly adjusting contrast
# plt.subplot(dim[0], dim[1], 6)
# plt.title('Rotation + Random Contrast')
# plt.imshow(tf.image.stateless_random_contrast(tf.image.rot90(image_data), lower=0.2, upper=0.8, seed=(1, 2)))

# # Data augmentation by adding noise + adjusting saturation
# plt.subplot(dim[0], dim[1], 7)
# plt.title('Noisy + Saturated Image')
# plt.imshow(tf.image.adjust_saturation(tf.image.random_jpeg_quality(image_data, min_jpeg_quality=5, max_jpeg_quality=20, seed=0), 2))


# # Data augmentation by cropping the image randomly, turning it then grayscaling it
# plt.subplot(dim[0], dim[1], 8)
# plt.title('Random Crop + Grayscale')
# plt.imshow(tf.image.rgb_to_grayscale(tf.image.rot90(tf.image.random_crop(image_data, size=(128, 128, 3), seed=0), k=3)))
# plt.colorbar()  
    
