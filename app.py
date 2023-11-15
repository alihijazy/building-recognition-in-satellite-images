import streamlit as st
import requests
import cv2
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import Image
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import load_model
import pandas as pd

st.set_page_config(page_title="Building detection tool in satellite images")


#comment


def slice_and_predict_masks(image_path, model, output_dir, threshold=0.7):
    # Load the image
    original_image = cv2.imread(image_path)

    # Get the dimensions of the original image
    height, width, _ = original_image.shape

    # Define tile size
    tile_size = 256

    # Calculate the number of tiles in both dimensions
    num_tiles_height = (height + tile_size - 1) // tile_size
    num_tiles_width = (width + tile_size - 1) // tile_size

    # Create an empty array to store the stitched masks
    stitched_masks = np.zeros((height, width), dtype=np.uint8)

    # Loop through each tile
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            # Calculate the coordinates for slicing
            y1 = i * tile_size
            y2 = min((i + 1) * tile_size, height)
            x1 = j * tile_size
            x2 = min((j + 1) * tile_size, width)

            # Slice the image tile
            tile = original_image[y1:y2, x1:x2]

            # Check if padding is needed
            pad_y = tile_size - (y2 - y1)
            pad_x = tile_size - (x2 - x1)

            if pad_y > 0 or pad_x > 0:
                # Add black padding if needed
                tile = cv2.copyMakeBorder(tile, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Preprocess the tile (e.g., resize and normalize)
            preprocessed_tile = cv2.resize(tile, (256, 256))
            preprocessed_tile = preprocessed_tile / 255.0  # Normalize to [0, 1]

            # Predict the mask for the tile using the model
            predicted_mask = model.predict(np.expand_dims(preprocessed_tile, axis=0))[0]

            # Resize the predicted mask to match the tile size
            predicted_mask = cv2.resize(predicted_mask, (x2 - x1, y2 - y1))

            # Threshold the predicted mask to create a binary mask
            binary_mask = (predicted_mask > threshold).astype(np.uint8) * 255

            # Add the binary mask to the stitched masks
            stitched_masks[y1:y2, x1:x2] = binary_mask

    # Save the stitched mask as an image
    output_mask_path = os.path.join(output_dir, 'stitched_mask.png')
    cv2.imwrite(output_mask_path, stitched_masks)

    # Plot the original image and stitched masks side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(stitched_masks, cmap='gray')
    plt.title('Stitched Binary Mask')

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def download_bing_maps_image(lat, lon, api_key, filename='bing_maps_image.jpg'):
    # Bing Maps Static Imagery API URL format
    URL = "https://dev.virtualearth.net/REST/v1/Imagery/Map/{imagerySet}/{centerPoint}/{zoomLevel}?mapSize={mapSize}&key={key}"

    # Specify the parameters
    imagerySet = "Aerial"
    centerPoint = f"{lat},{lon}"
    zoomLevel = 19
    mapSize = "1024,1024"

    # Create the full URL by formatting the parameters
    full_url = URL.format(
        imagerySet=imagerySet,
        centerPoint=centerPoint,
        zoomLevel=zoomLevel,
        mapSize=mapSize,
        key=api_key
    )

    # Send a request to the Bing Maps Static Imagery API
    response = requests.get(full_url)

    if response.status_code == 200:
        # Write the image to a file
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded: {filename}")
    else:
        print(f"Failed to download image: HTTP {response.status_code}")
        
trained_models_path = r"C:\Users\13472\Documents\Flatiron\phase 5\notebooks\models\pretrained\pretrained_model_10_epoch.h5"
model = load_model(trained_models_path)

api_key="AizKFVajXANq2_jb7LG0cBjXNcnrwJ3eD2nJJAgyAdzheAUDSoaffHA-wQwyHPHj"

def main():
    st.title("Building Detection in Satellite images")

    # Streamlit widgets to capture coordinates
    lat_lon_input = st.text_input("Enter Latitude and Longitude (comma-separated)", "37.840226, -122.271299")
    lat, lon = map(float, lat_lon_input.split(','))

    # Download and process button
    if st.button("Download Image and Process"):
            
         download_bing_maps_image(lat, lon, api_key=api_key)
         slice_and_predict_masks(r"bing_maps_image.jpg", model, "")
         pred_path = r"C:\Users\13472\Documents\Flatiron\phase 5\notebooks\stitched_mask.png"
         pred_img = Image.open(pred_path)
         # Display the image in the Streamlit app
         
         bing_img_path = r"C:\Users\13472\Documents\Flatiron\phase 5\notebooks\bing_maps_image.jpg"
         bing_img = Image.open(bing_img_path)
         st.image([bing_img, pred_img], caption=['Bing Image', 'Predicted Mask'], use_column_width=True)
main()

