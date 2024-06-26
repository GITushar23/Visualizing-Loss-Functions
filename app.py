import streamlit as st
import requests
import base64
from PIL import Image
import io
import math
import numpy as np

# Define the API URL
API_URL = "http://localhost:5000"  # Change this if your API is hosted elsewhere

# Function to process MNIST image
def process_mnist_image(image_data, loss_function, epoch):
    url = f"{API_URL}/process_mnist"
    files = {'image': ('image.png', image_data, 'image/png')}
    data = {'loss_function': loss_function, 'epoch': str(epoch)}
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        return base64.b64decode(response.json()['processed_image'])
    else:
        st.error(f"Error processing image: {response.text}")
        return None

# Function to process ResNet image
def process_resnet_image(image_data, loss_function, epoch):
    url = f"{API_URL}/process_resnet"
    files = {'image': ('image.png', image_data, 'image/png')}
    data = {'loss_function': loss_function, 'epoch': str(epoch)}
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        return base64.b64decode(response.json()['processed_image'])
    else:
        st.error(f"Error processing image: {response.text}")
        return None

# Function to create GIF (placeholder - implement on server side if needed)
def create_gif(image_data, loss_fn, model_type, image_width=100):
    # This function should be implemented on the server side
    # For now, we'll just return a message
    return f"GIF creation for {model_type} using {loss_fn} is not implemented in this demo"

# Function to display GIF
def display_gif(gif_path):
    st.image(gif_path, use_column_width=True)

# Sidebar for selecting options
st.sidebar.header("Options")
show_all_losses = st.sidebar.checkbox("Show all loss functions", value=True)

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    digit = None
    if not show_all_losses:
        loss_function_resnet = st.sidebar.selectbox("ResNet Loss Function", ["l1loss", "mseloss", "smoothl1loss"])
else:
    digit = st.sidebar.selectbox("Digit", [None] + list(range(10)), index=1)
    if not show_all_losses:
        loss_function_mnist = st.sidebar.selectbox("MNIST Loss Function", ["abs", "mse", "smoothabs"])

# New option for GIF
show_gif = st.sidebar.checkbox("Show GIF of all epochs", value=False)

# New option for specific epochs
show_specific_epochs = st.sidebar.checkbox("Show images for specific epochs", value=True)
if show_specific_epochs:
    max_epochs = 30 if not uploaded_file else 20  # 30 for MNIST, 20 for ResNet
    specific_epochs = st.sidebar.multiselect("Select epochs", range(1, max_epochs + 1), default=[1, 5, 10, 15, 20])

# Add user input for image width
if (uploaded_file and not show_gif) or not uploaded_file:
    default_width = 150 if uploaded_file else 100
    image_width = st.sidebar.number_input("Image width for epochs (pixels)", min_value=50, max_value=300, value=default_width)

st.title("Autoencoder Visualization")

def display_epoch_images(epochs, process_function, image_data, loss_fn, is_resnet=False, image_width=100):
    epochs_per_row = min(len(epochs), 5)
    num_rows = math.ceil(len(epochs) / epochs_per_row)
    
    for row in range(num_rows):
        cols = st.columns(epochs_per_row)
        for col in range(epochs_per_row):
            epoch_index = row * epochs_per_row + col
            if epoch_index < len(epochs):
                epoch = epochs[epoch_index]
                output = process_function(image_data, loss_fn, epoch)
                if output is not None:
                    output_image = Image.open(io.BytesIO(output))
                    cols[col].image(output_image, caption=f"Epoch {epoch}", use_column_width=False, width=image_width)

# Determine which model to use based on upload or MNIST selection
if uploaded_file:
    # Load and display uploaded image using ResNet model
    image_data = uploaded_file.getvalue()
    original_image_caption = "Original Uploaded Image"
    model_type = "resnet"

    # Display original image for ResNet
    st.image(image_data, caption=original_image_caption, use_column_width=False, width=224)

    loss_functions_resnet = ["l1loss", "mseloss", "smoothl1loss"] if show_all_losses else [loss_function_resnet]

    # Display reconstructed images or GIF using ResNet autoencoder
    for loss_fn in loss_functions_resnet:
        st.subheader(f"ResNet Autoencoder using {loss_fn}")

        if show_gif:
            gif_path_resnet = create_gif(image_data, loss_fn, "resnet")
            display_gif(gif_path_resnet)
        elif show_specific_epochs:
            display_epoch_images(specific_epochs, process_resnet_image, image_data, loss_fn, is_resnet=True, image_width=image_width)
        else:
            st.warning("Please select an option to visualize the results.")

elif digit is not None:
    # Load and display default image using MNIST model
    with open(f'actual_images/{digit}.png', 'rb') as f:
        image_data = f.read()
    original_image_caption = f"Original Image (Digit {digit})"
    model_type = "mnist"

    # Display original image for MNIST
    st.image(image_data, caption=original_image_caption, use_column_width=False, width=150)

    loss_functions_mnist = ["abs", "mse", "smoothabs"] if show_all_losses else [loss_function_mnist]

    # Display reconstructed images or GIF using MNIST autoencoder
    for loss_fn in loss_functions_mnist:
        st.subheader(f"MNIST Autoencoder using {loss_fn}")

        if show_gif:
            gif_path_mnist = create_gif(image_data, loss_fn, "mnist", image_width)
            display_gif(gif_path_mnist)
        elif show_specific_epochs:
            display_epoch_images(specific_epochs, process_mnist_image, image_data, loss_fn, image_width=image_width)
        else:
            st.warning("Please select an option to visualize the results.")
else:
    st.warning("Please upload an image or select a digit.")