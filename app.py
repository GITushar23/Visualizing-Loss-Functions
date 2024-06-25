import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to load model
def load_model(loss_function, epoch):
    model = Autoencoder()
    model.load_state_dict(torch.load(f'models/{loss_function}/epoch_{epoch}.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = transform(image)
    image = image.view(-1, 784).float()
    return image

# Sidebar for selecting options
st.sidebar.header("Options")
show_all_losses = st.sidebar.checkbox("Show all loss functions", value=True)
if not show_all_losses:
    loss_function = st.sidebar.selectbox("Loss Function", ["abs", "mse", "smoothabs"])
else:
    loss_function = None

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    digit = None
else:
    digit = st.sidebar.selectbox("Digit", [None] + list(range(10)), index=1)  # Added None option and set default to None

epochs = list(range(1, 31))
selected_epochs = st.sidebar.multiselect("Epochs", epochs, default=[1, 5, 10, 15, 20, 25, 30])

st.title("Autoencoder Visualization")

# Load and display original image
if uploaded_file:
    # Load and display uploaded image
    image = load_image(uploaded_file)
    original_image_caption = "Original Uploaded Image"
elif digit is not None:
    # Load and display default image
    image = load_image(f'actual_images/{digit}.png')
    original_image_caption = f"Original Image (Digit {digit})"
else:
    st.warning("Please upload an image or select a digit.")
    st.stop()

original_image = (image.numpy().reshape(28, 28) * 0.5 + 0.5)  # De-normalize the image to range [0, 1]
st.image(original_image, caption=original_image_caption, use_column_width=False, width=150)

# Define loss function titles
loss_function_titles = {
    "abs": "Regenerated using Absolute Error",
    "mse": "Regenerated using Mean Square Error",
    "smoothabs": "Regenerated using Smooth Absolute Error"
}

loss_functions = ["abs", "mse", "smoothabs"] if show_all_losses else [loss_function]

# Display reconstructed images for selected epochs
for loss_fn in loss_functions:
    st.subheader(loss_function_titles[loss_fn])
    cols = st.columns(len(selected_epochs))
    for col, epoch in zip(cols, selected_epochs):
        model = load_model(loss_fn, epoch)
        with torch.no_grad():
            output = model(image)
            output = output.view(28, 28).numpy()
        col.image(output, caption=f"Epoch {epoch}", use_column_width=False, width=100, clamp=True)
