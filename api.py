from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

# Define the autoencoder model for MNIST
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

# Define the autoencoder model using ResNet for general images
class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define transformations
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load models
def load_mnist_model(loss_function, epoch):
    model = Autoencoder()
    model.load_state_dict(torch.load(f'models/{loss_function}/epoch_{epoch}.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def load_resnet_model(loss_function, epoch):
    model = ResNetAutoencoder()
    model.decoder.load_state_dict(torch.load(f'resnetencoder_model/resnet_autoencoder_{loss_function}/epoch_{epoch}.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Image processing functions
def process_mnist_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    image = transform_mnist(image)
    image = image.view(-1, 784).float()
    return image

def process_resnet_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = transform_resnet(image).unsqueeze(0).float()
    return image

# API routes
@app.route('/process_mnist', methods=['POST'])
def process_mnist():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    loss_function = request.form.get('loss_function', 'mse')
    epoch = int(request.form.get('epoch', 1))

    image = process_mnist_image(image_file.read())
    model = load_mnist_model(loss_function, epoch)

    with torch.no_grad():
        output = model(image)
        output = output.view(28, 28).numpy()

    output_image = Image.fromarray((output * 255).astype(np.uint8))
    buffered = io.BytesIO()
    output_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({'processed_image': img_str})

@app.route('/process_resnet', methods=['POST'])
def process_resnet():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    loss_function = request.form.get('loss_function', 'mseloss')
    epoch = int(request.form.get('epoch', 1))

    image = process_resnet_image(image_file.read())
    model = load_resnet_model(loss_function, epoch)

    with torch.no_grad():
        output = model(image)
        output = output.squeeze(0).permute(1, 2, 0).numpy()
        output = ((output * 0.5 + 0.5) * 255).astype(np.uint8)

    output_image = Image.fromarray(output)
    buffered = io.BytesIO()
    output_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({'processed_image': img_str})

if __name__ == '__main__':
    app.run(debug=True)