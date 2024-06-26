import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import os

# Define the autoencoder model
class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()
        # Load pre-trained ResNet18 and remove the final fully connected layer
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Keep until the second last layer
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Define the decoder
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
        x = x.view(x.size(0), 512, 7, 7)  # Adjust dimensions for decoder
        x = self.decoder(x)
        return x

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the input size expected by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for pre-trained ResNet
])

# Load the Tiny ImageNet dataset
train_dir = 'tiny-imagenet-200/train'
val_dir = 'tiny-imagenet-200/val'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetAutoencoder().to(device)
criterion = nn.MSELoss()  # Example loss function
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

num_epochs = 10  # Example number of epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0
        num_val_samples = 0
        for data in val_loader:
            img, _ = data
            img = img.to(device)
            outputs = model(img)
            loss = criterion(outputs, img)
            val_loss += loss.item() * img.size(0)
            num_val_samples += img.size(0)
    
    val_loss /= num_val_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss}")
