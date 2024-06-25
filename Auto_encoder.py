import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

# Set manual seed for reproducibility
torch.manual_seed(0)

# Define the data path
DataPath = './mnist/'

# Set to False to avoid downloading MNIST if already downloaded
Download = False

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(DataPath, train=True, download=Download, transform=transform)
test_dataset = datasets.MNIST(DataPath, train=False, download=Download, transform=transform)

# Create DataLoaders
batch_size = 2
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

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

# Define the function to train and save model at each epoch
def train_and_save_model(criterion, optimizer_name, folder_prefix):
    autoencoder = Autoencoder().to(device)
    optimizer = getattr(optim, optimizer_name)(autoencoder.parameters(), lr=0.01)
    
    os.makedirs(f'models/{folder_prefix}', exist_ok=True)
    
    for epoch in range(30):
        print(f"Epoch {epoch + 1} / 30")
        total_loss = 0
        for batch in train_loader:
            input_tensor = batch[0].view(-1, 784).to(device)
            optimizer.zero_grad()
            output = autoencoder(input_tensor)
            current_loss = criterion(output, input_tensor)
            current_loss.backward()
            optimizer.step()
            total_loss += current_loss.item()
        
        total_loss /= len(train_loader)
        print(f"Loss for epoch {epoch + 1}: {total_loss}")
        
        # Save the model
        torch.save(autoencoder.state_dict(), f'models/{folder_prefix}/epoch_{epoch + 1}.pth')

# Train and save models with different loss functions
train_and_save_model(nn.L1Loss(), 'SGD', 'abs')
train_and_save_model(nn.MSELoss(), 'SGD', 'mse')
train_and_save_model(nn.SmoothL1Loss(), 'SGD', 'smoothabs')
