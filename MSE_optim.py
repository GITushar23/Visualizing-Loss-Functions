import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from torch.utils.tensorboard
from Network import Net  # Assuming Net is defined in Network.py

# Set to False to avoid downloading MNIST if already downloaded
Download = False

# Determine device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def train_net(net, data_loader, optimizer, epoch, train_error, writer):
    net = net.to(device)
    
    for tepoch in range(epoch):
        print(f'Epoch {tepoch+1}/{epoch}')
        total_loss = 0
        for input_tensor, _ in data_loader:
            input_tensor = input_tensor.to(device)  
            input_tensor = input_tensor.view(-1, 784).float()
            optimizer.zero_grad()
            output = net(input_tensor)
            loss = criterion(output, input_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(data_loader)
        print(f'Training Loss = {total_loss:.4f}')
        train_error[optimizer_name_to_idx[optimizer_name], tepoch] = total_loss
        
        # Log training loss to TensorBoard
        writer.add_scalar(f'Training Loss/{optimizer_name}', total_loss, tepoch)

# Load the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.MNIST(root='./mnist', train=True, download=Download, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Initialize the error tensor to store errors for each optimizer
num_optimizers = 3
num_epochs = 10
train_error = torch.zeros((num_optimizers, num_epochs))

# Define the models
models = {
    'Adagrad': Net(),
    'Adam': Net(),
    'SGD': Net()
}

# Move models to device
for model in models.values():
    model.to(device)

# Define the optimizers
optimizers = {
    'Adagrad': optim.Adagrad(models['Adagrad'].parameters(), lr=1e-4),
    'Adam': optim.Adam(models['Adam'].parameters(), lr=1e-4),
    'SGD': optim.SGD(models['SGD'].parameters(), lr=1e-4)
}

# Define the criterion (loss function)
criterion = nn.MSELoss()

# Dictionary to map optimizer names to indices in train_error tensor
optimizer_name_to_idx = {
    'Adagrad': 0,
    'Adam': 1,
    'SGD': 2
}

#------------------------------------------------------------------
#      Training       ---------------------------------------------
#------------------------------------------------------------------

print('Training models...')

# Create TensorBoard writer
writer = SummaryWriter()

for optimizer_name, model in models.items():
    print(f'Training with {optimizer_name} optimizer...')
    optimizer = optimizers[optimizer_name]
    train_net(model, train_loader, optimizer, epoch=num_epochs, train_error=train_error, writer=writer)

# Close the TensorBoard writer
writer.close()

# Example of accessing training errors
for i, optimizer_name in enumerate(models.keys()):
    print(f'Training errors for {optimizer_name}:')
    print(train_error[i])
    print()
