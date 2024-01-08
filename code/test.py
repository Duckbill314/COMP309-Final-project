import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from skimage.feature import hog
import numpy as np
import os

# HOG (gradient) feature extractor
def extract_hog(x):
    instances = []
    for i in x:
        features = hog(i.numpy(), cells_per_block=(1, 1), channel_axis=-1)
        instances.append(features)
    return torch.tensor(np.array(instances))

# Colour feature extractor
def extract_colour(x):
    instances = []
    for i in x:
        features = np.histogram(i.numpy(), bins=256, range=(0, 256), density=True)[0]
        instances.append(np.float32(features))
    return torch.tensor(np.array(instances))

# Define the CNN model (this one uses ReLU activation)
class reluCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_flatten = nn.Linear(64 * 32 ** 2, 128)
        self.fc_hog = nn.Linear(2304, 128)
        self.fc_colour = nn.Linear(256, 128)
        self.fc_comb = nn.Linear(128 * 3, 128)
        self.fc_output = nn.Linear(128, 3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, x_hog, x_colour):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_flatten(x)
        x_hog = self.fc_hog(x_hog)
        x_colour = self.fc_colour(x_colour)
        x_comb = torch.cat((x, x_hog, x_colour), dim=1)
        x_comb = self.fc_comb(x_comb)
        x_comb = self.relu(x_comb)
        x_output = self.fc_output(x_comb)
        return x_output
    
# Function for evaluating
def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs, extract_hog(inputs.permute(0, 2, 3, 1)), extract_colour(inputs.permute(0, 2, 3, 1)))
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            scores, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()
    return running_loss, correct

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a consistent size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image values
])

# Load the training dataset
data_root = os.getcwd() + '/testdata'
custom_dataset = ImageFolder(root=data_root, transform=transform)

# Define the model, dataset, and loss function
model = torch.load("model.pth")
test_loader = DataLoader(custom_dataset, batch_size=128, shuffle=True)
criterion = nn.CrossEntropyLoss()

# Evaluate on the training data
total_loss, total_correct = evaluate(model, test_loader, criterion)

# Report statistics
dataset_size = len(custom_dataset)
if (dataset_size != 0):
    avg_loss = total_loss/dataset_size
    accuracy = total_correct/dataset_size * 100
    print("Average loss: {:.3f} | Accuracy: {:.2f}".format(avg_loss, accuracy))