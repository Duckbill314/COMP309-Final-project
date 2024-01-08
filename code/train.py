import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from skimage.feature import hog
import numpy as np
import os
from sklearn.model_selection import KFold
import copy
from statistics import mean
import random
import matplotlib.pyplot as plt

random.seed(309)

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

# Define a baseline model
class myMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc_flatten = nn.Linear(64 * 128 ** 2, 128)
        self.fc_hog = nn.Linear(2304, 128)
        self.fc_colour = nn.Linear(256, 128)
        self.fc_comb = nn.Linear(128 * 3, 128)
        self.fc_output = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x, x_hog, x_colour):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_flatten(x)
        x_hog = self.fc_hog(x_hog)
        x_colour = self.fc_colour(x_colour)
        x_comb = torch.cat((x, x_hog, x_colour), dim=1)
        x_comb = self.fc_comb(x_comb)
        x_comb = self.relu(x_comb)
        x_output = self.fc_output(x_comb)
        return x_output

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
    
# Define the CNN model (this one uses sigmoid activation)
class sigmoidCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_flatten = nn.Linear(64 * 32 ** 2, 128)
        self.fc_hog = nn.Linear(2304, 128)
        self.fc_colour = nn.Linear(256, 128)
        self.fc_comb = nn.Linear(128 * 3, 128)
        self.fc_output = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, x_hog, x_colour):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_flatten(x)
        x_hog = self.fc_hog(x_hog)
        x_colour = self.fc_colour(x_colour)
        x_comb = torch.cat((x, x_hog, x_colour), dim=1)
        x_comb = self.fc_comb(x_comb)
        x_comb = self.sigmoid(x_comb)
        x_output = self.fc_output(x_comb)
        return x_output
    
# Define the CNN model (this one uses log softmax activation)
class softmaxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_flatten = nn.Linear(64 * 32 ** 2, 128)
        self.fc_hog = nn.Linear(2304, 128)
        self.fc_colour = nn.Linear(256, 128)
        self.fc_comb = nn.Linear(128 * 3, 128)
        self.fc_output = nn.Linear(128, 3)
        self.softmax = nn.LogSoftmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, x_hog, x_colour):
        x = self.conv1(x)
        x = self.softmax(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.softmax(x)
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_flatten(x)
        x_hog = self.fc_hog(x_hog)
        x_colour = self.fc_colour(x_colour)
        x_comb = torch.cat((x, x_hog, x_colour), dim=1)
        x_comb = self.fc_comb(x_comb)
        x_comb = self.softmax(x_comb)
        x_output = self.fc_output(x_comb)
        return x_output
    
# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a consistent size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image values
])

# Load the training dataset
data_root = os.getcwd() + '/traindata'
custom_dataset = ImageFolder(root=data_root, transform=transform)

# Function for generating folds for cross validation
def generate_folds(dataset, k=5, batch_size=32, rand=309):
    kf = KFold(n_splits=k, shuffle=True, random_state=rand)
    folds = []
    for train_index, val_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        folds.append((train_loader, val_loader))
    return folds

# Function for training one epoch
def train_epoch(model, data_loader, optimizer, criterion, is_CNN=True):
    model.train()
    running_loss = 0.0
    correct = 0
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        if is_CNN: # The CNN's first convolutional layer takes different dimensions than the MLP's first linear connection layer
            outputs = model(inputs, extract_hog(inputs.permute(0, 2, 3, 1)), extract_colour(inputs.permute(0, 2, 3, 1)))
        else:
            outputs = model(inputs.permute(0, 2, 3, 1), extract_hog(inputs.permute(0, 2, 3, 1)), extract_colour(inputs.permute(0, 2, 3, 1)))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        scores, predictions = torch.max(outputs.data, 1)
        correct += (predictions == labels).sum().item()
    return running_loss, correct

# Function for validating one epoch
def val_epoch(model, data_loader, criterion, is_CNN=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            if is_CNN: # The CNN's first convolutional layer takes different dimensions than the MLP's first linear connection layer
                outputs = model(inputs, extract_hog(inputs.permute(0, 2, 3, 1)), extract_colour(inputs.permute(0, 2, 3, 1)))
            else:
                outputs = model(inputs.permute(0, 2, 3, 1), extract_hog(inputs.permute(0, 2, 3, 1)), extract_colour(inputs.permute(0, 2, 3, 1)))
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            scores, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()
    return running_loss, correct

# Function for a whole learning cycle with k folds and the specified number of epochs
def learn(model, criterion, optimizer, folds, num_epochs=20, conv_iter=5, conv_thresh=0.05, acc_thresh=0.95):
    models = []
    losses = []
    accs = []
    
    k = len(folds)
    train_size = (k-1)/k * len(custom_dataset)
    val_size = 1/k * len(custom_dataset)
    
    for i in range(k):
        print("Start of fold {}/{}".format(i+1, k))
        (train_loader, val_loader) = folds[i]
        epoch = 0
        prev_acc = 0
        conv = 0
        early_stop = False
        # Early stopping if convergence is detected or an accuracy threshold is reached
        while (epoch < num_epochs) and (conv < conv_iter) and (early_stop == False):
            epoch += 1
            train_loss, train_corr = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_corr = val_epoch(model, val_loader, criterion)
            
            train_loss_avg = train_loss/train_size
            train_acc = train_corr/train_size
            val_loss_avg = val_loss/val_size
            val_acc = val_corr/val_size
            
            print("Epoch {}".format(epoch))
            print("Training | Loss: {}, Accuracy: {}".format(train_loss_avg, train_acc))
            print("Validation | Loss: {}, Accuracy: {}".format(val_loss_avg, val_acc))

            # Convergence detection 
            # (% difference between current and previous accuracy is less than a threshold, "conv_iter" times in a row)
            if (prev_acc == 0) or (abs((val_acc - prev_acc)/prev_acc) > conv_thresh):
                conv = 0
            else:
                conv += 1
            prev_acc = val_acc

            # High accuracy achieved
            if (val_acc > acc_thresh):
                early_stop = True

        models.append(copy.deepcopy(model))
        losses.append(val_loss_avg)
        accs.append(val_acc)

    print("Average of all folds | Loss: {}, Accuracy: {}".format(sum(losses)/k, sum(accs)/k))
    return models, losses, accs

# Revised learning function to generate the final models and statistics
def learn_final(model, criterion, optimizer, folds, num_epochs=20, conv_iter=5, conv_thresh=0.05, acc_thresh=0.95, is_CNN=True):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    k = len(folds)
    train_size = (k-1)/k * len(custom_dataset)
    val_size = 1/k * len(custom_dataset)

    # Takes k folds but only uses one, randomly selected
    (train_loader, val_loader) = folds[random.randrange(k)]
    epoch = 0
    while (epoch < num_epochs):
        epoch += 1
        if is_CNN: # The CNN's first convolutional layer takes different dimensions than the MLP's first linear connection layer
            train_loss, train_corr = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_corr = val_epoch(model, val_loader, criterion)
        else:
            train_loss, train_corr = train_epoch(model, train_loader, optimizer, criterion, is_CNN=False)
            val_loss, val_corr = val_epoch(model, val_loader, criterion, is_CNN=False)
            
        train_loss_avg = train_loss/train_size
        train_acc = train_corr/train_size
        val_loss_avg = val_loss/val_size
        val_acc = val_corr/val_size

        # Instead of only reporting final results like the other learner, 
        # this one reports the trend over all epochs
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc)
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)

    return copy.deepcopy(model), train_losses, train_accs, val_losses, val_accs

# Tuning block
losses = []
accuracies = []

models = [reluCNN()]
criteria = [nn.CrossEntropyLoss()]
folds_all = [generate_folds(custom_dataset, batch_size=32),
             generate_folds(custom_dataset, batch_size=64),
             generate_folds(custom_dataset, batch_size=128)]

for model in models:
    for criterion in criteria:
        for folds in folds_all:
            model_copies = [copy.deepcopy(model), copy.deepcopy(model), copy.deepcopy(model), copy.deepcopy(model)]
            optimizers = [optim.Adam(model_copies[0].parameters(), lr=0.001), optim.Adam(model_copies[1].parameters(), lr=0.01), 
                         optim.SGD(model_copies[2].parameters(), lr=0.001), optim.SGD(model_copies[3].parameters(), lr=0.01)]
            for i in range(len(optimizers)):
                _, loss, acc = learn(model_copies[i], criterion, optimizers[i], folds)
                losses.append(loss)
                accuracies.append(acc)

# Evaluation block 
processed_losses = []
processed_accuracies = []

for i in range(3):
    for j in range(4):
        avg_loss = mean(losses[i*4+j])
        avg_acc = mean(accuracies[i*4+j])
        processed_losses.append(avg_loss)
        processed_accuracies.append(avg_acc)

# Tuning block 2
losses_2 = []
accuracies_2 = []
models = [reluCNN(), sigmoidCNN(), softmaxCNN()]
criterion = nn.CrossEntropyLoss()
folds = generate_folds(custom_dataset, batch_size=128)

for model in models:
    model_copy = copy.deepcopy(model)
    optimizer = optim.Adam(model_copy.parameters(), lr=0.001)
    _, loss, acc = learn(model_copy, criterion, optimizer, folds)
    losses_2.append(loss)
    accuracies_2.append(acc)

# Evaluation block 2
processed_losses_2 = []
processed_accuracies_2 = []

for i in range(3):
    avg_loss = mean(losses_2[i])
    avg_acc = mean(accuracies_2[i])
    processed_losses_2.append(avg_loss)
    processed_accuracies_2.append(avg_acc)

# Final MLP model block
model = myMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
folds = generate_folds(custom_dataset, batch_size=128)
_, mlp_train_loss, mlp_train_acc, mlp_val_loss, mlp_val_acc = learn_final(model, criterion, optimizer, folds, num_epochs=30, is_CNN=False)

figure, axis = plt.subplots(1, 2)
epochs = range(1, 31)

figure_width, figure_height = figure.get_size_inches()
figure.set_size_inches(figure_width*1.5, figure_height)

axis[0].plot(epochs, mlp_train_loss, label="training")
axis[0].plot(epochs, mlp_val_loss, label="validation")
axis[0].set_title("Average loss of baseline MLP model")
axis[0].set_ylabel("loss")
axis[0].set_xlabel("epoch")
axis[0].legend()

axis[1].plot(epochs, mlp_train_acc, label="training")
axis[1].plot(epochs, mlp_val_acc, label="validation")
axis[1].set_title("Accuracy of baseline MLP model")
axis[1].set_ylabel("accuracy")
axis[1].set_xlabel("epoch")
axis[1].legend()

plt.savefig("baseline-mlp")
plt.show()

# Final CNN model block
model = reluCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
folds = generate_folds(custom_dataset, batch_size=128)
cnn_model, cnn_train_loss, cnn_train_acc, cnn_val_loss, cnn_val_acc = learn_final(model, criterion, optimizer, folds, num_epochs=30, is_CNN=True)

figure, axis = plt.subplots(1, 2)
epochs = range(1, 31)

figure_width, figure_height = figure.get_size_inches()
figure.set_size_inches(figure_width*1.5, figure_height)

axis[0].plot(epochs, cnn_train_loss, label="training")
axis[0].plot(epochs, cnn_val_loss, label="validation")
axis[0].set_title("Average loss of CNN model")
axis[0].set_ylabel("loss")
axis[0].set_xlabel("epoch")
axis[0].legend()

axis[1].plot(epochs, cnn_train_acc, label="training")
axis[1].plot(epochs, cnn_val_acc, label="validation")
axis[1].set_title("Accuracy of CNN model")
axis[1].set_ylabel("accuracy")
axis[1].set_xlabel("epoch")
axis[1].legend()

plt.savefig("cnn-model")
plt.show()

torch.save(cnn_model, "model.pth")