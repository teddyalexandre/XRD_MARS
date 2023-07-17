# Import basic libraries
import torch
import torchvision
import torchvision.transforms as transforms  # For data transformation
import torch.nn as nn   # For all neural network modules and functions
from torch.utils.data import DataLoader  # Better data management
from torch import optim # All optimizers (SGD, Adam...)
from sklearn.model_selection import train_test_split # For splitting the dataset into training and test datasets
from sklearn.preprocessing import MinMaxScaler # Scaling the numerical data so that they are comparable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random


# Compute the number of output features
def conv_output_size(input_size, stride, kernel_size, padding=0):
    return int((input_size + 2*padding - kernel_size) / stride + 1)

# Computes the size of the flattened layer at the end of the 3rd conv/pool layer structure
def vector_size(params):
    strides = params["strides"]
    kernel_sizes = params["kernels"]
    input_size = params["input_size"]
    s = conv_output_size(input_size, strides[0], kernel_sizes[0], kernel_sizes[0]/2)
    s = conv_output_size(s, 2, 3)
    s = conv_output_size(s, strides[1], kernel_sizes[1], kernel_sizes[1]/2)
    s = conv_output_size(s, 3, 3)
    s = conv_output_size(s, strides[2], kernel_sizes[2], kernel_sizes[2]/2)
    s = conv_output_size(s, 3, 3)
    return s*params['conv_channels']


class ConvNN(nn.Module):
    """Class which models the CNN -> operates on a 1D image (the XRD spectra) -> vector"""

    def __init__(self, params):
        """Constructor of the class ConvNN : 3 convolutional/pooling layers, 3 fully connected layers"""
        super(ConvNN, self).__init__()

        # Instanciate 3 convolutional layers
        kernel_sizes = params["kernels"]
        strides = params["strides"]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=params["conv_channels"], kernel_size=kernel_sizes[0],
                               stride=strides[0])
        self.conv2 = nn.Conv1d(in_channels=params["conv_channels"], out_channels=params["conv_channels"],
                               kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv3 = nn.Conv1d(in_channels=params["conv_channels"], out_channels=params["conv_channels"],
                               kernel_size=kernel_sizes[2], stride=strides[2])

        # Instanciate 3 (max) pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1)

        # Instanciate the ReLU activation function (which introduces non-linearity)
        self.relu = nn.ReLU()

        # Instanciate the softmax activation function (for the output layer) if we want the probabilities
        # self.softmax = nn.Softmax(dim=1)

        # Instanciate the Dropout function
        # self.dropout1 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.5)

        # Finally, instanciate 3 fully connected layers
        S = vector_size(params)
        self.fc1 = nn.Linear(S, 2300)
        self.fc2 = nn.Linear(2300, 1150)
        self.fc3 = nn.Linear(1150, 230)

    def forward(self, x):
        """Method which simulates the forward propagation in the CNN through the layers"""
        x = self.relu(self.conv1(x))
        # x = self.dropout1(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        # x = self.dropout1(x)
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        # x = self.dropout1(x)
        x = self.pool3(x)

        # Fully connected layers
        x = nn.Flatten(x)
        x = self.fc1(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.droput2(x)
        return self.fc3(x)


### Next step : train the model on some data to ensure its good functioning

## From CSV file (imposed format) we train the model and see how it performs

if __name__ == "__main__":

    # Define the hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Load the data (specify the path)
    dataset = "..."

    # Split the raw data into train set and test set
    trainset, testset = train_test_split(dataset, test_size=0.25, random_state=111)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Configure the device (computes on GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn = ConvNN().to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0

        # For each batch in the loader
        for inputs, labels in enumerate(trainloader):
            # Set the gradients back to 0
            optimizer.zero_grad()

            # Apply the model
            outputs = cnn(inputs)

            # Compare between the outputs from the CNN and the labels
            loss = criterion(outputs, labels)

            # Compute the gradients
            loss.backward()

            optimizer.step()
            
            running_loss += loss
        
        # Print the average loss for this epoch
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
