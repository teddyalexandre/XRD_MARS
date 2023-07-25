# Import basic libraries
import torch
import torchvision
import torchvision.transforms as transforms  # For data transformation
import torch.nn as nn   # For all neural network modules and functions
from torch.utils.data import DataLoader  # Better data management
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import optim # All optimizers (SGD, Adam...)

from sklearn.model_selection import train_test_split # For splitting the dataset into training and test datasets
from sklearn.preprocessing import MinMaxScaler # Scaling the numerical data so that they are comparable
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random


def conv_output_size(input_size, stride, kernel_size, padding=0):
    """Function that computes the number of output features
        Args:
        - input_size : number of input features
        - stride
        - kernel_size
        - padding, default value equal to 0
        Returns:
        - number of output features
    """
    return int((input_size + 2*padding - kernel_size) / stride) + 1


def vector_size(params):
    """Function that computes the size of the flattened layer, at the end of the 3rd convolution/maxpool
       layer structure :
        Args:
        - params : dictionary with all of the parameters (strides, kernel sizes, input size) of the model
        Returns:
        - the size of the flattened layer
    """
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
    """Class which models the CNN -> input is a 1D image (the XRD spectra signal), works as a vector"""

    def __init__(self, params, output_size=230):
        """Constructor of the class ConvNN : 3 convolutional/pooling layers, 3 fully connected layers
            Args:
            - params : list of the parameters of the CNN (strides, kernel sizes...)
            - output_size : number of neurons in the output layer (here 230 -> number of space groups)
        """
        super(ConvNN, self).__init__()

        kernel_sizes = params["kernels"]
        strides = params["strides"]

        # Instanciate 3 layers of 1D convolution/relu/pooling
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=params["conv_channels"], 
            kernel_size=kernel_sizes[0], stride=strides[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=params["conv_channels"], out_channels=params["conv_channels"],
            kernel_size=kernel_sizes[1], stride=strides[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1))

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=params["conv_channels"], out_channels=params["conv_channels"],
            kernel_size=kernel_sizes[2], stride=strides[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1))

        S = vector_size(params)         # Size of the flatten layer

        # Instanciate fully connected layers for classification task
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S, 2300),
            nn.Linear(2300, 1150),
            nn.Linear(1150, output_size)
            # nn.Softmax(dim = 1)       # If we want the probabilities as outputs
        )

    def forward(self, x):
        """Method which simulates the forward propagation in the CNN through the layers
            Args:
                - x : input of the CNN
            Returns :
                - out : output layer, output_size number of neurons (230 by default)"""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc_layers(out)
        return out

### Next step : train the model on data to ensure that it works fine

def train(train_loader, cnn, learning_rate, num_epochs):
    """Trains the CNN on training data
        Args:
        - train_loader = data that can be loaded in batches
        - cnn : the cnn model
        - learning rate : step between each iteration to minimize the loss function
        - num_epochs : number of iterations of training
    """
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Begin training :")
    for epoch in range(num_epochs):
        running_loss = 0.0

        # For each batch in the loader
        for inputs, labels in train_loader:
            # Set the gradients back to 0
            optimizer.zero_grad()

            # Apply the model
            outputs = cnn(inputs)

            # Compare between the outputs from the CNN and the labels
            loss = criterion(outputs, labels)

            # Compute the gradients
            loss.backward()

            # Performs a single optimization step (parameter update)
            optimizer.step()
            
            running_loss += loss
        
        # Print the average loss for one epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    print("Finished training !")


def compute_accuracy(test_loader, cnn, num_epochs):
    """Compute the accuracy of the model
        Args:
        - test_loader : data that can be loaded in batches
        - cnn : the cnn model
        - num_epochs : number of iterations
    """
    for epoch in range(num_epochs):
        accuracy = 0
        count = 0
        for inputs, labels in test_loader:
            y_pred = cnn(inputs)
            accuracy += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
        accuracy /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, accuracy*100))


## From CSV/Parquet file, we train the model and see how it performs
## Looking to perform distributed data parallelism with Pytorch to speed up training

if __name__ == "__main__":

    # Define the hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Load the data (specify the path)
    dataset = "..."

    # Split the raw data into train set and test set
    trainset, testset = train_test_split(dataset, test_size=0.25, random_state=111)

    # Create data loaders for train and test sets
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Configure the device (computes on GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instance of the class CNN
    cnn = ConvNN().to(device)

    # train the cnn on training set
    train(trainloader, cnn, learning_rate, num_epochs)

    # Compute the accuracy of the model
    compute_accuracy(testloader, cnn, num_epochs)