import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn as skl
import random as rd

import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data
from torch import optim

class ConvNN(nn.Module):
    """Class which models the CNN with 2 convolutional/max pooling layers -> operates on a 1D image (the XRD spectra)"""
    def __init__(self):
        """Constructor of the class ConvNN :
            Args:
            - num_classes : the number of independent classes at the end (number of neurons in the output layer)
        """
        super(ConvNN, self).__init__()
        
        # Instanciate the convolutional layers
        self.conv1 = nn.Conv1d(in_channels = 4501, out_channels = 64, kernel_size = 50, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = 2251, out_channels = 64, kernel_size = 25, stride = 3)
        
        # Instanciate the pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size = 3, stride = 2)
        self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 3)
        
        # Instanciate the ReLU activation function (which introduces non-linearity)
        self.relu = nn.ReLU()
        
        # Then we construct fully connected layers
        self.fc1 = nn.Linear(8064, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 1)
        
    def forward(self, x):
        """Method which simulates the forward propagation in the CNN through the layers"""
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Fully connected layers
        x = nn.Flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
    
### Next step : train the model on some data to ensure its good functioning

## From CSV file (imposed format) we train the model and see how it performs
column_names = ["Angles", "Intensity"]
data = pd.read_csv("Uchucchacuaite.csv", names=column_names)
data.dropna(inplace=True)
data.head()


angles = data["Angles"].values
intensity = data["Intensity"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(angles, intensity, test_size=0.2, random_state=42)


# Define the parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

cnn = ConvNN()
optimizer = optim.Adam(cnn.parameters(), lr=3e-4)       # We employ the stochastic gradient descent based optimizer : Adam
criterion = nn.CrossEntropyLoss()                       # We want to diminish the cross entropy loss


# Model training
for epoch in range(num_epochs):
    inputs, labels = data
    optimizer.zero_grad()
    
    # Compute the forward pass
    outputs = cnn(inputs)
    
    # Compute the loss function
    loss = criterion(outputs)
    
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()