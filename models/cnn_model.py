# Import basic libraries
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn as skl

from random import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch import optim

class ConvNN(nn.Module):
    """Class which models the CNN -> operates on a 1D image (the XRD spectra) -> vector"""
    def __init__(self):
        """Constructor of the class ConvNN : 3 convolutional/pooling layers, 3 fully connected layers"""
        super(ConvNN, self).__init__()
        
        # Instanciate 3 convolutional layers
        self.conv1 = nn.Conv1d(in_channels = 10001, out_channels = 80, kernel_size = 100, stride = 5)
        self.conv2 = nn.Conv1d(in_channels = 1000, out_channels = 80, kernel_size = 50, stride = 5)
        self.conv3 = nn.Conv1d(in_channels = 66, out_channels = 80, kernel_size = 25, stride = 2)
        
        # Instanciate 3 (max) pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size = 3, stride = 2)
        self.pool2 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        self.pool3 = nn.MaxPool1d(kernel_size = 3, stride = 1)
        
        # Instanciate the ReLU activation function (which introduces non-linearity)
        self.relu = nn.ReLU()
        
        # Instanciate the softmax activation function (for the output layer)
        #self.softmax = nn.Softmax(dim=1)
        
        # Instanciate the Dropout function
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        
        # Finally, instanciate 3 fully connected layers
        self.fc1 = nn.Linear(880, 2300)
        self.fc2 = nn.Linear(2300, 1150)
        self.fc3 = nn.Linear(1150, 230)
        
    def forward(self, x):
        """Method which simulates the forward propagation in the CNN through the layers"""
        x = self.relu(self.conv1(x))
        #x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        #x = self.dropout1(x)
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x))
        #x = self.dropout1(x)
        x = self.pool3(x)
        
        # Fully connected layers
        x = nn.Flatten(x)
        x = self.fc1(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #x = self.droput2(x)
        return self.fc3(x)
    
### Next step : train the model on some data to ensure its good functioning

## From CSV file (imposed format) we train the model and see how it performs
column_names = ["Angles", "Intensity"]
data = pd.read_csv("Uchucchacuaite.csv", names=column_names)
data.dropna(inplace=True)
data.head()

angles = np.array(data["Angles"].values)
intensity = np.array(data["Intensity"].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(angles, intensity, test_size=0.2, random_state=42)

# Define the parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

cnn = ConvNN()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

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