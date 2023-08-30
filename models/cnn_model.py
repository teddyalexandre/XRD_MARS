"""
This script writes and launches the CNN from the input data.
"""

import torch  # Import basic libraries
from torch import nn  # For all neural network modules and functions
from torch import optim  # All optimizers (SGD, Adam...)
from torch.utils.data import random_split, DataLoader  # Better data management
from data.dataset import XRDPatternDataset  # Import custom Dataset
import matplotlib.pyplot as plt  # Plot loss and accuracy vs epochs
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import time


def vector_size(params):
    """Compute the size of the flattened output after passing through the convolutional layers.
        Args:
            - params (dict) : parameters of the CNN.
        
        Returns:
            - the size of the flattened layer, after the last Conv/MaxPool layer (int)."""
    input_size = params["input_size"]
    kernel_sizes = params["kernels"]
    strides = params["strides"]
    conv_channels = params["conv_channels"]

    # Compute the output size after each convolutional and pooling layer
    for i in range(3):
        input_size = (input_size - kernel_sizes[i]) // strides[i] + 1
        if i == 0:
            input_size = (input_size - 3) // 2 + 1
        else:
            input_size = (input_size - 3) // 1 + 1

            # Multiply by the number of channels to get the flattened size
    return input_size * conv_channels


class ConvNN(nn.Module):
    """Class which models the CNN -> input is a 1D image (the XRD spectra signal), works as a vector"""

    def __init__(self, params, output_size):
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
            nn.BatchNorm1d(num_features=params["conv_channels"]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=params["conv_channels"], out_channels=params["conv_channels"],
                      kernel_size=kernel_sizes[1], stride=strides[1]),
            nn.BatchNorm1d(num_features=params["conv_channels"]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=3, stride=1))

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=params["conv_channels"], out_channels=params["conv_channels"],
                      kernel_size=kernel_sizes[2], stride=strides[2]),
            nn.BatchNorm1d(num_features=params["conv_channels"]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=3, stride=1))

        S = vector_size(params)  # Size of the flatten layer

        # Instanciate fully connected layers for classification task
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S, 2300),
            nn.Dropout(p=0.2),
            nn.Linear(2300, 1150),
            nn.Dropout(p=0.2),
            nn.Linear(1150, output_size))
        # nn.Softmax(dim=1))  # The probabilities to belong to a group space as outputs

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


def initialize_weights(model):
    """Initialize the weights of the model."""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


### Next step : train the model on data to ensure that it works fine

def train(train_loader, cnn, learning_rate, num_epochs, device):
    """Trains the CNN on training data.
        Args:
            - train_loader (DataLoader) : data that can be loaded in batches
            - cnn (ConvNN) : the cnn model
            - learning_rate (float) : step between each iteration to minimize the loss function
            - num_epochs (int) : number of iterations of training
            - device (torch.device) : device on which the calculations are made (CUDA GPU or CPU)
        Returns:
            - trainloss_list (list) : list of the loss values over epochs
    """
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainloss_list = []
    print("Begin training :")
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0

        # For each batch in the loader
        for angles, intensities, labels in train_loader:
            # Set the gradients back to 0
            optimizer.zero_grad()

            inputs = intensities
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply the model
            outputs = cnn(torch.unsqueeze(inputs, 1))

            # Compare between the outputs from the CNN and the labels
            loss = criterion(outputs, labels)

            # Compute the gradients
            loss.backward()

            # Performs a single optimization step (parameter update)
            optimizer.step()

            running_loss += loss

        # Print the average loss for one epoch
        epoch_loss = running_loss / len(train_loader)
        trainloss_list.append(epoch_loss.item())
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Finished training !")
    return trainloss_list


def compute_accuracy(test_loader, cnn, num_epochs, device):
    """Compute the accuracy of the model on the test dataset
        Args:
            - test_loader (DataLoader) : data that can be loaded in batches
            - cnn (ConvNN) : the cnn model
            - num_epochs (int) : number of iterations
        Returns:
            - accuracy_list (list) : list of accuracy values over epochs
    """
    accuracy_list = []
    for epoch in range(1, num_epochs + 1):
        correct = 0
        count = 0
        for angles, intensities, labels in test_loader:
            inputs = intensities
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = cnn(torch.unsqueeze(inputs, 1))
            pred_labels = torch.argmax(y_pred, dim=1)
            correct += (pred_labels == labels).float().sum()
            count += len(labels)
        accuracy = float(correct / count)
        accuracy_list.append(accuracy)
        print("Epoch %d: model accuracy %.2f" % (epoch, accuracy))
    return accuracy_list


def train_and_evaluate(train_loader, test_loader, cnn, learning_rate, num_epochs, device):
    """Trains the CNN on training data and evaluates accuracy after every 5 epochs.
        Args:
            - train_loader (DataLoader) : training data that can be loaded in batches
            - test_loader (DataLoader) : test data that can be loaded in batches
            - cnn (ConvNN) : the cnn model
            - learning_rate (float) : step between each iteration to minimize the loss function
            - num_epochs (int) : number of iterations of training
            - device (torch.device) : device on which the calculations are made (CUDA GPU or CPU)
        Returns:
            - trainloss_list (list) : list of the loss values over epochs
            - accuracy_list (list) : list of accuracy values over epochs
    """
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainloss_list = []
    accuracy_list = []
    print("Begin training :")

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0

        # Training loop
        for angles, intensities, labels in train_loader:
            optimizer.zero_grad()
            inputs = intensities.to(device)
            labels = labels.to(device)
            outputs = cnn(torch.unsqueeze(inputs, 1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss

        epoch_loss = running_loss / len(train_loader)
        trainloss_list.append(epoch_loss.item())
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

        correct = 0
        count = 0
        y_true = []
        y_pred_list = []
        with torch.no_grad():
            for angles, intensities, labels in test_loader:
                inputs = intensities.to(device)
                labels = labels.to(device)
                y_pred = cnn(torch.unsqueeze(inputs, 1))
                y_true.extend(labels.cpu().tolist())
                pred_labels = torch.argmax(y_pred, dim=1)
                y_pred_list.extend(pred_labels.cpu().tolist())
                correct += (pred_labels == labels).float().sum()
                count += len(labels)
        accuracy = float(correct / count)
        accuracy_list.append(accuracy)
        print(f"Model accuracy: {accuracy:.2f}")

        if epoch == num_epochs:
            # Build confusion matrix
            cf_matrix = confusion_matrix(y_true, y_pred_list)
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in range(1, 8)],
                                 columns = [i for i in range(1, 8)])
            plt.figure(figsize = (12,8))
            sns.heatmap(df_cm, annot=True)
            plt.savefig('./models/conf_matrix_7.png')

    print("Finished training and evaluation!")
    return trainloss_list, accuracy_list


def objective(trial):
    # Hyperparameters to be optimized
    batch_size = trial.suggest_int('batch_size', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    # num_epochs = trial.suggest_int('num_epochs', 50, 150)
    conv_channels = trial.suggest_int('conv_channels', 32, 128)
    # Including kernels and strides in the hyperparameter optimization
    kernels = trial.suggest_categorical('kernels', [[100, 50, 25], [50, 25, 12], [25, 12, 6]])
    strides = trial.suggest_categorical('strides', [[5, 5, 2], [4, 4, 2], [3, 3, 1]])
    conv_channels = trial.suggest_int('conv_channels', 32, 128)

    num_epochs = 15
    num_running_processes = 64

    # ... [rest of your data loading and device configuration code]
    dataset = XRDPatternDataset("/home/experiences/grades/alexandret/ruche/share-temp/XRD_MARS_datasets/pow_xrd_val.parquet")

    nb_space_groups = dataset.nb_space_group
    nb_crystal_systems = dataset.nb_crystal_systems
    print("number of space groups:", nb_space_groups)
    print("number of crystal systems:", nb_crystal_systems)

    # Split the raw data into train set and test set
    trainset, testset = random_split(dataset, [0.75, 0.25])

    # Create data loaders for train and test sets
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_running_processes)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_running_processes)

    # Configure the device (computes on GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = {
        "kernels": kernels,
        "strides": strides,
        "input_size": 10000,
        "conv_channels": conv_channels
    }

    cnn = ConvNN(params, nb_crystal_systems + 1).to(device)
    cnn = cnn.double()
    initialize_weights(cnn)

    trainloss_list, accuracy_list = train_and_evaluate(trainloader, testloader, cnn, learning_rate, num_epochs, device)

    # Return the negative accuracy because Optuna tries to minimize the objective
    return -accuracy_list[-1]


if __name__ == "__main__":
    # Define the hyperparameters
    batch_size = 82
    learning_rate = 0.0005012297485033033
    num_epochs = 10
    num_running_processes = 64

    start = time.time()
    # Create the dataset
    dataset = XRDPatternDataset("/home/experiences/grades/alexandret/ruche/share-temp/XRD_MARS_datasets/pow_xrd.parquet")

    nb_space_groups = dataset.nb_space_group
    nb_crystal_systems = dataset.nb_crystal_systems
    print("number of space groups:", nb_space_groups)
    print("number of crystal systems:", nb_crystal_systems)

    # Split the raw data into train set and test set
    trainset, testset = random_split(dataset, [0.75, 0.25])

    # Create data loaders for train and test sets
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_running_processes)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_running_processes)

    # Configure the device (computes on GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instance of the class CNN
    params = {
        "kernels": [100, 50, 25],
        "strides": [5, 5, 2],
        "input_size": 10000,
        "conv_channels": 64
    }
    cnn = ConvNN(params, nb_space_groups + 1).to(device)
    cnn = cnn.double()
    initialize_weights(cnn)

    # train the cnn on training set
    # trainloss_list = train(trainloader, cnn, learning_rate, num_epochs, device)

    # Compute the accuracy of the model
    # accuracy_list = compute_accuracy(testloader, cnn, num_epochs, device)

    trainloss_list, accuracy_list = train_and_evaluate(trainloader, testloader, cnn, learning_rate, num_epochs, device)

    print("Model's state_dict:")
    for param_tensor in cnn.state_dict():
        print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())


    end = time.time()
    #print(f"The CNN training and inference took : {end - start} seconds to execute, i.e. {(end - start) / 60} minutes, i.e. {(end - start) / 3600} hours.")
    # Plotting
    plt.figure(figsize=(12, 5))

    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), trainloss_list, ls="-")
    plt.title("Training loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.ylim(bottom=0)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    # Since accuracy is computed every 5 epochs, we adjust the x-axis accordingly
    plt.plot(range(1, num_epochs+1), accuracy_list, ls="-")
    plt.title("Accuracy curve on test set")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
