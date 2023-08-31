"""
This script proceeds to do inference and measures the time required for it
"""

from models.cnn_model import ConvNN
from data.dataset import XRDPatternDataset
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader  # Better data management
import time

# Configure the device (computes on GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instance of the class CNN
params = {
    "kernels": [100, 50, 25],
    "strides": [5, 5, 2],
    "input_size": 10000,
    "conv_channels": 64
}
output_size = 7

# Define the hyperparameters
batch_size = 82
num_running_processes = 64

# Modify if you want the 230 space groups (7 -> 230)
cnn = ConvNN(params, 7+1)
cnn.load_state_dict(torch.load("./models/cnn_optimized_7.pt"))
cnn.eval()

cnn.double()

# Specify the Parquet file to be parsed
dataset = XRDPatternDataset("/home/experiences/grades/alexandret/ruche/share-temp/XRD_MARS_datasets/pow_xrd.parquet")

trainset, testset = random_split(dataset, [0.75, 0.25])
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_running_processes)
testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_running_processes)

dataiter = iter(testloader)
_, intensities, labels = next(dataiter)

# Inference on data
start = time.time()
with torch.no_grad():
    outputs = cnn(torch.unsqueeze(intensities, 1))
    pred_labels = torch.argmax(outputs, dim=1)
    correct = (pred_labels == labels).float().sum()
    count = len(labels)
    accuracy = float(correct / count)
    print(f"Model accuracy: {accuracy:.2f}")

end = time.time()

print(f"The time inference is {end - start} seconds.")