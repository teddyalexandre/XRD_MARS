import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class XRayDiffractionDataset(Dataset):
    def __init__(self, csv_files):
        self.data = []
        self.scaler = MinMaxScaler()

        for file in csv_files:
            df = pd.read_csv(file)
            df = self.preprocess(df)
            self.data.append(df)

    def preprocess(self, df):
        # Assuming your columns are named 'angles' and 'intensity'
        angles = df['angles'].values.reshape(-1, 1)
        intensity = df['intensity'].values.reshape(-1, 1)

        # Min-Max scaling
        angles = self.scaler.fit_transform(angles)
        intensity = self.scaler.fit_transform(intensity)

        # Convert to tensor
        angles = torch.tensor(angles, dtype=torch.float32)
        intensity = torch.tensor(intensity, dtype=torch.float32)

        return angles, intensity

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

