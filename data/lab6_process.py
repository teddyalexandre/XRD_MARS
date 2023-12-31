"""
This script fetches data from the experimental software McXtrace.

Here, I only created data from the LaB6 crystal, but it should be generalized.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import MinMaxScaling


# Read from raw LaB6 data
data = np.loadtxt("./datasets/detector_diffraction_1693311944.th")

# Create DataFrame
df = pd.DataFrame(data, columns=["Angles", "Intensities", "I_err", "N"])

# We add the scattering vector magnitude column corresponding to each angle
angles = np.array(df["Angles"])
E0 = 16.99
l = 12.39842 / E0
q_vec = [(4 * np.pi * np.sin(np.radians(theta))) / l for theta in angles]
df.insert(1, "Q", q_vec)

# We get 10000 values to match the CNN's inputs
idx = np.round(np.linspace(0, df.shape[0] - 1, 10000)).astype(int)
df = df.iloc[idx]

# We plot the LaB6 spectra of the normalized intensity against the scattering vector magnitude
plt.plot(np.array(df["Q"]), MinMaxScaling(np.array(df["Intensities"])))
plt.xlabel("Scattering vector magnitude (nm-1)")
plt.ylabel("Normalized intensity (arbitrary unit)")
plt.title("LaB6 diffraction spectra (X ray)")
plt.show()

# Write processed data in a file text (to compare space groups, 
# one should create a Parquet file from this data and insert LaB6 space group. The CNN
# would then predict it, once passed as a torch.Tensor)
df.to_csv('./datasets/lab6_spectra_modified.txt', sep=' ', index=False)