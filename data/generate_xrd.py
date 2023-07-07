from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from scipy.special import wofz
import matplotlib.pyplot as plt



with MPRester(api_key="bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F") as mpr:
    # first retrieve the relevant structure
    structure = mpr.get_structure_by_material_id("mp-2680")


# important to use the conventional structure to ensure
# that peaks are labelled with the conventional Miller indices
sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()

# this example shows how to obtain an XRD diffraction pattern
# these patterns are calculated on-the-fly from the structure
calculator = XRDCalculator(wavelength="MoKa")
pattern = calculator.get_pattern(conventional_structure)
angles = pattern.x
intensities = pattern.y

# Prepare the data
steps = np.linspace(min(pattern.x), max(pattern.x), num=1000)
norm_signal = np.zeros_like(steps)

# Define the Voigt function
def V(x, alpha, gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

# Define the parameters for the Voigt function
alpha = 0.1  # Gaussian component HWHM
gamma = 0.1  # Lorentzian component HWHM

# Replace each peak in the XRD pattern with a Voigt profile
for i in range(len(pattern.x)):
    peak_position = pattern.x[i]
    peak_intensity = pattern.y[i]
    norm_signal += peak_intensity * V(steps - peak_position, alpha, gamma)

# Create a matplotlib line plot
plt.plot(steps, norm_signal)

# Set the plot title and axis labels
plt.title('XRD Pattern')
plt.xlabel('2 theta')
plt.ylabel('Intensity')

# Show the plot
plt.show()