from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from scipy.special import wofz
import pandas as pd
import matplotlib.pyplot as plt

with MPRester(api_key="bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F") as mpr:
    # first retrieve the relevant structure
    structure = mpr.get_structure_by_material_id("mp-2680")

def calc_std_dev(two_theta, tau, wavelength):
    """
    calculate standard deviation based on angle (two theta) and domain size (tau)
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
        wavelength: x-ray wavelength in angstrom
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation
    K = 0.94 ## shape factor
    wavelength = wavelength * 0.1 ## angstrom to nm
    theta = np.radians(two_theta/2.) ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
    return sigma**2



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

min_angle = 5
max_angle = 85

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