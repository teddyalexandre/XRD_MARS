"""
This script contains a few useful functions that are used to treat data
"""

import numpy as np
import os
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.special import wofz


def V(x, alpha, gamma):
    """Voigt function : convolution of Gaussian and Cauchy-Lorentz distributions.
        Args:
            alpha and gamma (float) : postive parameters of the Voigt function

        Returns:
            Value of Voigt function at position x (float)
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)

def ScatteringVectorList(angles, E0=17):
    """Returns the scattering vector for a corresponding angle"""
    wavelength = 12.39842 / E0
    return [(4 * np.pi * np.sin(np.radians(theta)) / wavelength) for theta in angles]


def Voigt(x, crystallite_size, wavelength, theta, K=0.9):
    """Voigt function : convolution of Gaussian and Cauchy-Lorentz distributions.
        Args:
            crystallite_size (float): Size of the crystallite.
            wavelength (float): X-ray wavelength.
            theta (float): Bragg angle.
            K (float, optional): Scherrer constant. Defaults to 0.9.

        Returns:
            Value of Voigt function at position x (float)
    """
    # Calculate parameters from Scherrer equation
    beta = K * wavelength / (crystallite_size * np.cos(theta))

    # Plug the parmaters into the pseudo-voigt function
    sigma = beta / (2 * np.sqrt(2 * np.log(2)))
    gamma = beta / 2

    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)

### Script to preprocess data before feeding the CNN : Scaling, Padding, outlier management...

def MinMaxScaling(signal):
    """Scales the XRD pattern between 0 and 1 to have the same treatment between data.
        Args:
            - signal (list) : list of floats corresponding to the intensity

        Returns:
            - list of floats between 0 and 1 (rescaling) (list)
    """
    min_signal = min(signal)
    max_signal = max(signal)
    if abs(min_signal - max_signal) < 1e-3:
        raise Exception("Difference between min and max is close from zero")
    else:
        return [(x - min_signal) / (max_signal - min_signal) for x in signal]



def calculate_xrd_from_cif(cif_path, crystallite_size, wavelength, K=0.9):
    """
    Calculate the X-ray diffraction (XRD) pattern for a structure from a CIF file and convolve it with a Voigt function.

    Args:
        cif_path (str): Path to the CIF file.
        alpha (float): The Lorentzian component for the Voigt function.
        gamma (float): The Gaussian component for the Voigt function.
        wavelength (str): The type of radiation used for the diffraction. Common choices are 'CuKa' and 'MoKa'.

    Returns:
        (dict): A dictionary with the following keys: 'Formula', 'Angles', 'Intensities', 'Space Group' and 'Crystal System'.
              'Formula' corresponds to the name of the chemical species.
              The 'Angles' and 'Intensities' keys correspond to the calculated XRD pattern convolved with a Voigt function.
              The 'Space Group' key corresponds to the space group of the structure. Idem for 'Crystal System'.
              If an error occurs during the calculation, all keys will have a corresponding value equal to None.
    """
    try:
        structure = Structure.from_file(cif_path)
        formula_name = os.path.splitext(os.path.basename(cif_path))[0]
        # Initialize an XRD calculator with a specific radiation type
        calculator = XRDCalculator(wavelength=wavelength)

        sga = SpacegroupAnalyzer(structure)
        conventional_structure = sga.get_conventional_standard_structure()
        space_group = sga.get_space_group_symbol()
        crystal_system = sga.get_crystal_system()

        dict_crystal_system = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                               'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7}

        crystal_system_id = dict_crystal_system[crystal_system]

        # Calculate the XRD pattern
        pattern = calculator.get_pattern(conventional_structure)

        # Prepare the data
        steps = np.linspace(min(pattern.x), max(pattern.x), num=10000)
        norm_signal = np.zeros_like(steps)

        # Replace each peak in the XRD pattern with a Voigt profile
        for i in range(len(pattern.x)):
            peak_position = pattern.x[i]
            peak_intensity = pattern.y[i]
            theta = np.radians(peak_position / 2)

            # Use the Voigt function
            norm_signal += peak_intensity * Voigt(steps - peak_position, crystallite_size, wavelength, theta, K)

        return {"Formula": formula_name, "Angles": steps, "Intensities": norm_signal, "Space Group": space_group,
                "Crystal System": crystal_system_id}

    except Exception as e:
        print(f"Error processing file {cif_path}: {e}")
        return {"Formula": None, "Angles": None, "Intensities": None, "Space Group": None, "Crystal System": None}
