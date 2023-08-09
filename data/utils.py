import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.special import wofz


def V(x, alpha, gamma):
    """Voigt function : convolution of Gaussian and Cauchy-Lorentz distributions
        Args:
            alpha and gamma (float) : postive parameters of the Voigt function

        Returns:
            Value of Voigt function at position x
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


### Script to preprocess data before feeding the CNN : Scaling, Padding, outlier management...

def MinMaxScaling(signal):
    """Scales the XRD pattern between 0 and 1 to have the same treatment between data
        Args:
            - signal (list) : list of floats corresponding to the intensity
        Returns:
            A scaled list of floats between 0 and 1
    """
    min_signal = min(signal)
    max_signal = max(signal)
    if abs(min_signal - max_signal) < 1e-3:
        raise Exception("Difference between min and max is close from zero")
    else:
        return [(x - min_signal) / (max_signal - min_signal) for x in signal]


def performPadding(pattern):
    """Pads the intensity with zeros where the range of the signal is not defined
    We get a new signal with angles between 5 and 85 degrees
        Args:
            - pattern (tuple) : tuple of two lists, angles and intensities
        Returns:
            A new pattern with padded angles and intensities"""
    angles, intensities = pattern
    new_angles, new_intensities = [], []
    step = angles[1] - angles[0]  # Step between two angles
    # Build the new lists
    min_angle = angles[0]

    # We pad on the left with zeros
    if min_angle > 5:
        new_angles.append(5)
        new_intensities.append(0)
        while new_angles[-1] < min_angle:
            new_angles.append(new_angles[-1] + step)
            new_intensities.append(0)

    # We concat with the intensities and angles from the unpadded pattern
    new_angles = new_angles + angles
    new_intensities = new_intensities + intensities

    # We pad on the right with zeros
    max_angle = angles[-1]
    if max_angle < 85:
        while new_angles[-1] < 85:
            new_angles.append(new_angles[-1] + step)
            new_intensities.append(0)

    return new_angles, new_intensities


def calculate_xrd_from_cif(cif_path, alpha, gamma, wavelength):
    """
    Calculate the X-ray diffraction (XRD) pattern for a structure from a CIF file and convolve it with a Voigt function.

    Args:
        cif_path (str): Path to the CIF file.
        alpha (float): The Lorentzian component for the Voigt function.
        gamma (float): The Gaussian component for the Voigt function.
        wavelength (str): The type of radiation used for the diffraction. Common choices are 'CuKa' and 'MoKa'.

    Returns:
        dict: A dictionary with two keys: 'XRD Pattern' and 'Space Group'. The 'XRD Pattern' key corresponds to the
              calculated XRD pattern convolved with a Voigt function. The 'Space Group' key corresponds to the space
              group of the structure. If an error occurs during the calculation, both keys will have a value of None.
    """
    try:
        structure = Structure.from_file(cif_path)

        # Initialize an XRD calculator with a specific radiation type
        calculator = XRDCalculator(wavelength=wavelength)

        sga = SpacegroupAnalyzer(structure)
        conventional_structure = sga.get_conventional_standard_structure()
        space_group = sga.get_space_group_symbol()

        # Calculate the XRD pattern
        pattern = calculator.get_pattern(conventional_structure)

        # Prepare the data
        steps = np.linspace(min(pattern.x), max(pattern.x), num=10000)
        norm_signal = np.zeros_like(steps)

        # Replace each peak in the XRD pattern with a Voigt profile
        for i in range(len(pattern.x)):
            peak_position = pattern.x[i]
            peak_intensity = pattern.y[i]
            norm_signal += peak_intensity * V(steps - peak_position, alpha, gamma)

        return {"XRD Pattern": (steps, norm_signal), "Space Group": space_group}

    except Exception as e:
        print(f"Error processing file {cif_path}: {e}")
        return {"XRD Pattern": None, "Space Group": None}
