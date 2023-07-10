import matplotlib.pyplot as plt
import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.special import wofz


def V(x, alpha, gamma):
    """Voigt function."""
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


def calculate_xrd_pattern(material_id, api_key, alpha, gamma, wavelength):
    """Calculate the XRD pattern for a material and convolve it with a Voigt function."""
    with MPRester(api_key=api_key) as mpr:
        # Retrieve the structure
        structure = mpr.get_structure_by_material_id(material_id)

    # Use the conventional structure to ensure that peaks are labelled with the conventional Miller indices
    sga = SpacegroupAnalyzer(structure)
    conventional_structure = sga.get_conventional_standard_structure()

    # Obtain an XRD diffraction pattern
    calculator = XRDCalculator(wavelength=wavelength)
    pattern = calculator.get_pattern(conventional_structure)

    # Prepare the data
    steps = np.linspace(min(pattern.x), max(pattern.x), num=1000)
    norm_signal = np.zeros_like(steps)

    # Replace each peak in the XRD pattern with a Voigt profile
    for i in range(len(pattern.x)):
        peak_position = pattern.x[i]
        peak_intensity = pattern.y[i]
        norm_signal += peak_intensity * V(steps - peak_position, alpha, gamma)

    return norm_signal, steps


def process_material(material_id, api_key, alpha, gamma, wavelength):
    # Calculate the XRD pattern
    norm_signal = calculate_xrd_pattern(material_id, api_key, alpha, gamma, wavelength)

    # Fetch the space group
    with MPRester(api_key=api_key) as mpr:
        structure = mpr.get_structure_by_material_id(material_id)
    sga = SpacegroupAnalyzer(structure)
    space_group = sga.get_space_group_symbol()

    return {"XRD Pattern": norm_signal, "Space Group": space_group}