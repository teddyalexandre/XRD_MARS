import multiprocessing
import os

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from data.utils import V


class SimpleGen(object):
    """
    Base class used to process pymatgen from cif files and calculate diffractogram and
    pymatgen structure object characteristics like space group and bravais lattics
    """

    def __init__(self, cif_dir, wavelength, resolution=4500, min_angle=10.0, max_angle=80.0, min_domain_size=1,
                 max_domain_size=100):
        """

        Args:
            cif_dir: directory containing CIFs
        """
        self.data = pd.DataFrame({}, columns=["Intensity", "Angles", "Space_Group"])
        self.num_spectra = 0
        self.calculator = XRDCalculator(wavelength=wavelength)
        self.wavelength = wavelength
        self.resolution = resolution
        self.num_cpu = multiprocessing.cpu_count()
        self.ref_dir = cif_dir
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size

    def load_structures(self, alpha, gamma):
        data = []
        # Loop over all CIF files in the directory
        for filename in os.listdir(self.ref_dir):
            if filename.endswith('.cif'):
                try:
                    cif_path = os.path.join(self.ref_dir, filename)
                    structure = Structure.from_file(cif_path)
                    sga = SpacegroupAnalyzer(structure)
                    conventional_structure = sga.get_conventional_standard_structure()
                    space_group = sga.get_space_group_symbol()
                    pattern = self.calculator.get_pattern(conventional_structure,
                                                          two_theta_range=(self.min_angle, self.max_angle))
                    steps = np.linspace(min(pattern.x), max(pattern.x), num=self.resolution)
                    norm_signal = np.zeros_like(steps)
                    for i in range(len(pattern.x)):
                        peak_position = pattern.x[i]
                        peak_intensity = pattern.y[i]
                        norm_signal += peak_intensity * V(steps - peak_position, alpha, gamma)

                    data.append({"Intensity": norm_signal, "Angles": steps, "Space_Group": space_group})

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    data.append({"Intensity": None, "Angles": None, "Space_Group": None})

        self.data = pd.DataFrame(data)
        self.num_spectra = len(data)

    def calc_std_dev(self, tau):
        """
        Args:
            tau: domain size in nm

        Returns:
        """
        two_theta = self.max_angle - self.min_angle
        K = 0.9  ## shape factor
        wavelength = self.calculator.wavelength * 0.1
        theta = np.radians(two_theta / 2.)
        beta = (K * wavelength) / (np.cos(theta) * tau)

        sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degree(beta)
        return 2 * sigma

    def save_to_paquet(self, filename):
        assert len(self.data) > 0
        self.data.to_parquet(filename)

