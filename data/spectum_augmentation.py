"""
This script performs data augmentation on ingested data (script unused yet)
"""

import random

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal

from utils import V


class Augmentator(object):
    def __int__(self, cif_file,
                wavelength,
                resolution=4501,
                max_strain=0.04,
                min_domain_size=1,
                max_domain_size=100,
                min_angle=10.0,
                max_angle=80.0
                ):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.resolution = resolution
        self.max_strain = max_strain
        self.calculator = XRDCalculator(wavelength=wavelength)
        self.wavelength = wavelength
        self.file_path = cif_file
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.strain_range = np.linspace(0.0, max_strain, 100)
        self.structure = None
        self.load_structure()
        self.possible_domains = np.linspace(min_domain_size, max_domain_size, 100)

    def load_structure(self):
        struct = Structure.from_file(self.file_path)
        sga = SpacegroupAnalyzer(struct)
        self.structure = sga.get_conventional_standard_structure()

    @property
    def lattice(self):
        return self.structure.lattice

    @property
    def matrix(self):
        return self.structure.lattice.matrix

    @property
    def sg(self):
        return self.structure.get_space_group_info()[1]

    @property
    def strained_struc(self):
        ref_struc = self.structure.copy()
        if ref_struc.is_ordered:
            xtal_struc = pyxtal()
            xtal_struc.from_seed(ref_struc)
            current_strain = random.choice(self.strain_range)
            xtal_struc.apply_perturbation(d_lat=current_strain, d_coor=0.0)
            pmg_struc = xtal_struc.to_pymatgen()
            return pmg_struc
        else:
            ref_struc.lattice = self.strained_lattice
            return ref_struc

    @property
    def diag_range(self):
        max_strain = self.max_strain
        return np.linspace(1 - max_strain, 1 + max_strain, 1000)

    @property
    def off_diag_range(self):
        max_strain = self.max_strain
        return np.linspace(0 - max_strain, 0 + max_strain, 1000)

    @property
    def strain_tensor(self):
        diag_range = self.diag_range
        off_diag_range = self.off_diag_range
        s11, s22, s33 = [random.choice(diag_range) for v in range(3)]
        s12, s13, s21, s23, s31, s32 = [random.choice(off_diag_range) for v in range(6)]
        sg_class = self.sg_class

        if sg_class in ["cubic", "orthorhombic", "monoclinic", "high-sym hexagonal/tetragonal"]:
            v1 = [s11, 0, 0]
        elif sg_class == "low-sym hexagonal/tetragonal":
            v1 = [s11, s12, 0],
        elif sg_class == "triclinic":
            v1 = [s11, s12, s13]

        if sg_class in ['cubic', 'high-sym hexagonal/tetragonal']:
            v2 = [0, s11, 0]
        elif sg_class == "orthorhombic":
            v2 = [0, s22, 0]
        elif sg_class == "monoclinic":
            v2 = [0, s22, s23]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v2 = [-s12, s22, 0]
        elif sg_class == "triclinic":
            v2 = [s21, s22, s23]

        if sg_class == 'cubic':
            v3 = [0, 0, s11]
        elif sg_class == 'high-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'monoclinic':
            v3 = [0, s23, s33]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'triclinic':
            v3 = [s31, s32, s33]

        return np.array([v1, v2, v3])

    @property
    def sg_class(self):
        sg = self.sg
        if sg in list(range(195, 231)):
            return "cubic"
        elif sg in list(range(16, 76)):
            return "orthorhombic"
        elif sg in list(range(3, 16)):
            return "monoclinic"
        elif sg in list(range(1, 3)):
            return "triclinic"
        elif sg in list(range(76, 195)):
            if sg in list(range(75, 83)) + list(range(143, 149)) + list(range(168, 175)):
                return 'low-sym hexagonal/tetragonal'
            else:
                return 'high-sym hexagonal/tetragonal'

    @property
    def stained_matrix(self):
        return np.matmul(self.matrix, self.strain_tensor)

    @property
    def strained_lattice(self):
        return Lattice(self.stained_matrix)

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
        return sigma ** 2

    @property
    def broadened_spectrum(self):
        pattern = self.calculator.get_pattern(self.structure, two_theta_range=(self.min_angle, self.max_angle))
        steps = np.linspace(self.min_angle, self.max_angle, self.resolution)

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = random.choice(self.possible_domains)
        norm_signal = np.zeros_like(steps)
        alpha = self.calc_std_dev(domain_size)
        gamma = 0.01
        # Replace each peak in the XRD pattern with a Voigt profile
        for i in range(len(pattern.x)):
            peak_position = pattern.x[i]
            peak_intensity = pattern.y[i]
            norm_signal += peak_intensity * V(steps - peak_position, alpha, gamma)

        # Normalize signal
        norm_signal = 100 * norm_signal / max(norm_signal)

        # Generate Poisson noise
        noise = np.random.poisson(norm_signal)
        # Add the noise to the signal
        noisy_signal = norm_signal + noise
        sga = SpacegroupAnalyzer(self.structure)
        space_group = sga.get_space_group_symbol()
        res = {"Intensity": noisy_signal, "angles": steps, "Space_Group": space_group}
        return res

    @property
    def strained_spectrum(self):
        struc = self.strained_struc
        pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
        angles, intensities = pattern.x, pattern.y

        steps = np.linspace(self.min_angle, self.max_angle, self.resolution)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang - steps))
            signals[i, idx] = intensities[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        norm_signal = np.zeros_like(steps)
        for i in range(len(pattern.x)):
            peak_position = pattern.x[i]
            peak_intensity = pattern.y[i]
            norm_signal += peak_intensity * V(steps - peak_position, 0.05, 0.01)

        # Normalize signal
        norm_signal = 100 * norm_signal / max(norm_signal)

        noise = np.random.poisson(norm_signal)
        noisy_signal = norm_signal + noise

        sga = SpacegroupAnalyzer(struc)
        space_group = sga.get_space_group_symbol()
        res = {"Intensity": noisy_signal, "angles": steps, "Space_Group": space_group}

        return res
