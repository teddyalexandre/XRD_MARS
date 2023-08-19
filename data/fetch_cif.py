from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
import glob
import os
import pandas as pd

api_anass = "bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F"
api_teddy = "wV2nzQ5zNVhlugrbV6CSDbGYsEc2YmFU"

# Create a directory containing all the CIF files if it doesn't exist
path = "./data/cif_files"
if not os.path.exists(path):
    os.makedirs(path)

# Removes the previous files in the folder cif_files
files_cif = glob.glob("./data/cif_files/*")
for f in files_cif:
    os.remove(f)

# With the API from Materials Project, we obtain a cif file for every material
with MPRester(api_key=api_anass) as mpr:
    materials = mpr.get_structures(['*', '*-*', '*-*-*', '*-*-*-*', '*-*-*-*-*', '*-*-*-*-*-*',
                                    '*-*-*-*-*-*-*', '*-*-*-*-*-*-*-*', '*-*-*-*-*-*-*-*-*'])

    # For each material, build a CIF symmetrized file
    for i in range(len(materials)):
        material = materials[i]
        #material_id = mpr.get_material_ids(material.formula)
        #print(f"Material ID : {material_id}, Material composition : {material.composition}, Space group : {material.get_space_group_info()}")
        CifWriter(struct=material, symprec=None).write_file('./data/cif_files/{}.cif'.format(material.formula))