from mp_api.client import MPRester
from utils import process_material
from joblib import Parallel, delayed
import pandas as pd
from pymatgen.io.cif import CifWriter
import glob
import os

api_anass = "bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F"
api_teddy = "wV2nzQ5zNVhlugrbV6CSDbGYsEc2YmFU"

# Create a directory containing all the CIF files if it doesn't exist
path = "./data/cif_files"
if not os.path.exists(path):
    os.makedirs(path)

with MPRester(api_key=api_teddy) as mpr:

    # Fetch the list of materials given their formulas
    suffix1 = "-*"
    suffix2 = "*"
    list_materials = ["La","B","O","C","H","P","S","U","Np","Pu"]
    materials = mpr.get_structures(list_materials)

    # Removes the previous files in the folder cif_files
    files_cif = glob.glob("./data/cif_files/*")
    for f in files_cif:
        os.remove(f)
    
    # For each material, build a CIF symmetrized file
    for i in range(len(materials)):
        material = materials[i]
        #material_id = mpr.get_material_ids(material.formula)
        #print(f"Material ID : {material_id}, Material composition : {material.composition}, Space group : {material.get_space_group_info()}")
        CifWriter(struct=material, symprec=None).write_file('./data/cif_files/{}.cif'.format(material.formula))
