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

# Remove the formula csv file if it exists
if os.path.exists("./data/formula_names.csv"):
    os.remove("./data/formula_names.csv")

# Removes the previous files in the folder cif_files
files_cif = glob.glob("./data/cif_files/*")
for f in files_cif:
    os.remove(f)

with MPRester(api_key=api_teddy) as mpr:

    # Fetch the list of materials given their formulas (more than 100000 with these 25 chemical species)
    # Sample of a few materials
    list_materials = ["La","B","O","C","H",
                      "Ca","Bu","Cu","Zn","Si",
                      "U","Np","Pu","N","Na",
                      "Mg","Al","Cl","Fe","Co",
                      "Ni","Mn","Hg","Ac","Cr"]
    
    # We add a combination of these materials
    list_full_mat = [[x, x+"-*", x+"-*-*"] for x in list_materials]
    # Flatten the list
    list_full_mat = [val for sub_list in list_full_mat for val in sub_list]
    formula_names = []

    # We do subsamples of the list above, otherwise the API doesn't run smooth
    for k in range(0,len(list_full_mat),5):
        small_list = list_full_mat[k:k+5]
        materials = mpr.get_structures(small_list)
    
        # For each material, build a CIF symmetrized file
        for i in range(len(materials)):
            material = materials[i]
            #material_id = mpr.get_material_ids(material.formula)
            #print(f"Material ID : {material_id}, Material composition : {material.composition}, Space group : {material.get_space_group_info()}")
            CifWriter(struct=material, symprec=None).write_file('./data/cif_files/{}.cif'.format(material.formula))
            formula_names.append(material.formula)
    
    df = pd.DataFrame(formula_names, columns=["Formulas"])
    df.to_csv("./data/formula_names.csv")

