"""
More or less useless file, computes the Parquet file from COD database (not good though, since there is an
obvious heterogeneity between the CIF files). One might have to stick with Materials Project, but I let the script
if one still wants to use it."""

from utils import calculate_xrd_from_cif
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

path_to_cod_cifs = "/home/experiences/grades/alexandret/ruche/share-temp/XRD_MARS_datasets/val_data/"

files_cif_list = np.array(pd.read_csv("./datasets/cod_cif_files_validation.txt", sep=" ", dtype=str))
files_cif_list = files_cif_list[:100]
files_cif = ["".join(path_to_cod_cifs + file) for file in files_cif_list]


# Use joblib to process the materials in parallel (all 64 CPUs)
data = Parallel(n_jobs=64)(delayed(calculate_xrd_from_cif)(f, 1.0, 0.721) for f in files_cif)

# Save the dataset to a Parquet file
pandas_df = pd.DataFrame(data)

# Write dataframe in a Parquet file
pandas_df.to_parquet(path_to_cod_cifs + '../pow_xrd_val.parquet')
