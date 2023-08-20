"""
This script creates the dataset as a Parquet file, given the CIF files collected with 'fetch_cif.py'
"""

#import argparse
#import logging
from utils import calculate_xrd_from_cif
from joblib import Parallel, delayed
import pandas as pd
import glob
import os
import time

start = time.time()
### In this script, we use the MPRester API from Materials Project to access information from the database in a structured way

files_cif = glob.glob("./data/cif_files/*.cif")

# Removes the previous parquet file if it exists
try:
    os.remove("./data/pow_xrd.parquet")
except Exception as e:
    pass

# Use joblib to process the materials in parallel (all 64 CPUs)
data = Parallel(n_jobs=64)(delayed(calculate_xrd_from_cif)(f, 0.05, 0.01, 0.721) for f in files_cif)

# Save the dataset to a Parquet file
pandas_df = pd.DataFrame(data)

# Write dataframe in a Parquet file
pandas_df.to_parquet('./data/pow_xrd.parquet')

end = time.time()
# We measure how long the creation of the Parquet file took
print(f"The code took : {end - start} seconds to execute, i.e. {(end - start) / 60} minutes, i.e. {(end - start) / 3600} hours.")