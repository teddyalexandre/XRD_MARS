from utils import calculate_xrd_from_cif
from joblib import Parallel, delayed
import pandas as pd
import glob
import pandas as pd
import os

### In this script, we use the MPRester API from Materials Project to access information from the database in a structured way

files_cif = glob.glob("./data/cif_files/*.cif")

# Removes the previous parquet file if it exists
try:
    os.remove("./data/pow_xrd.parquet")
except Exception as e:
    pass

# Use joblib to process the materials in parallel (all CPUs)

data = Parallel(n_jobs=-1)(delayed(calculate_xrd_from_cif)(f, 0.05, 0.01, 0.721) for f in files_cif)

# Save the dataset to a Parquet file
df = pd.DataFrame(data)
df.to_parquet("./data/pow_xrd.parquet")