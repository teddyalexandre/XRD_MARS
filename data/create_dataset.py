from mp_api.client import MPRester
from utils import process_material
from joblib import Parallel, delayed
import pandas as pd

# Initialize the MPRester
with MPRester(api_key="bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F") as mpr:

    # Initialize an empty list to store the data
    data = []

    # Fetch the list of material IDs
    material_docs = mpr.summary.search(num_chunks=10)

    # Use joblib to process the materials in parallel
    data = Parallel(n_jobs=-1)(delayed(process_material)(material_doc.material_id, "bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F", 0.1, 0.1, "MoKa") for material_doc in material_docs)

    # Save the dataset to a file
    df = pd.DataFrame(data)
    df.to_parquet("pow_xrd.parquet")
