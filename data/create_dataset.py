from mp_api.client import MPRester
from utils import process_material
from joblib import Parallel, delayed
import pandas as pd

### In this script, we use the MPRester API from Materials Project to access information from the database in a structured way

# Both API Keys to use the API (change depending on who uses the program)
api_anass = "bo70Q5XVKyZdImV77bFXHO2cDKdvVQ6F"
api_teddy = "wV2nzQ5zNVhlugrbV6CSDbGYsEc2YmFU"

# Initialize the MPRester
with MPRester(api_key=api_teddy) as mpr:

    # Initialize an empty list to store the data
    data = []

    # Fetch the list of material IDs (returns a MPDataDoc object with data inside)
    material_docs = mpr.summary.search(material_ids=["mp-2680"], fields=["material_id", "crystal_system", "spacegroup_number"])

    # Use joblib to process the materials in parallel (all CPUs)
    data = Parallel(n_jobs=-1)(delayed(process_material)(material_doc.material_id, api_teddy, 0.1, 0.1, "MoKa") for material_doc in material_docs)

    # Save the dataset to a Parquet or CSV file
    df = pd.DataFrame(data)
    df.to_parquet("pow_xrd.parquet")
    #df.to_csv("pow_xrd.csv")
    print(df)