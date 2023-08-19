# SOLEIL MARS Beamline : Phase prediction - Space group identification using XRD patterns and Deep Learning methods

SOLEIL project to perform phase identification in multiphase compounds, using synthetic XRD powder patterns.

We employ deep learning techniques to highlight such properties, as well as space group identification, using a pre-trained CNN.

The project is split in three folders :

- data : Contains all the files (datasets) and scripts to manage/clean/preprocess the data
- models : Contains the scripts that will encapsulate the models employed (Convolutional neural network / Variational auto-encoder) to predict space groups / phases from the XRD patterns
- tests : Contains the scripts that perform tests, to ensure everything works fine at every step of the project.

There is also a file 'requirements.txt', with all the python librairies required to run the project. Before anything, one must type in a command line, at the root of the project :

```bash
pip install -r requirements.txt
```

The program begins by getting all of the Materials Project CIF files we can obtain, with their API. So first, run the Python script (manually or in command line) _fetch_cif.py_ (from the _data_ folder) to regroup the 118399 existing cif files from Materials Project into a folder _cif_files_.

Then, from this directory, run the Python script _preprocess.py_ to generate the dataset (in a Parquet format), given the CIF files. Generating the whole dataset takes about 6 hours (with 64 GPU cores from the virtual machine), so this might take a bit of time.

Finally, we feed the data in batches to a learning CNN so that it predicts space groups from chemical species, or their crystal systems. Following this work, we will try to predict phases in a mixture model (not done yet). However, the CNN performs quite badly (poor accuracy), so different optimization techniques will be employed.
