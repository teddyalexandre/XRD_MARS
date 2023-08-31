# SOLEIL MARS Beamline (Summer 2023 internship) : Phase prediction - Space group identification using XRD patterns and Deep Learning methods

SOLEIL project to perform phase identification in multiphase compounds at long term, using synthetic XRD powder patterns.

We employ deep learning techniques to highlight such properties, as well as space group identification, using a pre-trained convolutional neural network (CNN).

The project is split in three folders :

- data : Contains all the files (datasets) and scripts to manage/clean/preprocess the data before feeding it to the model.
- models : Contains the scripts that will encapsulate the models employed (Convolutional neural network / Variational auto-encoder) to predict space groups / phases from the XRD patterns
- tests : Contains the scripts that perform tests and experiments with data (Notebooks), to ensure everything works fine at every step of the project.

There is also a file 'requirements.txt', with all the python librairies required to run the project. Before anything, one must type in a command line, at the root of the project :

```bash
pip install -r requirements.txt
```

Every other library needed can be installed using pip in the terminal (one can add it in the requirements file later on). Any code editor can be used (I personnally worked on VSCode, but any Python editor will do the job).

The tasks will require to make computations on a GPU, so a virtual machine is required (re-grades02).

1/ The program begins by getting all of the Materials Project CIF files we can obtain, with their available API. So first, run the Python script (manually or in command line) _fetch_cif.py_ (from the _data_ folder) to regroup the 118399 existing CIF files from Materials Project into a folder _cif_files_.

2/ Then, from this directory, run the Python script _preprocess.py_ to generate the dataset (in a Parquet format), given the CIF files. Generating the whole dataset takes about 6 hours (with 64 GPU cores from the virtual machine), so this might take a bit of time.

3/ Finally, we feed the data in batches to a learning CNN so that it predicts space groups from chemical species, or their crystal systems. One can appreciate the curves from the Cross-entropy loss and the accuracy over epochs. I also provided a confusion matrix to perform better evaluation. That sums up my work for the 10 weeks here at SOLEIL.

This work is deployed in SOLEIL's GitLab platform so that it can be continued later and improved to perform phase identification in a mixture of powder crystals. But before considering this, a few things should be fixed :

- Perform better data augmentation (peak broadening, add noise to the model, possibly find other data...)
- Propose a model with a variational auto-encoder to compare with the CNN and possibly use for phase identification.
- Compare the CNN's performance with synthetic data and with experimental data (from McXtrace).
- At a longer term, provide a web interface (drag and drop) that takes a signal as input and returns a prediction of the space group / crystal system or the phase in mixture models.


_Teddy ALEXANDRE (Summer 2023 intern) and Anass BELLACHEHAB (Scientific staff)_