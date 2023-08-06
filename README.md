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