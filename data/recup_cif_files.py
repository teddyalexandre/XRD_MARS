from pathlib import Path
import os
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from time import time
import urllib



def download_cif(txt_path, path_output_dir):
    # reading text file
    with open(txt_path) as f:
        cif_urls = f.read().splitlines()

    # make the output directory
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    
    # download CIFs
    for idx, cif_url in enumerate(cif_urls):
        print(str(idx)+'/'+str(len(cif_urls)-1))
        urllib.request.urlretrieve(
            cif_url,
            path_output_dir+'/'+cif_url.split('/')[-1],
        )

if __name__ == '__main__':
    txt_path = './COD-selection.txt'
    path_output_dir = './cif_files'
    download_cif(txt_path, path_output_dir)

    start = time()

    list_of_cif = []
    for file in Path(path_output_dir).glob('*.cif'):
        dico = MMCIF2Dict(file)
        temp = pd.DataFrame.from_dict(dico, orient='index')
        temp = temp.transpose()
        temp.insert(0, 'Filename', Path(file).stem) #to get the .CIF filename
        list_of_cif.append(temp)
    df = pd.concat(list_of_cif)

    end = time()
    print(df["_space_group_symop_id"])
    df.to_csv("./dataset_from_cif.csv")