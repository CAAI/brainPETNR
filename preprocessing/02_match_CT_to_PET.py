import json
import os
from pydicom import dcmread
from pathlib import Path
import pandas as pd
from shutil import move, rmtree
import time
import argparse

################################
parser = argparse.ArgumentParser(description='Generate anonym ID for each patient folder and extract study UID')
parser.add_argument("-pet", "--pet-data", 
                    help="Directory containing patients raw data. Will use current working directory if nothing passed", 
                    type=str, default=os.getcwd())
parser.add_argument("-ct", "--ct-data", 
                    help="Directory containing patients CT data. Will use current working directory if nothing passed", 
                    type=str, default=os.getcwd())

args = parser.parse_args()

data_path = Path(args.pet_data)
ct_path = Path(args.ct_data)
project_id = data_path.parent.name
patient_info_file = data_path.joinpath(f'{project_id}_unanonymized_patient_info.csv')
patient_info = pd.read_csv(patient_info_file, index_col=0)

for dirpath, _, ct_files in os.walk(ct_path):
    if ct_files:
        if ct_files[0].startswith('.'):
            continue
        dcm_folder = Path(dirpath)
        # find the UID from the CT header
        ctf = dcm_folder.joinpath(ct_files[0])
        with dcmread(ctf) as ds:
            study_uid = ds.StudyInstanceUID
        
        print('move:', dirpath)
        
        # match UID to patient folder
        print(patient_info[patient_info['StudyInstanceUID'] == study_uid])
        match = patient_info['StudyInstanceUID'] == study_uid
        patient_idx = patient_info[match].index
        if list(patient_idx):
            # there's UID match
            move_to = data_path.joinpath(patient_idx[0]).joinpath('umap')
        else:
            move_to = None
                
        # rename files with DCM extension
        for f in dcm_folder.iterdir():
            if not f.name.endswith('.dcm'):
                f.rename(dcm_folder.joinpath(f.name + ".dcm"))
        
        # move CT folder to the correct patient
        if move_to:
            move(dirpath, move_to)
            print('to:', move_to)
            print('Removing: ', dcm_folder.parent)
            rmtree(dcm_folder.parent)
                    
                    
        else:
            print("StudyInstanceUID in CT not matching any PET data for ", dcm_folder.parent.name)