from parsers.dicom import DicomParser
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def main():
    """ Parses all info contained in PET and CT dicom files (after reconstruction).
        Derives rescaling value of PET data, 
        which is usefull in case of dose reduction or scan time reduction.
        Exports info for all patients to one csv file.
    """
    parser = argparse.ArgumentParser(
        description=
        """ Parses all info contained in PET and CT dicom files (after reconstruction).
            Exports info for all patients to one csv file.""")
    parser.add_argument(
        "-i",
        "--input",
        help="Input directory (reconstructed data). Default: cwd.",
        type=str,
        default=os.getcwd())

    args = parser.parse_args()
    data_dir = Path(args.input)
    project_id = data_dir.parent.name

    # get all patients ids
    patient_folders = [d for d in data_dir.iterdir() if d.is_dir()]

    infos = {}
    pet_files = set()
    for patient_folder in tqdm(patient_folders):
        patient = patient_folder.name
        pet_folder = patient_folder.joinpath('PET_5mm')
        if not pet_folder.exists():
            continue
        dcm_files = [
            f for f in pet_folder.iterdir()
            if '.dcm' in f.name or '.ima' in f.name
        ]

        if dcm_files:
            dcm_parser = DicomParser(dcm_files[0])
            dcm_parser.get_header_info()
            infos[patient] = dcm_parser.info

        # check type of PET and lowdose extensions
        subfolders = [d for d in patient_folder.iterdir() if d.is_dir()]
        for sf in subfolders:
            pet_files.add(sf.name)

    print('Cleaning up data and saving to', data_dir)
    # make a DataFrame and clean up
    df = pd.DataFrame(infos).transpose()
    df['Activity (MBq)'] = df['RadionuclideTotalDose'].apply(
        lambda x: int(x) / 1e6)
    # export dataframe as csv
    df.to_csv(data_dir.joinpath(f'{project_id}_anonymized_patient_info.csv'))

    # deriving scaling factors for each lowdose instances
    radionuclide_halflife = float(df['RadionuclideHalfLife'].mode()[0])
    base_frame_duration = float(df['ActualFrameDuration'].mode()[0]) / 1000

    scaling = {'CT': 1.0, 'MR': 1.0}
    activity = {'CT': 1.0, 'MR': 1.0}
    for name in pet_files:
        if name in scaling:
            continue
        if 'segmentation' in name:
            continue
        if 'sec' in name:
            # name template is PET_LD{frame_duration}sec{delay}delay
            delay = int(name.split('sec')[-1].replace('delay', ''))
            frame_duration = int(name.split('_LD')[-1].split('sec')[0])
            scaling[name] = 1.0 / (np.exp(
                -delay * np.log(2) / radionuclide_halflife))
            activity[name] = frame_duration / base_frame_duration
        elif 'pct' in name:
            # name template is PET_LD{chop_percent}pct
            pct = int(name.split("_LD")[-1].replace('pct', ''))
            scaling[name] = 100.0 / pct
            activity[name] = pct / 100.0
        else:
            scaling[name] = 1.0
            activity[name] = 1.0

    dfs = pd.DataFrame([scaling, activity]).transpose()
    dfs.columns = ['intensity_scaling', 'activity']
    dfs.to_csv(data_dir.joinpath(f'{project_id}_scaling_factors.csv'))


if __name__ == '__main__':
    main()