from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import json
import nibabel as nib
from datetime import datetime as dt
import argparse
from pathlib import Path
# -- user inputs
#k = 6
#test_size = .2
#data_path = '/homes/raphael/Projects/Lowdose_PiB/data_anonymized/'
#data_extension = '.nii'


def main(data_path, k=6, test_size=.2, data_extension='.nii.gz'):
    project_id = data_path.parent.name
    if test_size:
        json_file = data_path.joinpath(
            f'{project_id}_train_test_split_{k}_fold.json')
    else:
        json_file = data_path.joinpath(
            f'{project_id}_train_val_set_independent_test.json')
        
    if not json_file.exists():
        print(f"Shuffling the data with the following inputs:")
        print("Data folder:", data_path)
        print("Number of folds:", k)
        print("Split ratio:", test_size)

        summary = {}
        patients = [d.name for d in data_path.iterdir() if d.is_dir()]
        if test_size:
            for i in range(k):
                summary[f'train_{i}'], summary[f'test_{i}'] = train_test_split(
                    patients, test_size=test_size)
                
        else:
            summary['train_0'], summary['test_0'] = shuffle(patients), []

        # save the data size as well !!! this is assuming that all data has the same shape
        rand_patient = data_path.joinpath(patients[0])
        rand_dat_file = [f for f in rand_patient.iterdir()
                         if data_extension in f.name]
        rand_dat = nib.load(rand_dat_file[0]).get_fdata()
        # if len(rand_dat.shape) < 4:
        #     rand_dat = rand_dat.reshape((rand_dat.shape, 1))
        summary['data_shape'] = rand_dat.shape

        # save the date
        summary['creation_timestamp'] = dt.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S")

        # save the json file
        with open(json_file, 'w') as outfile:
            json.dump(summary, outfile, indent=1)

        print("Saved file", json_file)
        print("Job complete. ")

    else:
        with open(json_file, 'r') as outfile:
            splt = json.load(outfile)
        print(
            f"Train/test data already shuffled on {splt['creation_timestamp']} in: \n{json_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Creates json file with the train and valid datasets for k folds')
    parser.add_argument("--input",
                        help="Specify the input patient data path (where to shuffle data from).",
                        type=str,
                        default=os.getcwd())
    parser.add_argument(
        "-v", "--verbose", help="Print out verbose information", action="store_true")
    parser.add_argument(
        "-k", "--kfold", help="Number of folds. Default 6", type=int, default=6)
    parser.add_argument(
        "-r", "--ratio", help="Test data ratio. Default 0.2", type=float, default=0.2)
    parser.add_argument(
        "-e", "--extension", help="Extension of data file", type=str, default=".nii.gz")
    args = parser.parse_args()

    main(Path(args.input), args.kfold, args.ratio, args.extension)
