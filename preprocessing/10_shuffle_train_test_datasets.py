from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import json
import nibabel as nib
from datetime import datetime
import argparse
from pathlib import Path


def main():
    """ K fold cross validation split of the patient data used for training.
        Configurable inputs are: num of folds and train/test splitting ratio.
    """
    parser = argparse.ArgumentParser(
        description=
        'Creates json file with the train and valid datasets for k folds')
    parser.add_argument("--input",
                        help="Input directory (data_for_training).",
                        type=str,
                        default=os.getcwd())
    parser.add_argument("-k",
                        "--kfold",
                        help="Number of folds. Default 6",
                        type=int,
                        default=6)
    parser.add_argument("-r",
                        "--ratio",
                        help="Test data ratio. Default 0.2",
                        type=float,
                        default=0.2)
    parser.add_argument("-e",
                        "--extension",
                        help="Extension of data file",
                        type=str,
                        default=".nii.gz")

    args = parser.parse_args()
    data_path = Path(args.input)
    kfold = args.kfold
    test_size = args.ratio
    data_extension = args.extension
    project_id = data_path.parent.name

    if test_size:
        json_file = data_path.joinpath(
            f'{project_id}_train_test_split_{kfold}_fold.json')
    else:
        json_file = data_path.joinpath(
            f'{project_id}_train_val_set_independent_test.json')

    if not json_file.exists():
        print(f"Shuffling the data with the following inputs:")
        print("Data folder:", data_path)
        print("Number of folds:", kfold)
        print("Split ratio:", test_size)

        summary = {}
        patients = [d.name for d in data_path.iterdir() if d.is_dir()]
        if test_size:
            for i in range(kfold):
                summary[f'train_{i}'], summary[f'test_{i}'] = train_test_split(
                    patients, test_size=test_size)

        else:
            summary['train_0'], summary['test_0'] = shuffle(patients), []

        # save the data size as well !!! this is assuming that all data has the same shape
        rand_patient = data_path.joinpath(patients[0])
        rand_dat_file = [
            f for f in rand_patient.iterdir() if data_extension in f.name
        ]
        rand_dat = nib.load(rand_dat_file[0]).get_fdata()
        summary['data_shape'] = rand_dat.shape

        # save the date
        summary['creation_timestamp'] = datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S")

        # save to json file
        with open(json_file, 'w') as outfile:
            json.dump(summary, outfile, indent=1)

        print("Saved file", json_file)
        print("Job complete. ")

    else:
        with open(json_file, 'r') as outfile:
            splt = json.load(outfile)
        print(
            f"Train/test data already shuffled on {splt['creation_timestamp']} in: \n{json_file}"
        )


if __name__ == "__main__":
    main()
