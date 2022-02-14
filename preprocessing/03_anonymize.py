import os
from subprocess import run
from shutil import rmtree
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def anonymize(origin_folder, dest_folder, an_id):
    dest_folder.mkdir(parents=True, exist_ok=True)
    cmd_list = [
        r'C:\JSRecon12\anonymize.exe', '-i',
        str(origin_folder), '-o',
        str(dest_folder), '-n', an_id, '-p', '-u'
    ]
    run(cmd_list)

    step_failed = len(os.listdir(dest_folder)) != len(
        os.listdir(origin_folder))
    time.sleep(1)
    return step_failed


def main():
    parser = argparse.ArgumentParser(
        description=
        'Generate anonym ID for each patient folder and extract study UID')
    parser.add_argument(
        "-i",
        "--input",
        help=
        "Directory containing patients data. Will use current working directory if nothing passed",
        type=str,
        default=os.getcwd())
    parser.add_argument(
        "-p",
        "--project-id",
        help=
        "Project keyword or ID to prefix to all anonym IDs. Default is name of parent folder",
        type=str,
        default='')
    args = parser.parse_args()

    raw_data_dir = args.input
    anonym_data_dir = Path(raw_data_dir + '_anonymized')
    anonym_data_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir = Path(raw_data_dir)
    project_id = args.project_id
    if not project_id:
        project_id = raw_data_dir.parent.name

    json_file = raw_data_dir.joinpath(
        f"{project_id}_patient_to_anonym_id.json")
    with open(json_file, 'r') as f:
        patient_to_anonym_id = json.load(f)
    """ ANONYMIZATION """

    failed_anonymization = {}
    for patient_folder, anonym_id in tqdm(patient_to_anonym_id.items()):
        # print(f"Anonymization {counter+1}/{len(patient_to_anonym_id)}: {patient_folder} -> {anonym_id}")
        print(f"Anonymizing {patient_folder} -> {anonym_id}")
        patient_in = raw_data_dir.joinpath(patient_folder)
        if not patient_in.exists():
            continue

        if not 'umap' in os.listdir(patient_in):
            print('No CT for this patient. Skipping.')
            continue

        patient_out = anonym_data_dir.joinpath(anonym_id).joinpath('PET')
        if patient_out.exists():
            if len(os.listdir(patient_out)) == len(os.listdir(patient_in)):
                # patient already anonymized
                print('Patient already anonymized')
                continue

        # run anonymization
        anonymization_failed = anonymize(patient_in, patient_out, anonym_id)
        if anonymization_failed:
            print(
                f"Error anonymizing {patient_folder} -> {anonym_id}. Retrying in the end"
            )
            failed_anonymization[patient_folder] = anonym_id
            # delete incomplete data
            rmtree(patient_out, ignore_errors=True)

    failed_anonymization_final = {}
    # rerunning for failed anonymization
    for patient_folder, anonym_id in failed_anonymization.items():
        print(f"Retrying anonymization {patient_in.name} -> {anonym_id}")
        patient_in = raw_data_dir.joinpath(patient_folder)
        patient_out = anonym_data_dir.joinpath(anonym_id)
        anonymization_failed = anonymize(patient_in, patient_out, anonym_id)
        if anonymization_failed:
            failed_anonymization_final[patient_folder] = anonym_id

            print(
                f"Error anonymizing for the 2nd time: {patient_folder} -> {anonym_id}. Storing info"
            )
            with open(f'{project_id}_anonymization_log.txt', 'a') as lf:
                lf.write(
                    f"""Error anonymizing some files from {patient_folder} -> {anonym_id}
                            Num of files originally: {len(os.listdir(patient_in))}
                            Num of files anonymized: {len(os.listdir(patient_out))}\n  """
                )

            # remove incomplete folder
            rmtree(patient_out, ignore_errors=True)

    if failed_anonymization_final:
        json_file = anonym_data_dir.joinpath(
            f'{project_id}_failed_anonymizations.json')
        with open(json_file, 'w') as outfile:
            json.dump(failed_anonymization_final, outfile, indent=1)


if __name__ == '__main__':
    main()
