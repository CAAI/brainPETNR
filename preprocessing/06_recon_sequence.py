import os
import re
import sys
import subprocess
import shutil
import time
import multiprocessing as mp
# from tqdm import tqdm
# import json
import argparse
from pathlib import Path
import pandas as pd
import yaml
from itertools import repeat


def edit_params_file(base_file, temp_file, formatted_dur):
    # format the duration first

    with open(base_file, "r") as f:
        lines = f.readlines()

    newp = []
    for line in lines:
        if 'LMFrames' in line and '#' not in line:
            line = re.sub(":= (.*)", f':= {formatted_dur}', line)
            print(line)
        newp.append(line)

    new_params = ''.join(newp)

    with open(temp_file, "w") as f:
        f.write(new_params)


def e7_sequence(input_dir, output_dir, param_file, recon_type):

    # setting up the directory names
    data_dir_converted = input_dir.parent.joinpath(
        f"{input_dir.name}-Converted")
    # check if Converted folder already exist -> recon was started but not finished. Start clean
    if data_dir_converted.exists():
        shutil.rmtree(data_dir_converted, ignore_errors=True)

    if output_dir.exists():
        print(output_dir, "already reconstructed!")
        return

    # running JSRecon12.js
    subprocess.run([
        "cscript", "C:\\JSRecon12\\JSRecon12.js",
        str(input_dir),
        str(param_file)
    ])

    # Moving to sub-directory
    LM00_folder = data_dir_converted.joinpath(f"{input_dir.name}-LM-00")
    os.chdir(LM00_folder)

    # Running Histogramming.bat
    subprocess.run([f"Run-00-{input_dir.name}-LM-00-Histogramming.bat"])

    # Running Makeumap.bat
    subprocess.run([f"Run-01-{input_dir.name}-LM-00-Makeumap.bat"])

    # running LM-00-OP.bat
    subprocess.run([f"Run-04-{input_dir.name}-LM-00-{recon_type}.bat"])
    time.sleep(1)

    # finding the HEADER file
    hdr_file_naming = f"-LM-00-{recon_type}_000"
    hdr_file = f"{input_dir.name}{hdr_file_naming}.v.hdr"
    if not LM00_folder.joinpath(hdr_file).exists():
        hdr_file_naming += "_000"
        hdr_file = f"{input_dir.name}{hdr_file_naming}.v.hdr"

    # running IF2Dicom.js
    subprocess.run([
        "cscript", "C:\\JSRecon12\\IF2Dicom.js", hdr_file,
        f"Run-05-{input_dir.name}-LM-00-IF2Dicom.txt"
    ])

    # Copying the DICOM folder over to base directory
    dicom_folder = LM00_folder.joinpath(
        f"{input_dir.name}{hdr_file_naming}.v-DICOM")
    shutil.move(dicom_folder, output_dir)
    print("Moving:", dicom_folder.name, '    to:', output_dir)

    # copy the original CT data
    ct_dicom = output_dir.parent.joinpath("CT")
    if not ct_dicom.exists():
        shutil.copytree(input_dir.joinpath("umap"), ct_dicom)

    # go back and delete converted data
    os.chdir("..\\..")
    time.sleep(2)
    if data_dir_converted.exists():
        shutil.rmtree(data_dir_converted, ignore_errors=True)
    print('Deleting:', data_dir_converted)


def process_wrapper(patient_folder, configs, df, out_dir):

    mode = configs['activity']
    acq_time = configs['acq_time']
    delay_time = configs['start_delay_for_recon']
    percent = configs['percent']
    blurring = configs['blurring']
    recon_type = configs['recon_type']
    project_id = configs['project']  # also base folder name
    # DRIVE letter on WINDOWS
    drive = Path(patient_folder.drive + '/')
    e7_params_file_path = drive.joinpath(configs['e7_params_file_path'])
    # get the parameter file from arg or automatically
    e7_params = e7_params_file_path.joinpath(
        f"{project_id}_jsrecon_params_{blurring}mm.txt")
    print(e7_params)
    if not e7_params.exists():
        raise FileNotFoundError(
            """ Could not find params file for JSRecon to run. Check config file."""
        )

    # create temp folder to store individual params file
    temp_folder = e7_params_file_path.joinpath("temp")
    temp_folder.mkdir(parents=True, exist_ok=True)

    patient_id = patient_folder.name
    # else edit params file and do reconstruction
    delay_for_recon = df.loc[patient_id, 'delay for recon']

    data_folder = patient_folder.joinpath('PET')
    output_folder = f"PET_{blurring}mm"
    if mode == 'dose reduction':
        data_folder = patient_folder.joinpath(f"PET_LD{percent}pct")
        output_folder = f"PET_{blurring}mm_LD{percent}pct"

    ####### special for time reduction recons ###########
    # pib 540:120 (9min:2min)    570:60  (1min)
    # pib 600:300 (10min:5min)
    # pe2i 270:60 (4min30sec:1min)     285:30 (0.5 min)
    elif mode == 'time reduction':
        if delay_for_recon != 'all':
            delay_time = int(delay_for_recon.split(':')[0]) + delay_time
        delay_for_recon = f'{delay_time}:{acq_time}'

        output_folder = f"PET_{blurring}mm_LD{acq_time}sec{delay_time}delay"
    ###########################################

    output_folder = out_dir.joinpath(patient_id).joinpath(output_folder)
    # create temporary params file
    temp_params_file = temp_folder.joinpath(
        f"{patient_id}_{output_folder.name}_{e7_params.stem}")
    edit_params_file(e7_params, temp_params_file, delay_for_recon)

    try:
        e7_sequence(data_folder, output_folder, temp_params_file, recon_type)
    except Exception as e:
        print(e)

    # remove temp file
    temp_params_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description=
        'Runs the JSRecon12 reconstruction sequence on all folders in a given data directory.'
    )
    parser.add_argument("-i",
                        "--input",
                        help="Input directory. Default: cwd.",
                        type=str,
                        default=os.getcwd())
    parser.add_argument("-o",
                        "--output",
                        help="Output directory. Default: 'data_recon'.",
                        type=str,
                        default='data_dicom')
    parser.add_argument("-c",
                        "--config",
                        help="Config file for recon in .yaml format",
                        type=str,
                        default=Path(
                            os.getcwd()).parent.joinpath('recon_config.yaml'))
    parser.add_argument("-mp",
                        "--multiprocess",
                        help="How many processes to use in parallel.",
                        type=int,
                        default=0)
    args = parser.parse_args()

    data_dir = Path(args.input)

    with open(args.config) as cf:
        config = yaml.safe_load(cf)

    recon_data_dir = data_dir.parent.joinpath(args.output)
    recon_data_dir.mkdir(parents=True, exist_ok=True)

    lower_thresh = config['dose_threshold']
    scanner_name = config['scanner']

    pt_file_info_gen = data_dir.glob(f'*anonymized_patient_info.csv')
    pt_file_info = [
        f for f in pt_file_info_gen if f.parent.name in data_dir.name
    ]
    # lm_info = next(lm_info_gen)
    pt_file_info = pt_file_info[0]
    df = pd.read_csv(pt_file_info, index_col=0)

    print(
        "\n################ RECONSTRUCTION STARTING WITH CONFIG ##############"
    )
    for k, v in config.items():
        print(k.ljust(40), v)
    print(
        "#####################################################################\n"
    )

    # filter which patients to reconstruct - based on config file
    df_filter = df.copy()
    # filtering based on scanner type
    if scanner_name != 'ALL':
        df_filter = df_filter[df_filter['ScannerName'] == scanner_name]
    # filtering based on dose value
    if lower_thresh:
        df_filter = df_filter[df_filter['tracer dose (MBq)'] > lower_thresh]
    # filtering based on image duration
    duration_mode = df['image duration (sec)'].mode()[0]
    df_filter = df_filter[
        df_filter['image duration (sec)'] > int(0.9 * duration_mode)]

    good_patients = df_filter.index
    patient_folders = [
        d for d in data_dir.iterdir() if d.is_dir() if d.name in good_patients
    ]

    # parallelize patient reconstruction
    if args.multiprocess:
        with mp.Pool(args.multiprocess) as p:
            # need to use repeat + zip as the patient_folders is a list but the
            # other args are unique values.
            p.starmap(
                process_wrapper,
                zip(patient_folders, repeat(config), repeat(df),
                    repeat(recon_data_dir)))

    else:
        for patient in patient_folders:
            process_wrapper(patient, config, df, recon_data_dir)

    print("RECONSTRUCTION DONE")
    print("JOB COMPLETE!\n")


if __name__ == '__main__':
    main()
