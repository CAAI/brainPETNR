import os
import re
import sys
import subprocess
import shutil
import time
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import pandas as pd
import yaml
from multiprocessing import Pool


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


def e7_sequence(input_dir, output_dir, param_file, recon_type, is_test):

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

    # check inside the header file for pixel spacing in test mode
    if is_test:
        with open(hdr_file) as hf:
            hf_content = hf.readlines()
        hf_content = [
            l for l in hf_content if 'scale factor (mm/pixel) [1]' in l
        ]
        print(*hf_content)
        print(
            'Remember to check zoom value in param file to calculate accordingly!'
        )
        sys.exit(-1)

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
    # rmtree(data_dir, ignore_errors=True)
    # print('Deleting:', data_dir)
    # if not is_test:
    #     rmtree(data_dir_converted, ignore_errors=True)
    print('Deleting:', data_dir_converted)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Runs the JSRecon12 reconstruction sequence on all folders in a given data directory.'
    )
    parser.add_argument(
        "-i",
        "--input",
        help=
        "Directory containing patients data. Will use current working directory if nothing passed",
        type=str,
        default=os.getcwd())
    parser.add_argument(
        "-o",
        "--output",
        help=
        "Output directory for recon data. Default is 'data_recon' if nothing passed.",
        type=str,
        default='data_dicom')
    parser.add_argument("-c",
                        "--config",
                        help="Config file for recon",
                        type=str,
                        default=Path(
                            os.getcwd()).parent.joinpath('recon_config.yaml'))
    # parser.add_argument(
    #     "-p",
    #     "--params",
    #     help=
    #     "Path to params file. Detects params file containing project name if none given",
    #     type=str,
    #     default='')
    parser.add_argument("-t",
                        "--test",
                        help="Test run for 1 patient",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    data_dir = Path(args.input)

    with open(args.config) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    recon_data_dir = data_dir.parent.joinpath(args.output)
    recon_data_dir.mkdir(parents=True, exist_ok=True)
    base_folder = Path(data_dir.drive + '/')
    project_id = config['project']
    recon_type = config['recon_type']
    lower_thresh = config['dose_threshold']
    scanner_name = config['scanner']
    blurring = config['blurring']
    e7_params_file_path = Path(config['e7_params_file_path'])
    mode = config['activity']
    acq_time = config['acq_time']
    delay_time = config['start_delay_for_recon']
    percent = config['percent']

    is_test = args.test
    if is_test:
        print('This is a test run on 1 patient..')

    lm_info_gen = data_dir.glob(f'*anonymized_patient_info.csv')
    lm_info = [f for f in lm_info_gen if f.parent.name in data_dir.name]
    # lm_info = next(lm_info_gen)
    lm_info = lm_info[0]
    df = pd.read_csv(lm_info, index_col=0)
    # make copy of df into output dir (recon)
    # df.to_csv(recon_data_dir.joinpath(lm_info.name))

    # get the parameter file from arg or automatically
    e7_params = e7_params_file_path.joinpath(
        f"{project_id}_jsrecon_params_{blurring}mm.txt")

    if not e7_params.exists():
        raise FileNotFoundError(
            """ Could not find params file for JSRecon to run. Check config file."""
        )

    print(f"RECONSTRUCTION STARTING with recon type {recon_type}")

    # make filters on df and use index for for-loop
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
    if is_test:
        patient_folders = patient_folders[:1]

    # create temp folder to store individual params file
    temp_folder = e7_params.parent.joinpath("temp")
    temp_folder.mkdir(parents=True, exist_ok=True)

    fails = {}
    for pf in tqdm(patient_folders):

        patient_id = pf.name
        # else edit params file and do reconstruction
        delay_for_recon = df.loc[patient_id, 'delay for recon']

        data_folder = pf.joinpath('PET')
        output_folder = f"PET_{blurring}mm"
        if mode == 'dose reduction':
            data_folder = pf.joinpath(f"PET_LD{percent}pct")
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
        output_folder = recon_data_dir.joinpath(patient_id).joinpath(
            output_folder)
        # create temporary params file
        temp_params_file = temp_folder.joinpath(
            f"{patient_id}_{output_folder.name}_{e7_params.stem}")
        edit_params_file(e7_params, temp_params_file, delay_for_recon)

        try:
            e7_sequence(data_folder, output_folder, temp_params_file,
                        recon_type, is_test)
        except Exception as e:
            print(e)
            fails[patient_id] = str(e)

        # remove temp file
        temp_params_file.unlink()

    if fails:
        print(f'\n\nFailed to reconstruct {len(fails)} datasets')
        json_file = data_dir.joinpath(
            f'{project_id}_failed_reconstructions_with_code.json')
        with open(json_file, 'w') as outfile:
            json.dump(fails, outfile, indent=1)

    print("RECONSTRUCTION DONE")
    print("JOB COMPLETE!\n")
    """ in principle we can multiprocess this script, 
        however there's a Windows related error on the recon machines
        
        Would simply need to put what's inside the for loop in a function:
        
        for pf in tqdm(patient_folders): --> def process_wrapper(pf):
            ...
        
        with Pool(20) as p:
            p.map(process_wrapper, patient_folders)
        
        """


if __name__ == '__main__':
    main()
