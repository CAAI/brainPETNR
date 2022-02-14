import sys

sys.path.append("..")
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from parsers.ptd import ListmodeFileParser
import argparse
import numpy as np
from datetime import datetime


# functions for working on the DATAFRAME
def format_date(d, t):
    return datetime.strptime(f"{d} {t}", "%Y:%m:%d %H:%M:%S")


def time_delay(col):
    """ Function to be used on pandas dataframe.
        Calculates the time delay between injection and scan starting time.

    Args:
        col (pd.DataFrame): pandas dataframe

    Returns:
        (pd.Series): one column to stitch to the dataframe
    """
    return pd.to_timedelta(col['study time'] -
                           col['tracer injection time']).seconds


def delay_for_recon(col, base_duration, avg_delay_at_mode):
    """ Function to be used on pandas dataframe.
        Calculates when to start the recon given 
        the acquisition time and the delay.

    Args:
        col (pd.DataFrame): pandas dataframe
        base_duration (int): duration for reconstruction.
        avg_delay_at_mode (int): average delay time accross all patients.

    Returns:
        (str): 'all' or delay:duration
    """
    if col['image duration (sec)'] <= base_duration:
        return 'all'
    else:
        # if delay for patient above mean then 0
        recon_delay = np.maximum(0,
                                 avg_delay_at_mode - col['time delay (sec)'])
        # if delay for recon is too long
        recon_delay = np.minimum(recon_delay,
                                 col['image duration (sec)'] - base_duration)
        return f'{int(recon_delay)}:{int(base_duration)}'


def main():
    """ Parsing listmode file to prepare for reconstruction.
        Specifically calculating the reconstruction duration and start offset.
    """

    # scanner maps
    scanner_type = {
        'RHPET6NAV': 'TRUEPOINT',
        'RHKFASIEPTR5': 'VISION',
        'RHPET5NAV': 'TRUEPOINT',
        'RHKFASIEPTR3': 'VISION',
        'RHPET4NAV': 'MCT',
        'RHKFASIEPRT3': 'VISION',
        'RHPET3NAV': 'MCT',
        'RHPET8NAV': 'MCT',
        'RHKFASIEPTR30': 'MCT'
    }

    ################################
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
        "Project keyword or ID to prefix the patient database. Default is parent folder name.",
        type=str,
        default='')
    args = parser.parse_args()

    anonym_data_dir = Path(args.input)
    project_id = args.project_id
    if not project_id:
        project_id = anonym_data_dir.parent.name

    print('Initiating data parsing for project', project_id)
    folders = [
        f for f in anonym_data_dir.iterdir()
        if f.is_dir() and 'lowdose' not in f.name
    ]

    infos = {}
    for p_folder in tqdm(folders):
        """ PARSE LISTMODE FILE FOR INFO"""
        parser1 = ListmodeFileParser(p_folder.joinpath('PET'))
        # parse listmode file into a list of lines
        parser1.read_tail(stopword='DICM')
        # parse lines containing := into dictionnary
        parser1.get_primary_info(include='=', exclude='!')
        # parse the rest of the info (where values aren't labeled (no :=))
        parser1.get_secondary_info()
        # turn values to int if possible
        parser1.clean_info()

        infos[p_folder.name] = parser1.info

    print('Cleaning up data and saving to', anonym_data_dir)
    # make a DataFrame and clean up
    df = pd.DataFrame(infos).transpose()

    ## expanding on certain columns to be used in RECON

    for col_name in ['study', 'tracer injection']:
        df[f'{col_name} time'] = df.apply(lambda col: format_date(
            col[f'{col_name} date (yyyy:mm:dd)'], col[
                f'{col_name} time (hh:mm:ss GMT+00:00)']),
                                          axis=1)
        df.drop(f'{col_name} date (yyyy:mm:dd)', axis=1, inplace=True)
        df.drop(f'{col_name} time (hh:mm:ss GMT+00:00)', axis=1, inplace=True)

    # convert tracer dose to MBq
    df['tracer dose (MBq)'] = df[
        'tracer activity at time of injection (Bq)'].apply(
            lambda x: float(x) / 1e6)
    df.drop('tracer activity at time of injection (Bq)', axis=1, inplace=True)

    # get delay between injection and scan
    df['time delay (sec)'] = df.apply(time_delay, axis=1)
    # get the most common image duration (mode)
    duration_mode = df['image duration (sec)'].mode()[0]
    # get the average delay between injection and scan
    avg_delay_at_mode = df[df['image duration (sec)'] ==
                           duration_mode]['time delay (sec)'].mean()
    # find the starting value for recon
    df['delay for recon'] = df.apply(delay_for_recon,
                                     args=(duration_mode, avg_delay_at_mode),
                                     axis=1)

    # map the StationName onto ScannerName
    df['ScannerName'] = df['StationName'].map(scanner_type)

    # export dataframe as csv
    df.to_csv(
        anonym_data_dir.joinpath(f'{project_id}_anonymized_patient_info.csv'))


if __name__ == '__main__':
    main()