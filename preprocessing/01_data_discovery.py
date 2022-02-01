import sys
sys.path.append("..")
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from parsers.ptd import ListmodeFileParser, PetctFileParser
import argparse

def get_name(name):
    
    # remove digits from patient's name
    p = ''.join([c for c in name if not c.isdigit()])
    # removes trailing underscores that comes from removing date and time
    while p.endswith('_'):
        p = p[:-1]
    return p


################################
parser = argparse.ArgumentParser(description='Generate anonym ID for each patient folder and extract StudyInstanceUID + SeriesInstanceUID (CT)')
parser.add_argument("-i", "--input", 
                    help="Directory containing patients PET data not yet anonymized. Will use current working directory if nothing passed", 
                    type=str, default=os.getcwd())
parser.add_argument("-p", "--project-id", 
                    help="Project keyword or ID to prefix to all anonym IDs. Default is name of parent folder", 
                    type=str, default='')
args = parser.parse_args()

project_id = args.project_id
raw_data_dir = args.input
anonym_data_dir = Path(raw_data_dir + '_anonymized')
anonym_data_dir.mkdir(parents=True, exist_ok=True)
raw_data_dir = Path(raw_data_dir)
if not project_id:
    project_id = raw_data_dir.parent.name

print('Initiating data parsing for project', project_id)
folders = [f for f in raw_data_dir.iterdir() if f.is_dir()]

infos = {}
for p_folder in tqdm(folders):
    """ PARSE LISTMODE FILE FOR INFO"""
    parser1 = ListmodeFileParser(p_folder)
    # parse listmode file into a list of lines
    parser1.read_tail(stopword='DICM')
    # parse lines containing := into dictionnary
    parser1.get_primary_info(include='=', exclude='!')
    # parse the rest of the info (where values aren't labeled (no :=))
    parser1.get_secondary_info()
    # turn values to int if possible
    parser1.clean_info()
    
    """ PARSE PETCT FILE FOR SeriesInstanceUID"""
    parser2 = PetctFileParser(p_folder)
    # skip patients without CT file
    if parser2.file_in:
        parser2.read_tail(stopword='CTSeriesDicomUIDforAC')
        parser2.get_primary_info()
        parser2.clean_info()
    
    infos[p_folder.name] = {**parser1.info, **parser2.info}
    
print('Cleaning up data and saving to', raw_data_dir)
# make a DataFrame and clean up
df = pd.DataFrame(infos).transpose()
# remove patients without CT data
df.dropna(subset=['CTSeriesDicomUIDforAC'], inplace=True)
# drop duplicates
subset_for_comparison = ['StudyInstanceUID', 
                         'CTSeriesDicomUIDforAC', 
                         "study time (hh:mm:ss GMT+00:00)", 
                         "study date (yyyy:mm:dd)", 
                         "name of data file"]
df.drop_duplicates(subset=subset_for_comparison, keep='last', inplace=True)

# create anonym IDs
df['folder'] = df.index
df['name'] = df['folder'].apply(get_name)

# generate anonym IDs
patient_to_anonym_id = {}
for c, p in enumerate(df['name'].unique()):
    # get folders corresponding to 1 patient (may be several)
    p_folders = [d for d in df.index if p in d]
    for i, pf in enumerate(p_folders):
        # create unique anonym ID 
        anonym_id = f'{project_id}_{c:03}_{i:03}'
        patient_to_anonym_id[pf] = anonym_id

df['AnonymID'] = df['folder'].map(patient_to_anonym_id)

# export data to csv
# df.to_csv(raw_data_dir.joinpath(f'{project_id}_unanonymized_patient_info.csv'))

# for Claes to fetch CT data
df_for_ct_match = df[['StudyInstanceUID', 'CTSeriesDicomUIDforAC']]
df_for_ct_match.to_csv(raw_data_dir.joinpath(f'{project_id}_data_for_ct_fetch.csv'))

# saving patient_name <-> Anonym ID to json file
json_file1 = raw_data_dir.joinpath(f'{project_id}_patient_to_anonym_id.json')
df['AnonymID'].to_json(json_file1, indent=1)

# saving Anonym-ID <-> StudyUID to json file
json_file2 = raw_data_dir.joinpath(f'{project_id}_anonym_id_to_uid.json')
df_tosave = df[['AnonymID', 'StudyInstanceUID']].set_index('AnonymID', drop=True)
df_tosave['StudyInstanceUID'].to_json(json_file2, indent=1)
