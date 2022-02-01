import os
from subprocess import run
from shutil import copy2, copytree, rmtree, move, ignore_patterns
from tqdm import tqdm
import argparse 
from pathlib import Path
import pandas as pd
import time

def edit_chopper_params_file(params_file, pct=[10]):    
    new_params = []
    with open(params_file, "r") as f:
        for line in f.readlines():
            # comment out any line that contains Retain
            if 'Retain' in line and not '#' in line:
                #line = '# ' + line
                continue
            new_params.append(line)
        # add new Retain values at the end of file
        for pc in pct:
            new_params.append(f'Retain := {pc}\n')

    new_params = ''.join(new_params)

    with open(params_file, "w") as f:
        f.write(new_params)
    
    time.sleep(1)
        

def get_lm_file(folder):
    ptd_files = [f for f in folder.iterdir() if '.ptd' in f.name]
    sorted_files = sorted(ptd_files, key=os.path.getsize, reverse=True)
    return sorted_files[0]


##########################
##### USER INPUT #########
##########################
parser = argparse.ArgumentParser(description='Generate anonym ID for each patient folder and extract study UID')
parser.add_argument("-i", "--input", 
                    help="Directory containing patients data. Will use current working directory if nothing passed", 
                    type=str, default=os.getcwd())
# parser.add_argument("-o", "--output", 
#                     help="Output directory for chopped files. Default is 'raw_data_anonymized_lowdose'", 
#                     type=str, default="raw_data_anonymized_lowdose")
parser.add_argument("-cp", "--chop-percent", 
                    help="Percent for lowdose chopping. Use space to separate multiple values e.g. 10 20", 
                    nargs='*', type=int, default=[])
parser.add_argument("-fl", "--fixed-lowdose", 
                    help="""Target fixed lowdose in MBq value if not using 'chop-percent'. 
                            Will calculate chop-percent param based on the actual dose of study.""", 
                    type=int, default=0)
args = parser.parse_args()

data_dir = Path(args.input)
project_id = data_dir.parent.name

# chopper params
chopper_params_file = r"C:\JSRecon12\LMChopper64\LMChopper_params.txt"
percent = args.chop_percent
fixed_lowdose_value = args.fixed_lowdose    

# load info dataframe to work with doses
lm_info = data_dir.joinpath(f'{project_id}_anonymized_patient_info.csv')
df = pd.read_csv(lm_info, index_col=0)
# calculate the chop percent (fixed or varying)
if fixed_lowdose_value:
    df[f'LD{fixed_lowdose_value}MBq'] = df['tracer dose (MBq)'].apply(lambda x: [round(fixed_lowdose_value / x * 100, 2)])
elif percent:
    for pc in percent:
        df[f'LD{pc}pct'] = pc
else:
    raise ValueError('Must define --chop-percent or --fixed-lowdose. Use -h for help.')

# overwrite df to info file
df.to_csv(lm_info)
df.to_csv(data_dir.joinpath(lm_info.name))

##########################
##### CHOPPER ############
##########################
# save original version of params file to restore in the end
with open(chopper_params_file, "r") as f:
    old_params = f.read()

# only edit the param file once if a percent is given
if percent:
    edit_chopper_params_file(chopper_params_file, percent)
    

anonym_folders = [d for d in data_dir.iterdir() if d.is_dir()]
### Restructuring folders with lowdose + chopping LM file
for anonym_folder in tqdm(anonym_folders):
    anonym_id = anonym_folder.name
    pet_folder = anonym_folder.joinpath('PET')
    
    lowdose_anonym_folder = data_dir.joinpath(anonym_id).joinpath(f"PET_LD{percent[-1]}pct")
    
    # check if data already been chopped
    if lowdose_anonym_folder.exists():
        if len(os.listdir(lowdose_anonym_folder)) == len(os.listdir(pet_folder)):
            # patient already anonymized
            print('Data already chopped. Skipping.')
            continue
        else:
            # only partially chopped. Removing incomplete folder to start over.
            rmtree(lowdose_anonym_folder, ignore_errors=True)
        
    lowdose_anonym_folder.mkdir(parents=True, exist_ok=True)
    # copy content of anonymized folder except the LISTMODE file
    lm_file = get_lm_file(pet_folder)
    for item_path in pet_folder.iterdir():
        if item_path != lm_file:
            dst_path = lowdose_anonym_folder.joinpath(item_path.name)
            copy_func = copy2 if item_path.is_file() else copytree
            copy_func(item_path, dst_path)

    # chop the LM file directly in the lowdose folder
    os.chdir(lowdose_anonym_folder)
    # find out the chop percent
    if fixed_lowdose_value:
        percent = df.loc[anonym_id, f'LD{fixed_lowdose_value}MBq']
        edit_chopper_params_file(chopper_params_file, percent)
        full_dose_value = df.loc[anonym_id, 'tracer dose (MBq)']
        print(f"Chopping {anonym_id} with dose {full_dose_value} MBq by", *percent, "%")
    
    chop_cmd_list = ["cscript", "C:\JSRecon12\LMChopper64\LMChopper64.js", str(lm_file)]
    run(chop_cmd_list)

    # delete the IMA file
    ima_file = lowdose_anonym_folder.joinpath('TempDicomHeader.IMA')
    try:
        os.remove(ima_file)
    except OSError:
        print('No IMA file to remove')
        
    # create as many folders as percent values and copy over
    if len(percent) > 1:
        
        for counter, pc in enumerate(percent):
            folder_pc = data_dir.joinpath(anonym_id).joinpath(f"PET_LD{pc}pct")
            ignore_files = [f.name for f in lowdose_anonym_folder.iterdir() if lm_file.stem in f.name and f"-{pc:03}.000" not in f.name]
            if counter != len(percent) - 1:
                print(f"Copying {lowdose_anonym_folder.name} -> {folder_pc.name}")
                copytree(lowdose_anonym_folder, folder_pc, ignore=ignore_patterns(*ignore_files))
        
        # break at the last iteration and rename instead of copying
        for f in ignore_files:
            lowdose_anonym_folder.joinpath(f).unlink()
        # time.sleep(2)
        # below line raise PermissionError file in use by another process.. let's skip renaming
        # move(lowdose_anonym_folder, folder_pc)
    
        
print('Done chopping, Restoring LMChopper_params.txt')
with open(chopper_params_file, "w") as f:
    f.write(old_params)

os.chdir(data_dir)

    
    
    



