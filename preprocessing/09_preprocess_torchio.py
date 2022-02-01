import os
from pathlib import Path
from tqdm import tqdm
import argparse
import torchio as tio
import shutil
import numpy as np
from rhtorch.utilities.config import UserConfig


# normalizaions
def ct_soft_norm(data):
    data_trunc = np.maximum(-50, data)
    data_trunc = np.minimum(data_trunc, 50)
    return data_trunc / 50

    
# copy PET file to dst folder
def copy_and_preprocess_file(file, base_filename, out_dir, new_shape, normalize_func=None, crop_func=None):
    if not file.exists():
        return 
    
    copy_file = out_dir.joinpath(f"{base_filename}.nii.gz")
    if not copy_file.exists(): 
        shutil.copy2(file, copy_file)
    
    # crop and normalize PET for training
    output_file = out_dir.joinpath(f"{base_filename}_{new_shape}_NORM.nii.gz")
    if not output_file.exists():
        image = tio.ScalarImage(copy_file)
        if normalize_func:
            data = normalize_func(image)
        if crop_func:
            data = crop_func(data)
        data.save(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Resample all nifty files in a given directory based on target pixel size.')
    parser.add_argument("-i", "--input-dir",
                        help="Directory containing patients data. Will use current working directory if nothing passed",
                        type=str, default=os.getcwd())
    parser.add_argument("-o", "--output", 
                        help="Output directory for recon data. Default is 'data_for_training' if nothing passed.", 
                        type=str, default='data_for_training')
    parser.add_argument("-c", "--config",
                        help="Config file else than 'config.yaml' in project directory (input dir)",
                        type=str, default=os.getcwd() + '/config.yaml')
    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    config = UserConfig(args, mode='preprocess').hparams
    output_dir = data_dir.parent.joinpath(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # normalization
    normalize = tio.Lambda(lambda x: x / config['pet_normalization_constant'])
    normalize_mr = tio.Lambda(lambda x: x / config['mr_normalization_constant'])
    normalize_ct = tio.Lambda(ct_soft_norm)
    
    # cropping - from config file
    final_shape = ""
    crop = None
    if 'cropping' in config:
        crop_config = config['cropping']
        crop = tio.Crop((*crop_config['x_lim'], *crop_config['y_lim'], *crop_config['z_lim']))
        x_dim = config['data_shape'][0] - np.sum(crop_config['x_lim'])
        y_dim = config['data_shape'][1] - np.sum(crop_config['y_lim'])
        z_dim = config['data_shape'][2] - np.sum(crop_config['z_lim'])
        final_shape = f"{x_dim}x{y_dim}x{z_dim}"
    
    # printing preprocessing steps
    print(f"""
        NORMALIZATION:
        PET -> divided by {config['pet_normalization_constant']}
        MR -> divided by {config['mr_normalization_constant']}
        CT -> "soft tissue normalization": values between (-50, 50) kept and divided by 50
        
        CROPPING:
        original shape: {config['data_shape'][0]} x {config['data_shape'][1]} x {config['data_shape'][2]}
           final shape: {x_dim} x {y_dim} x {z_dim}
    """)

    pet_folders = [d for d in data_dir.glob("*/*") if d.is_dir()]
    for folder in tqdm(pet_folders):
        patient_id = folder.parent.name
        patient_out_dir = output_dir.joinpath(patient_id)
        patient_out_dir.mkdir(parents=True, exist_ok=True)
        
        # copy PET or MR file to dst folder
        pet_file = folder.joinpath("pet_to_avg_bet", "func_maths_flirt_masked_flirt.nii.gz")
        pet_base_filename = folder.name + "_MNI_BET"
        copy_and_preprocess_file(pet_file, pet_base_filename, patient_out_dir, final_shape, normalize, crop)
        
        mr_file = folder.joinpath("mr_to_avg", "func_maths_flirt_masked_flirt.nii.gz")
        mr_base_filename = folder.name + "_MNI_BET"
        copy_and_preprocess_file(mr_file, mr_base_filename, patient_out_dir, final_shape, normalize_mr, crop)
            
        # copy CT file to dst folder
        ct_file = folder.joinpath('ct_to_avg', 'struct_thresh_smooth_brain_flirt.nii.gz')
        ct_base_filename = "CT_MNI_BET.nii.gz"
        copy_and_preprocess_file(ct_file, ct_base_filename, patient_out_dir, final_shape, normalize_ct, crop)
        
        
if __name__ == '__main__':
    main()