#!/usr/bin/env python3
import argparse
from pathlib import Path
from tqdm import tqdm
from rhtorch.utilities.config import UserConfig
from rhscripts.conversion import nifty_to_dcm
from rhscripts.nifty import (
    inv_affine, 
    merge_images, 
    apply_mask, 
    inv_mask, 
    isotropic_smooth,
    reg_resample
)

def save_dicom(pet_inferred_nifty,
               ref_pet_nifty,
               ref_pet_dicom,
               registered_pet_nifty,
               output_dicom_container, 
               sigma_val = 0, 
               inv_bet=True):
    """

    Args:
        pet_nifty (str or path): nifty file inferred from model
        ref_pet_nifty (str or path): nifty file in original resolution
        ref_pet_dicom (str or path): dicom folder obtained from e7 recon
        lowdose_pet_nifty (str or path): from config.input_files.name[0], used for training
        output_dicom_container (str or path): path where to save the output dicom folder
        sigma_val (float): to blur lowdose pet image
        inv_bet (bool): patch the rest of the head to the BET pet image
    """
    
    inferred_patient_folder = pet_inferred_nifty.parent
    registered_patient_folder = registered_pet_nifty.parent.parent
    
    if inv_bet:
        """ INVERSE skull stripping of lowdose PET image """
        
        # apply normal mask to inferred pet image to zero the background
        masked_inferred_pet_nifty = inferred_patient_folder.joinpath("masked_" + pet_inferred_nifty.name)
        mask_file = registered_patient_folder.joinpath("mask_to_avg", "struct_thresh_smooth_brain_mask_flirt.nii.gz")
        if not masked_inferred_pet_nifty.exists():
            print("Apply skull strip mask to inferred image to ensure background = 0")
            apply_mask(pet_inferred_nifty, mask_file, masked_inferred_pet_nifty)
       
        # prepare (blur) lowdose pet image
        if sigma_val:
            blurred_ref_pet = registered_pet_nifty.parent.joinpath("blurred_" + registered_pet_nifty.name)
            if not blurred_ref_pet.exists():
                print("Blurring the original PET image (not done when reconstructing initially)")                    
                isotropic_smooth(registered_pet_nifty, blurred_ref_pet, sigma_val)
        else:
            blurred_ref_pet = registered_pet_nifty
            
        # inverse mask - if not done
        inv_mask_file = mask_file.parent.joinpath("inv_" + mask_file.name)
        if not inv_mask_file.exists():
            print("Inverting skull strip mask")
            inv_mask(mask_file, inv_mask_file)
            
        # apply inverted mask to lowdose pet image
        inv_skull_strip = inferred_patient_folder.joinpath("anti_bet_" + blurred_ref_pet.name)
        if not inv_skull_strip.exists():
            print("Applying inverted skull strip mask to original PET")
            apply_mask(blurred_ref_pet, inv_mask_file, inv_skull_strip)
            
        
        #### MERGE skull strip + rest of pet image from lowdose file
        full_pet_infer = inferred_patient_folder.joinpath("full_" + pet_inferred_nifty.name)
        if not full_pet_infer.exists():
            print("Merging inferred skull strip + inverted skull strip into full image")
            merge_images(masked_inferred_pet_nifty, inv_skull_strip, full_pet_infer)
    
    else:
        full_pet_infer = pet_inferred_nifty
    
    
    ##### RESAMPLING 256x256x256 MNI space -> original res patient space(400x400x109)        
    inv_aff_file = registered_patient_folder.joinpath("inv_aff_transf", "inv_aff_pet_to_avg.txt")
    resampled_pet_nifty = inferred_patient_folder.joinpath("original_res_" + pet_inferred_nifty.name)
    if not resampled_pet_nifty.exists():
        print("Resampling to original resolution")
        reg_resample(ref_pet_nifty, full_pet_infer, inv_aff_file, resampled_pet_nifty, 'LIN', 0.0)
    
    
    ######## CONVERTING TO DICOM FORMAT
    # convert nifti to dcm
    # if not output_dicom_container.exists():
    #     print("Final step: converting nifty to dicom folder")
    #     nifty_to_dcm(resampled_pet_nifty, 
    #         ref_pet_dicom, 
    #         output_dicom_container, 
    #         patient_id=patient, 
    #         forceRescaleSlope=True)

    
def main():
    parser = argparse.ArgumentParser(
        description='Infer new data from input model and save a nifti file + dicom folder')
    parser.add_argument("-c", "--config",
                        help="Config file of saved model",
                        type=str, default='config.yaml')
    parser.add_argument("-rd", "--reference-dicom", 
                        help="Directory containing the original dicom files. Default 'data_recon' in project dir.", 
                        type=str, default='data_recon')

    args = parser.parse_args()
    external_set = 'external' in args.config
    # always need to blur low dose data 2mm -> 5mm
    sigma = 3.9/2.3548
    
    # load configs in inference mode
    user_configs = UserConfig(args, mode='infer')
    model_dir = user_configs.rootdir
    configs = user_configs.hparams

    # paths and variables info
    project_dir = Path(configs['project_dir'])
    # project_id = project_dir.name
    data_dir = Path(configs['data_folder'])
    # lowdose pet name used for training
    input_filename = configs['input_files']['name'][0]
    # base name before cropping and registration
    base_filename = input_filename.split('_MNI_BET')[0]
    
    model_name = configs['model_name']
    # data dir containing the reconstructed PET in dicom format
    reference_dicom_dir = project_dir.joinpath('data_recon')
    # data dir containing the NIFTY formats
    nifty_dir = project_dir.joinpath('data_registered')

    infer_dir_name = 'inferences_external' if external_set else 'inferences'
    infer_dir = model_dir.joinpath(infer_dir_name)
    if not infer_dir.exists():
        raise FileNotFoundError("No inferences found! Run torchio_inference.py first.")
    
    patient_folders = [f for f in infer_dir.iterdir()]
    for inferred_patient_folder in tqdm(patient_folders):

        patient = inferred_patient_folder.name
        # inferred nifti PET
        inferred_pet_nifty = next(inferred_patient_folder.iterdir())
        
        # reference nifti PET in orginal resolution (before registration)
        ref_pet_nifty = nifty_dir.joinpath(patient, base_filename, "dcm_conv_func", "func.nii.gz")
        
        # registered nifty PET - in data_registered
        registered_pet_nifty = nifty_dir.joinpath(patient, base_filename, "pet_to_avg", "func_maths_flirt_flirt.nii.gz")
        
        # ref dicom folder - in data_recon
        ref_pet_dicom = reference_dicom_dir.joinpath(patient, base_filename)
        
        # output dicom folder
        output_dicom_container = inferred_patient_folder.joinpath(base_filename + "_INFER")
        
        bet = 'BET' in input_filename
        
        if not output_dicom_container.exists():
            # save a dicom folder
            try:
                save_dicom(inferred_pet_nifty,
                    ref_pet_nifty,
                    ref_pet_dicom,
                    registered_pet_nifty,
                    output_dicom_container,
                    sigma, 
                    bet)
            except RuntimeError:
                print(f"Failed to convert {patient}. Continuing.")

        else:
            print(f'Data already converted to DICOM with model {model_name}')


if __name__ == '__main__':
    main()