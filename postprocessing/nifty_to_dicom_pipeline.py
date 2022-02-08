#!/usr/bin/env python3
import argparse
from pathlib import Path
from tqdm import tqdm
from rhtorch.utilities.config import UserConfig
from rhscripts.conversion import nifty_to_dcm
from nipype.interfaces import fsl


def save_dicom(pet_inferred_nifty, registered_patient_folder, ref_pet_dicom,
               output_dicom_container):
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
    sigma = 3.9 / 2.3548
    inferred_patient_folder = pet_inferred_nifty.parent
    patient = inferred_patient_folder.name
    original_res_pet = registered_patient_folder.joinpath("pet", "pet.nii.gz")
    registered_pet = registered_patient_folder.joinpath(
        "pet_to_avg", "pet_to_avg.nii.gz")
    """ INVERSE skull stripping of lowdose PET image """
    # prepare (blur) lowdose pet image
    blurred_ref_pet = inferred_patient_folder.joinpath("blurred_" +
                                                       registered_pet.name)
    if not blurred_ref_pet.exists():
        print("Blurring the low-activity PET image")
        smoothing = fsl.IsotropicSmooth()
        smoothing.inputs.in_file = registered_pet
        smoothing.inputs.sigma = sigma
        smoothing.inputs.out_file = blurred_ref_pet
        smoothing.run()

    # inverse mask - if not done
    mask_file = registered_patient_folder.joinpath("mask_to_avg",
                                                   "mask_to_avg.nii.gz")
    inv_mask_file = mask_file.parent.joinpath("inv_" + mask_file.name)
    if not inv_mask_file.exists():
        print("Inverting skull strip mask")
        inverter = fsl.UnaryMaths()
        inverter.inputs.in_file = mask_file
        inverter.inputs.operation = 'binv'
        inverter.inputs.out_file = inv_mask_file
        inverter.inputs.output_type = "NIFTI_GZ"
        inverter.run()

    # apply inverted mask to lowdose pet image
    inv_skull_strip = inferred_patient_folder.joinpath("anti_bet_" +
                                                       blurred_ref_pet.name)
    if not inv_skull_strip.exists():
        print("Applying inverted skull strip mask to original PET")
        mask = fsl.ApplyMask()
        mask.inputs.in_file = blurred_ref_pet
        mask.inputs.mask_file = inv_mask_file
        mask.inputs.out_file = inv_skull_strip
        mask.run()

    #### MERGE skull strip + rest of pet image from lowdose file
    full_pet_infer = inferred_patient_folder.joinpath("full_" +
                                                      pet_inferred_nifty.name)
    if not full_pet_infer.exists():
        print("Merging inferred skull strip + brain contour into full image")
        multiplier = fsl.BinaryMaths()
        multiplier.inputs.in_file = pet_inferred_nifty
        multiplier.inputs.operation = 'add'
        multiplier.inputs.operand_file = inv_skull_strip
        multiplier.inputs.out_file = full_pet_infer
        multiplier.run()

    ##### CONCATENATING pet_to_ct + ct_to_avg AFFINE TRANSFORMS
    ct_to_avg_transf = registered_patient_folder.joinpath(
        "ct_to_avg", "aff_ct_to_avg.txt")
    pet_to_ct_transf = registered_patient_folder.joinpath(
        "pet_to_ct", "aff_pet_to_ct.txt")
    aff = inferred_patient_folder.joinpath("aff.txt")
    if not aff.exists():
        print("Inverting the affine transform to revert registration.")
        aff_transf = fsl.ConvertXFM()
        aff_transf.inputs.concat_xfm = True
        aff_transf.inputs.in_file = pet_to_ct_transf
        aff_transf.inputs.in_file2 = ct_to_avg_transf
        aff_transf.inputs.out_file = aff
        aff_transf.run()

    #### INVERTING PET_TO_AVG AFFINE TRANSFORM
    inv_aff = inferred_patient_folder.joinpath("inv_aff.txt")
    if not inv_aff.exists():
        inv_aff_transf = fsl.ConvertXFM()
        inv_aff_transf.inputs.invert_xfm = True
        inv_aff_transf.inputs.in_file = aff
        inv_aff_transf.inputs.out_file = inv_aff
        inv_aff_transf.run()

    ##### RESAMPLING 256x256x256 MNI space -> original res patient space(400x400x109)
    resampled_pet_nifty = inferred_patient_folder.joinpath(
        "original_res_" + pet_inferred_nifty.name)
    if not resampled_pet_nifty.exists():
        rsl2 = fsl.FLIRT()
        rsl2.inputs.reference = original_res_pet
        rsl2.inputs.in_file = full_pet_infer
        rsl2.inputs.in_matrix_file = inv_aff
        rsl2.inputs.apply_xfm = True
        rsl2.inputs.out_file = resampled_pet_nifty
        rsl2.run()

    ######## CONVERTING TO DICOM FORMAT
    if not output_dicom_container.exists():
        print("Final step: converting nifty to dicom folder")
        nifty_to_dcm(resampled_pet_nifty,
                     ref_pet_dicom,
                     output_dicom_container,
                     patient_id=patient,
                     forceRescaleSlope=True)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Infer new data from input model and save a nifti file + dicom folder')
    parser.add_argument("-c",
                        "--config",
                        help="Config file of saved model",
                        type=str,
                        default='config.yaml')
    parser.add_argument("-im",
                        "--infer-mode",
                        help="Which patient set to use: test or eval.",
                        type=str,
                        default='test')
    parser.add_argument(
        "-rd",
        "--reference-dicom",
        help=
        "Directory containing the original dicom files. Default 'data_dicom' in project dir.",
        type=str,
        default='data_dicom')

    args = parser.parse_args()
    infer_mode = args.infer_mode
    # always need to blur low dose data 2mm -> 5mm
    # sigma = 3.9 / 2.3548

    # load configs in inference mode
    user_configs = UserConfig(args, mode='infer')
    configs = user_configs.hparams
    model_dir = Path(configs['project_dir']).joinpath('trained_models',
                                                      configs['model_name'])

    # paths and variables info
    project_dir = Path(configs['project_dir'])
    # project_id = project_dir.name
    # data_dir = Path(configs['data_folder'])
    # lowdose pet name used for training
    input_filename = configs['input_files']['name'][0]
    # base name before cropping and registration
    base_filename = input_filename.split('_MNI_BET')[0]

    model_name = configs['model_name']
    # data dir containing the reconstructed PET in dicom format
    reference_dicom_dir = project_dir.joinpath(args.reference_dicom)
    # data dir containing the NIFTY formats
    nifty_dir = project_dir.joinpath('data_registered')

    infer_dir_name = f'inferences_{infer_mode}_set'
    infer_dir = model_dir.joinpath(infer_dir_name)
    print(model_dir)
    if not infer_dir.exists():
        raise FileNotFoundError(
            "No inferences found! Run torchio_inference.py first.")

    patient_folders = [f for f in infer_dir.iterdir()]
    for inferred_patient_folder in tqdm(patient_folders):

        patient = inferred_patient_folder.name
        # inferred nifti PET
        inferred_pet_nifty = next(inferred_patient_folder.iterdir())

        # registered nifty PET - in data_registered
        registered_dir = nifty_dir.joinpath(patient, base_filename)

        # ref dicom folder - in data_recon
        ref_pet_dicom = reference_dicom_dir.joinpath(patient, base_filename)

        # output dicom folder
        output_dicom_container = inferred_patient_folder.joinpath(
            base_filename + "_INFER")

        # bet = 'BET' in input_filename

        if not output_dicom_container.exists():
            # save a dicom folder
            try:
                save_dicom(inferred_pet_nifty, registered_dir, ref_pet_dicom,
                           output_dicom_container)
            except RuntimeError:
                print(f"Failed to convert {patient}. Continuing.")

        else:
            print(f'Data already converted to DICOM with model {model_name}')


if __name__ == '__main__':
    main()