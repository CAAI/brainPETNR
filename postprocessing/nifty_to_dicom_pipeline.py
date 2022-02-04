#!/usr/bin/env python3
import argparse
from pathlib import Path
from tqdm import tqdm
from rhtorch.utilities.config import UserConfig
from rhscripts.conversion import nifty_to_dcm
from nipype.interfaces import fsl
# from nipype.interfaces.dcm2nii import Dcm2niix
# from nipype import Node, Workflow, DataGrabber
# import multiprocessing as mp

# class JobProcess:
#     """
#     Encapsulates pipeline behavior. Primarily, this handles setting up the pipeline.

#     Args:

#        job (Job) : Job object containing information on data to be processed.

#     Attributes:

#        wf (Workflow) : nipype Workflow describing the pipeline.

#     """

#     def __init__(self, data_id, file_in, scaling_factor, out_dir):
#         self.data_id = data_id
#         self.scaling_factor = scaling_factor
#         self.out_dir = out_dir.joinpath(data_id)
#         self.data_dir = file_in.parent.parent
#         self.pet_name = file_in.name
#         self.wf = Workflow(name=self.pet_name, base_dir=self.out_dir)

#         self._preliminary_setup()

#     def _preliminary_setup(self):
#         """
#         Creates and connects nodes later stages of processing will depend on, .e.g,
#         specifies folders containing CT resp. PET images, and converts DICOM
#         to NIfTI. Also, skullstripping is performed, images are spatially normalized.

#         """
#         self.datasource = Node(interface=DataGrabber(infields=['data_id'],
#                                                      outfields=['func']),
#                                name='data')
#         self.datasource.inputs.template = "%s/%s"
#         self.datasource.inputs.base_directory = self.data_dir
#         self.datasource.inputs.data_id = self.data_id
#         self.datasource.inputs.template_args = dict(
#             func=[["data_id", self.pet_name]])
#         self.datasource.inputs.sort_filelist = True


def save_dicom(pet_inferred_nifty, ref_pet_nifty, ref_pet_dicom,
               registered_pet_nifty, output_dicom_container):
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
    registered_patient_folder = registered_pet_nifty.parent.parent
    """ INVERSE skull stripping of lowdose PET image """

    # apply normal mask to inferred pet image to zero the background
    masked_inferred_pet_nifty = inferred_patient_folder.joinpath(
        "masked_" + pet_inferred_nifty.name)
    mask_file = registered_patient_folder.joinpath(
        "mask_to_avg", "struct_thresh_smooth_brain_mask_flirt.nii.gz")
    if not masked_inferred_pet_nifty.exists():
        print("Apply skull strip mask to inferred image.")
        mask = fsl.ApplyMask()
        mask.inputs.in_file = pet_inferred_nifty
        mask.inputs.mask_file = mask_file
        mask.inputs.out_file = masked_inferred_pet_nifty
        mask.run()

    # prepare (blur) lowdose pet image
    blurred_ref_pet = registered_pet_nifty.parent.joinpath(
        "blurred_" + registered_pet_nifty.name)
    if not blurred_ref_pet.exists():
        print("Blurring the low-activity PET image")
        smoothing = fsl.IsotropicSmooth()
        smoothing.inputs.in_file = registered_pet_nifty
        smoothing.inputs.sigma = sigma
        smoothing.inputs.out_file = blurred_ref_pet
        smoothing.run()

    # inverse mask - if not done
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
        multiplier.inputs.in_file = masked_inferred_pet_nifty
        multiplier.inputs.operation = 'add'
        multiplier.inputs.operand_file = inv_skull_strip
        multiplier.inputs.out_file = full_pet_infer
        multiplier.run()

    ##### RESAMPLING 256x256x256 MNI space -> original res patient space(400x400x109)
    inv_aff_file = registered_patient_folder.joinpath(
        "inv_aff_transf", "inv_aff_pet_to_avg.txt")
    resampled_pet_nifty = inferred_patient_folder.joinpath(
        "original_res_" + pet_inferred_nifty.name)
    if not resampled_pet_nifty.exists():
        print("Resampling to original resolution")
        rsl = fsl.FLIRT()
        rsl.inputs.reference = ref_pet_nifty
        rsl.inputs.in_file = full_pet_infer
        rsl.inputs.in_matrix_file = inv_aff_file
        rsl.inputs.apply_xfm = True
        rsl.inputs.out_file = resampled_pet_nifty
        rsl.run()

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
    parser.add_argument(
        "-rd",
        "--reference-dicom",
        help=
        "Directory containing the original dicom files. Default 'data_recon' in project dir.",
        type=str,
        default='data_dicom')

    args = parser.parse_args()
    external_set = 'external' in args.config
    # always need to blur low dose data 2mm -> 5mm
    sigma = 3.9 / 2.3548

    # load configs in inference mode
    user_configs = UserConfig(args, mode='infer')
    model_dir = user_configs.rootdir
    configs = user_configs.hparams

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

    infer_dir_name = 'inferences_external' if external_set else 'inferences'
    infer_dir = model_dir.joinpath(infer_dir_name)
    if not infer_dir.exists():
        raise FileNotFoundError(
            "No inferences found! Run torchio_inference.py first.")

    patient_folders = [f for f in infer_dir.iterdir()]
    for inferred_patient_folder in tqdm(patient_folders):

        patient = inferred_patient_folder.name
        # inferred nifti PET
        inferred_pet_nifty = next(inferred_patient_folder.iterdir())

        # reference nifti PET in orginal resolution (before registration)
        ref_pet_nifty = nifty_dir.joinpath(patient, base_filename, "pet",
                                           "pet.nii.gz")

        # registered nifty PET - in data_registered
        registered_pet_nifty = nifty_dir.joinpath(patient, base_filename,
                                                  "pet_to_avg",
                                                  "pet_to_avg.nii.gz")

        # ref dicom folder - in data_recon
        ref_pet_dicom = reference_dicom_dir.joinpath(patient, base_filename)

        # output dicom folder
        output_dicom_container = inferred_patient_folder.joinpath(
            base_filename + "_INFER")

        # bet = 'BET' in input_filename

        if not output_dicom_container.exists():
            # save a dicom folder
            try:
                save_dicom(inferred_pet_nifty, ref_pet_nifty, ref_pet_dicom,
                           registered_pet_nifty, output_dicom_container)
            except RuntimeError:
                print(f"Failed to convert {patient}. Continuing.")

        else:
            print(f'Data already converted to DICOM with model {model_name}')


if __name__ == '__main__':
    main()