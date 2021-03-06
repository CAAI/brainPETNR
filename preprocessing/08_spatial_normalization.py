#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import pandas as pd
from nipype.interfaces import fsl
from nipype.interfaces.niftyreg import RegTools
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype import Node, Workflow, DataGrabber
import multiprocessing as mp


class JobProcess:
    """
    Encapsulates pipeline behavior. Primarily, this handles setting up the pipeline.

    Args:

       job (Job) : Job object containing information on data to be processed.

    Attributes:

       wf (Workflow) : nipype Workflow describing the pipeline.

    """

    def __init__(self,
                 data_id,
                 file_in,
                 scaling_factor,
                 out_dir,
                 bet_mni=False):
        self.data_id = data_id
        self.scaling_factor = scaling_factor
        self.out_dir = out_dir.joinpath(data_id)
        self.data_dir = file_in.parent.parent
        self.template = self.data_dir.parent.joinpath('etc').joinpath(
            "avg_template.nii.gz")
        self.unit_transf = self.data_dir.parent.joinpath('etc').joinpath(
            "unity_transform.txt")
        self.pet_name = file_in.name
        self.key = 'mr' if 'MR' in self.pet_name else 'pet'
        self.wf = Workflow(name=self.pet_name, base_dir=self.out_dir)

        self.preliminary_setup()
        if bet_mni:
            self.to_mni()
        else:
            self.resample()

    def preliminary_setup(self):
        """
        Creates and connects nodes later stages of processing will depend on, .e.g,
        specifies folders containing CT resp. PET images, and converts DICOM
        to NIfTI.
        
        """
        self.datasource = Node(interface=DataGrabber(
            infields=['data_id'], outfields=['struct', "func"]),
                               name='data')
        self.datasource.inputs.template = "%s/%s"
        self.datasource.inputs.base_directory = self.data_dir
        self.datasource.inputs.data_id = self.data_id
        self.datasource.inputs.template_args = dict(
            struct=[["data_id", 'CT']], func=[["data_id", self.pet_name]])
        self.datasource.inputs.sort_filelist = True

        self.convert_func = Node(interface=Dcm2niix(), name='pet')
        self.convert_func.inputs.out_filename = "pet"
        self.convert_func.inputs.merge_imgs = True

        self.convert_struct = Node(interface=Dcm2niix(), name='ct')
        self.convert_struct.inputs.out_filename = "ct"
        self.convert_struct.inputs.merge_imgs = True

        self.wf.connect([(self.datasource, self.convert_func,
                          [("func", "source_dir")])])

        self.wf.connect([(self.datasource, self.convert_struct,
                          [("struct", "source_dir")])])

        self.rescale_func = Node(interface=fsl.maths.BinaryMaths(),
                                 name='rescaled_pet')
        self.rescale_func.inputs.operation = 'mul'
        self.rescale_func.inputs.operand_value = self.scaling_factor
        self.rescale_func.inputs.out_file = "scaled_pet_to_standard.nii.gz"

        self.wf.connect([(self.convert_func, self.rescale_func,
                          [("converted_files", "in_file")])])

    def to_mni(self):
        """ Skull stripping and spatial normalization (registration to MNI space)
        """
        self.pet_to_ct = Node(interface=fsl.FLIRT(), name=f'{self.key}_to_ct')
        self.pet_to_ct.inputs.cost_func = "corratio"
        self.pet_to_ct.inputs.cost = "corratio"
        self.pet_to_ct.inputs.dof = 6
        self.pet_to_ct.inputs.out_matrix_file = "aff_pet_to_ct.txt"
        self.pet_to_ct.inputs.verbose = True
        self.pet_to_ct.inputs.out_file = "pet_to_ct.nii.gz"

        self.wf.connect([(self.convert_struct, self.pet_to_ct, [
            ("converted_files", "reference")
        ]), (self.rescale_func, self.pet_to_ct, [("out_file", "in_file")])])

        ct_threshold_below_above = Node(interface=fsl.Threshold(),
                                        name='ct_threshold')
        ct_threshold_below_above.inputs.thresh = 0
        ct_threshold_below_above.inputs.direction = "below"
        ct_threshold_below_above.inputs.args = " -uthr 100 "

        self.wf.connect([(self.convert_struct, ct_threshold_below_above,
                          [("converted_files", "in_file")])])

        ct_smooth = Node(interface=fsl.IsotropicSmooth(), name='ct_smooth')
        ct_smooth.inputs.sigma = 1

        self.wf.connect([(ct_threshold_below_above, ct_smooth, [("out_file",
                                                                 "in_file")])])

        ct_bet = Node(interface=fsl.BET(), name='ct_bet')
        ct_bet.inputs.robust = True
        ct_bet.inputs.mask = True

        self.wf.connect([(ct_smooth, ct_bet, [("out_file", "in_file")])])

        self.pet_to_ct_bet = Node(interface=fsl.ApplyMask(),
                                  name=f'{self.key}_to_ct_bet')
        self.wf.connect([
            (self.pet_to_ct, self.pet_to_ct_bet, [("out_file", "in_file")]),
            (ct_bet, self.pet_to_ct_bet, [("mask_file", "mask_file")])
        ])

        self.ct_to_avg = Node(interface=fsl.FLIRT(), name='ct_to_avg')
        self.ct_to_avg.inputs.cost_func = "corratio"
        self.ct_to_avg.inputs.cost = "corratio"
        self.ct_to_avg.inputs.dof = 12
        self.ct_to_avg.inputs.verbose = True
        self.ct_to_avg.inputs.reference = self.template
        self.ct_to_avg.inputs.out_matrix_file = "aff_ct_to_avg.txt"
        self.ct_to_avg.inputs.out_file = "ct_to_avg.nii.gz"

        self.wf.connect([(ct_bet, self.ct_to_avg, [("out_file", "in_file")])])

        self.pet_to_avg_bet = Node(interface=fsl.FLIRT(),
                                   name=f"{self.key}_to_avg_bet")
        self.pet_to_avg_bet.inputs.apply_xfm = True
        self.pet_to_avg_bet.inputs.out_file = "pet_to_avg_bet.nii.gz"
        self.wf.connect([(self.ct_to_avg, self.pet_to_avg_bet, [
            ("out_matrix_file", "in_matrix_file")
        ]), (self.ct_to_avg, self.pet_to_avg_bet, [("out_file", "reference")]),
                         (self.pet_to_ct_bet, self.pet_to_avg_bet,
                          [("out_file", "in_file")])])

        self.pet_to_avg = Node(interface=fsl.FLIRT(),
                               name=f"{self.key}_to_avg")
        self.pet_to_avg.inputs.apply_xfm = True
        self.pet_to_avg.inputs.out_file = "pet_to_avg.nii.gz"
        self.wf.connect([
            (self.ct_to_avg, self.pet_to_avg, [("out_matrix_file",
                                                "in_matrix_file")]),
            (self.ct_to_avg, self.pet_to_avg, [("out_file", "reference")]),
            (self.pet_to_ct, self.pet_to_avg, [("out_file", "in_file")])
        ])

        # new add on to get mask
        self.mask_to_avg = Node(interface=fsl.FLIRT(), name="mask_to_avg")
        self.mask_to_avg.inputs.apply_xfm = True
        self.mask_to_avg.inputs.out_file = "mask_to_avg.nii.gz"
        self.wf.connect([(self.ct_to_avg, self.mask_to_avg, [
            ("out_matrix_file", "in_matrix_file")
        ]), (self.ct_to_avg, self.mask_to_avg, [("out_file", "reference")]),
                         (ct_bet, self.mask_to_avg, [("mask_file", "in_file")])
                         ])

    def resample(self):
        """ Resampling node in case no registration was performed.
            Resampling to 256x256x256.
            WARNINGS: the chg_res_val is specific to the input data.
                      should be done by getting the pixel size of original data
                      + the target pixel number.
        """

        self.pet_resampled = Node(interface=RegTools(),
                                  name=f"{self.key}_resampled")
        self.pet_resampled.inputs.chg_res_val = 1.418, 1.418, 1.022
        self.pet_resampled.inputs.iso_flag = True
        self.pet_resampled.inputs.inter_val = 'LIN'
        self.pet_resampled.inputs.out_file = f"{self.key}_resampled.nii.gz"
        self.wf.connect([(self.rescale_func, self.pet_resampled,
                          [("out_file", "in_file")])])

        self.ct_resampled = Node(interface=RegTools(), name="ct_resampled")
        self.ct_resampled.inputs.chg_res_val = 1.172, 1.172, 1.031
        self.ct_resampled.inputs.iso_flag = True
        self.ct_resampled.inputs.inter_val = 'LIN'
        self.ct_resampled.inputs.out_file = "ct_resampled.nii.gz"
        self.wf.connect([(self.convert_struct, self.ct_resampled,
                          [("converted_files", "in_file")])])


def main():
    """ Registration sequence going from the reconstructed DICOM to a nifti file
        ready for training. The 'full-registration' option is 
        for performing MNI registration and skull stripping.
        Otherwise the data is resamped to 256x256x256.
    """
    # input folder
    parser = argparse.ArgumentParser(
        description=
        'Convert dicom files to Nifti and makes registration to MNI space.')
    parser.add_argument("-i",
                        "--input-dir",
                        help="Directory containing patients data. Default cwd",
                        type=str,
                        default=os.getcwd())
    parser.add_argument("-o",
                        "--output-dir",
                        help="Output dir. Default 'data_registered'",
                        type=str,
                        default='data_registered')
    parser.add_argument("-fr",
                        "--full-registration",
                        help="Whether to register data + skull strip",
                        action="store_true",
                        default=False)
    parser.add_argument(
        "-k",
        "--key",
        help="Specific PET file name to process. Default is all.",
        type=str,
        default='')
    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    project_id = data_dir.parent.name
    pet_key = args.key
    data_out_dir = data_dir.parent.joinpath(args.output_dir)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    # dataframe containing scaling factors
    df_file = data_dir.joinpath(
        data_dir.joinpath(f'{project_id}_scaling_factors.csv'))
    df = pd.read_csv(df_file, index_col=0)

    def process_wrapper(pet_folder):
        try:
            patient_id = pet_folder.parent.name
            scaling = df.loc[pet_folder.name, 'intensity_scaling']
            job = JobProcess(patient_id, pet_folder, scaling, data_out_dir,
                             args.full_registration)
            job.wf.run()
        except Exception as e:
            print("Error processing:", e)

    pet_folders = [
        d for d in data_dir.glob("*/*")
        if pet_key in d.name and 'CT' not in d.name
    ]

    # splitting jobs accross several CPU
    with mp.Pool(32) as p:
        p.map(process_wrapper, pet_folders)
    # process_wrapper(pet_folders[0])


if __name__ == '__main__':
    main()