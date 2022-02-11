#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import argparse
from nipype.interfaces import fsl
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.utility import Function
from nipype import Node, Workflow, DataGrabber
from rhtorch.utilities.config import UserConfig
import multiprocessing as mp
import utils


class JobProcess:
    """
    Encapsulates pipeline behavior. Primarily, this handles setting up the pipeline.

    Args:

       job (Job) : Job object containing information on data to be processed.

    Attributes:

       wf (Workflow) : nipype Workflow describing the pipeline.

    """

    def __init__(self, data_id, data_dir, pet_dicom_name, configs, out_dir):
        self.data_id = data_id
        self.configs = configs
        self.out_dir = out_dir.joinpath(data_id)
        self.data_dir = data_dir
        # self.training_dir = self.data_dir.parent.joinpath('data_for_training')
        self.template = self.data_dir.parent.joinpath('etc').joinpath(
            "avg_template.nii.gz")
        # if not specified by user, get lowdose tag from configs
        if not pet_dicom_name:
            pet_input = configs['input_files']['name'][0]
            pet_dicom_name = pet_input.split('_MNI_BET')[0]
        self.pet_dicom_name = pet_dicom_name
        self.wf = Workflow(name=self.pet_dicom_name, base_dir=self.out_dir)

        # process by step
        self.to_mni_space()
        self.make_inference()
        self.to_patient_space()
        self.to_dicom()

    def to_mni_space(self):
        """
        Creates and connects nodes later stages of processing will depend on, .e.g,
        specifies folders containing CT resp. PET images, and converts DICOM
        to NIfTI. Also, skullstripping is performed, images are spatially normalized.
        
        """
        self.datasource = Node(interface=DataGrabber(
            infields=['data_id'], outfields=['struct', "func"]),
                               name='data')
        self.datasource.inputs.template = "%s/%s"
        self.datasource.inputs.base_directory = self.data_dir
        self.datasource.inputs.data_id = self.data_id
        self.datasource.inputs.template_args = dict(
            struct=[["data_id", 'CT']],
            func=[["data_id", self.pet_dicom_name]])
        self.datasource.inputs.sort_filelist = True

        self.pet = Node(interface=Dcm2niix(), name='pet')
        self.pet.inputs.out_filename = "pet"
        self.pet.inputs.merge_imgs = True

        self.ct = Node(interface=Dcm2niix(), name='ct')
        self.ct.inputs.out_filename = "ct"
        self.ct.inputs.merge_imgs = True

        self.wf.connect([(self.datasource, self.pet, [("func", "source_dir")])
                         ])

        self.wf.connect([(self.datasource, self.ct, [("struct", "source_dir")])
                         ])

        # rescaling not necessary - or is it ??
        # self.rescale_func = Node(interface=fsl.maths.BinaryMaths(),
        #                          name='rescaled_pet')
        # self.rescale_func.inputs.operation = 'mul'
        # self.rescale_func.inputs.operand_value = self.scaling_factor
        # self.rescale_func.inputs.out_file = "scaled_pet_to_standard.nii.gz"

        # self.wf.connect([(self.convert_func, self.rescale_func,
        #                   [("converted_files", "in_file")])])

        self.pet_to_ct = Node(interface=fsl.FLIRT(), name='pet_to_ct')
        self.pet_to_ct.inputs.cost_func = "corratio"
        self.pet_to_ct.inputs.cost = "corratio"
        self.pet_to_ct.inputs.dof = 6
        self.pet_to_ct.inputs.out_matrix_file = "aff_pet_to_ct.txt"
        self.pet_to_ct.inputs.verbose = True
        self.pet_to_ct.inputs.out_file = "pet_to_ct.nii.gz"

        self.wf.connect([
            (self.ct, self.pet_to_ct, [("converted_files", "reference")]),
            (self.pet, self.pet_to_ct, [("converted_files", "in_file")])
        ])

        ct_threshold_below_above = Node(interface=fsl.Threshold(),
                                        name='ct_threshold')
        ct_threshold_below_above.inputs.thresh = 0
        ct_threshold_below_above.inputs.direction = "below"
        ct_threshold_below_above.inputs.args = " -uthr 100 "

        self.wf.connect([(self.ct, ct_threshold_below_above,
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
                                  name='pet_to_ct_bet')
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
                                   name="pet_to_avg_bet")
        self.pet_to_avg_bet.inputs.apply_xfm = True
        self.pet_to_avg_bet.inputs.out_file = "pet_to_avg_bet.nii.gz"
        self.wf.connect([(self.ct_to_avg, self.pet_to_avg_bet, [
            ("out_matrix_file", "in_matrix_file")
        ]), (self.ct_to_avg, self.pet_to_avg_bet, [("out_file", "reference")]),
                         (self.pet_to_ct_bet, self.pet_to_avg_bet,
                          [("out_file", "in_file")])])

        self.pet_to_avg = Node(interface=fsl.FLIRT(), name="pet_to_avg")
        self.pet_to_avg.inputs.apply_xfm = True
        self.pet_to_avg.inputs.out_file = "pet_to_avg.nii.gz"
        self.wf.connect([
            (self.ct_to_avg, self.pet_to_avg, [("out_matrix_file",
                                                "in_matrix_file")]),
            (self.ct_to_avg, self.pet_to_avg, [("out_file", "reference")]),
            (self.pet_to_ct, self.pet_to_avg, [("out_file", "in_file")])
        ])

        self.mask_to_avg = Node(interface=fsl.FLIRT(), name="mask_to_avg")
        self.mask_to_avg.inputs.apply_xfm = True
        self.mask_to_avg.inputs.out_file = "mask_to_avg.nii.gz"
        self.wf.connect([(self.ct_to_avg, self.mask_to_avg, [
            ("out_matrix_file", "in_matrix_file")
        ]), (self.ct_to_avg, self.mask_to_avg, [("out_file", "reference")]),
                         (ct_bet, self.mask_to_avg, [("mask_file", "in_file")])
                         ])

    def make_inference(self):
        self.inferred_pet_to_avg_bet = Node(interface=Function(
            function=utils.infer_from_model,
            input_names=["in_file", "configs"],
            output_names=["out_file"]),
                                            name="inferred_pet_to_avg_bet")
        self.inferred_pet_to_avg_bet.inputs.configs = self.configs
        self.wf.connect([(self.pet_to_avg_bet, self.inferred_pet_to_avg_bet,
                          [("out_file", "in_file")])])

    def to_patient_space(self):
        """ INVERSE skull stripping of lowdose PET image """
        # prepare (blur) lowdose pet image

        self.blurred_pet_to_avg = Node(interface=fsl.IsotropicSmooth(),
                                       name="blurred_pet_to_avg")
        self.blurred_pet_to_avg.inputs.sigma = 3.9 / 2.3548  # 2mm -> 5mm
        self.blurred_pet_to_avg.inputs.out_file = "blurred_pet_to_avg.nii.gz"
        self.wf.connect([(self.pet_to_avg, self.blurred_pet_to_avg,
                          [("out_file", "in_file")])])

        # inverse mask - if not done
        self.inverted_mask_to_avg = self.smoothing = Node(
            interface=fsl.UnaryMaths(), name="inverted_mask_to_avg")
        self.inverted_mask_to_avg.inputs.operation = 'binv'
        self.inverted_mask_to_avg.inputs.out_file = "inv_mask_to_avg.nii.gz"
        self.wf.connect([(self.mask_to_avg, self.inverted_mask_to_avg,
                          [("out_file", "in_file")])])

        # apply inverted mask to lowdose pet image
        self.reserved_skull_strip = Node(interface=fsl.ApplyMask(),
                                         name="reserved_skull_strip")
        self.reserved_skull_strip.inputs.out_file = "reserved_skull_strip.nii.gz"
        self.wf.connect([(self.blurred_pet_to_avg, self.reserved_skull_strip,
                          [("out_file", "in_file")]),
                         (self.inverted_mask_to_avg, self.reserved_skull_strip,
                          [("out_file", "mask_file")])])

        #### MERGE skull strip + rest of pet image from lowdose file
        self.inferred_pet_to_avg = Node(interface=fsl.BinaryMaths(),
                                        name="inferred_pet_to_avg")
        self.inferred_pet_to_avg.inputs.operation = 'add'
        self.inferred_pet_to_avg.inputs.out_file = "inferred_pet_to_avg.nii.gz"
        self.wf.connect([(self.inferred_pet_to_avg_bet,
                          self.inferred_pet_to_avg, [("out_file", "in_file")]),
                         (self.reserved_skull_strip, self.inferred_pet_to_avg,
                          [("out_file", "operand_file")])])

        ##### CONCATENATING pet_to_ct + ct_to_avg AFFINE TRANSFORMS
        self.aff_pet_to_avg = Node(interface=fsl.ConvertXFM(),
                                   name="aff_pet_to_avg")
        self.aff_pet_to_avg.inputs.concat_xfm = True
        self.aff_pet_to_avg.inputs.out_file = "aff_pet_to_avg.txt"
        self.wf.connect([(self.pet_to_ct, self.aff_pet_to_avg,
                          [("out_matrix_file", "in_file")]),
                         (self.ct_to_avg, self.aff_pet_to_avg,
                          [("out_matrix_file", "in_file2")])])

        #### INVERTING PET_TO_AVG AFFINE TRANSFORM
        self.inv_aff_pet_to_avg = Node(interface=fsl.ConvertXFM(),
                                       name="inv_aff_pet_to_avg")
        self.inv_aff_pet_to_avg.inputs.invert_xfm = True
        self.inv_aff_pet_to_avg.inputs.out_file = "inv_aff_pet_to_avg.txt"
        self.wf.connect([(self.aff_pet_to_avg, self.inv_aff_pet_to_avg,
                          [("out_file", "in_file")])])

        ##### RESAMPLING 256x256x256 MNI space -> original res patient space(400x400x109)
        self.inferred_pet = Node(interface=fsl.FLIRT(), name="inferred_pet")
        self.inferred_pet.inputs.apply_xfm = True
        self.inferred_pet.inputs.out_file = "inferred_pet.nii.gz"
        self.wf.connect([(self.inv_aff_pet_to_avg, self.inferred_pet, [
            ("out_file", "in_matrix_file")
        ]), (self.pet, self.inferred_pet, [("converted_files", "reference")]),
                         (self.inferred_pet_to_avg, self.inferred_pet,
                          [("out_file", "in_file")])])

    def to_dicom(self):
        self.inferred_dicom = Node(interface=Function(
            function=utils.nifty_to_dicom,
            input_names=[
                "in_file", "ref_container", "out_container", "patient_id"
            ],
            output_names=["out_folder"]),
                                   name="inferred_dicom")
        self.inferred_dicom.inputs.patient_id = self.data_id
        self.inferred_dicom.inputs.out_container = self.out_dir.joinpath(
            self.pet_dicom_name, "denoised_dicom")
        self.wf.connect([(self.inferred_pet, self.inferred_dicom,
                          [("out_file", "in_file")]),
                         (self.datasource, self.inferred_dicom,
                          [("func", "ref_container")])])

    def clean_up(self, keep_nifti=False):
        output_dir = self.out_dir.joinpath(self.pet_dicom_name)
        all_files = [f for f in output_dir.iterdir() if f.is_file()]
        all_directories = [d for d in output_dir.iterdir() if d.is_dir()]
        # first delete random files
        for f in all_files:
            f.unlink()
        # then keep the good stuff
        if keep_nifti:
            keep_files = {
                'pet': 'original_pet',
                'inferred_pet': 'denoised_pet'
            }
            # move the kept nifties 1 level up (out of their folder)
            for k, v in keep_files.items():
                org_file = output_dir.joinpath(k, k + '.nii.gz')
                dst_file = output_dir.joinpath(v + '.nii.gz')
                if org_file.exists():
                    os.rename(org_file, dst_file)
        # finally delete all folders except dicom container
        for d in all_directories:
            if d.name != 'denoised_dicom':
                shutil.rmtree(d)


if __name__ == '__main__':

    # input folder
    parser = argparse.ArgumentParser(
        description='Inference pipeline dicom to dicom.')
    parser.add_argument("-i",
                        "--input-dir",
                        help="Input directory. Default: cwd.",
                        type=str,
                        default=os.getcwd())
    parser.add_argument("-o",
                        "--output-dir",
                        help="Output directory. Default: 'inferences'.",
                        type=str,
                        default='inferences')
    parser.add_argument("-c",
                        "--config",
                        help=".yaml config from model.",
                        type=str,
                        default=Path(
                            os.getcwd()).parent.joinpath('config.yaml'))
    parser.add_argument("-t",
                        "--tag",
                        help="Low dose dicom folder. Default: from config",
                        type=str,
                        default='')
    parser.add_argument("-mp",
                        "--multiprocess",
                        help="Whether to parallelize processes.",
                        action="store_true",
                        default=False)
    args = parser.parse_args()
    data_dir = Path(args.input_dir)
    configs = UserConfig(args, mode='infer').hparams
    inference_dir = data_dir.parent.joinpath(args.output_dir)
    inference_dir.mkdir(parents=True, exist_ok=True)

    ## LOW DOSE TAG (WHICH PET FILE TO USE FOR INFERENCE)
    def process_wrapper(patient_id):
        try:
            job = JobProcess(patient_id, data_dir, args.tag, configs,
                             inference_dir)
            job.wf.run()
            job.clean_up(keep_nifti=True)
        except Exception as e:
            print("Error processing:", e)

    ###################  parallelize inference jobs ###################
    patients = [p.name for p in data_dir.iterdir() if p.is_dir()]
    # splitting jobs accross several CPU
    if args.multiprocess:
        with mp.Pool(32) as p:
            p.map(process_wrapper, patients)
    else:
        for patient in patients:
            process_wrapper(patient)
