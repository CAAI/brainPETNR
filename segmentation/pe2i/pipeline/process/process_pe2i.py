#!/usr/bin/env python2
"""
Module containing functions and classes related to the processing of data.
In particular, function for starting processing and class for defining the pipeline.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../.."))
import multiprocessing as mp
from pipeline import settings
from pipeline.process.nn import nn_predict
from pipeline.process.jobs import Job
from pipeline.intensities import extract_putamen_posterior
from nipype.interfaces.niftyseg import LabelFusion
from nipype.interfaces.fsl.maths import MathsCommand
from nipype.interfaces.fsl import (FLIRT,
                                   Threshold,
                                   BET,
                                   IsotropicSmooth,
                                   ConvertXFM)
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.utility import Function
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype import Node, Workflow, DataGrabber

# Semaphore for limiting the amount of jobs running simultaneously.
JOB_SEMAPHORE = mp.Semaphore(6)


class JobProcess(object):

    """
    Encapsulates pipeline behavior. Primarily, this handles setting up the pipeline.

    Args:

       job (Job) : Job object containing information on data to be processed.

    Attributes:

       wf (Workflow) : nipype Workflow describing the pipeline.

    """

    def __init__(self, job):
        self._job = job
        self.wf = Workflow(name="PETCTPE2I",
                           base_dir=self._job.base_dir)

        self._preliminary_setup()

        self._setup_putamen_caudate()

        self._setup_left_right_putamen_caud()

        self._setup_posterior_left_right_putamen()
        

    def _setup_posterior_left_right_putamen(self):
        """
        Creates and connects nodes related to setting up the posterior putamen segmentation (for
        each hemisphere separately).
        """

        self.extract_putamen_posterior_left = Node(interface=Function(
            input_names=["putamen_filename"],
            output_names=["out"],
            function=extract_putamen_posterior.extract),
            name="extract_putamen_posterior_left")

        self.wf.connect(self.left_seg, "out_file",
                        self.extract_putamen_posterior_left, "putamen_filename")

        self.extract_putamen_posterior_right = Node(interface=Function(
            input_names=["putamen_filename"],
            output_names=["out"],
            function=extract_putamen_posterior.extract),
            name="extract_putamen_posterior_right")

        self.wf.connect(self.right_seg, "out_file",
                        self.extract_putamen_posterior_right, "putamen_filename")

    def _setup_left_right_putamen_caud(self):
        """
        Creates and connects nodes related to setting up the putamen and caudatus segmentations for
        each hemisphere separately.
        """

        self.left_seg = Node(
            interface=MathsCommand(),
            name="extract_left_segmentation")

        # Isolates left hemisphere
        self.left_seg.inputs.args = "-roi 128 -1 0 -1 0 -1 0 -1"

        self.wf.connect(self.nn_segmentation, "out_file",
                        self.left_seg, "in_file")

        self.right_seg = Node(
            interface=MathsCommand(),
            name="extract_right_segmentation")

        # Isolates right hemisphere
        self.right_seg.inputs.args = "-roi 0 128 0 -1 0 -1 0 -1"

        self.wf.connect(self.nn_segmentation, "out_file",
                        self.right_seg, "in_file")

    def _setup_putamen_caudate(self):
        """
        Creates and connects nodes related to setting up the putamen and caudatus segmentations for
        *both* hemisphere collectively.
        """


        nn_webservice = Node(interface=Function(function=nn_predict,
                                                input_names=["ct_file",
                                                             "pet_file"],
                                                output_names=["out_file"]),
                             name="nn_webservice")

        self.wf.connect(self.pet_to_avg, "out_file",
                        nn_webservice, "pet_file")

        self.wf.connect(self.ct_to_avg, "out_file",
                        nn_webservice, "ct_file")

        self.nn_segmentation = Node(interface=MRIConvert(),
                                    iterfield=["in_file"],
                                    name="nn_segmentation")
        self.nn_segmentation.inputs.resample_type = "nearest"

        self.wf.connect([(nn_webservice, self.nn_segmentation,
                          [("out_file", "in_file")]),
                         (self.pet_to_avg, self.nn_segmentation,
                          [("out_file", "reslice_like")])
                         ])

    def _preliminary_setup(self):
        """
        Creates and connects nodes later stages of processing will depend on, .e.g,
        specifies folders containing CT resp. PET images, and converts DICOM
        to NIfTI.
        Also, skullstripping is performed, images are spatially normalized, and
        initial segmentations of putmanen, caudatus and cerebellum GM are
        derived.
        """
        self.datasource = Node(interface=DataGrabber(infields=['data_id'],
                                                     outfields=[
                                                         'struct', "func"]),
                               name='data')
        self.datasource.inputs.template = "%s/%s"
        self.datasource.inputs.base_directory = settings.DATA_DIR
        self.datasource.inputs.data_id = self._job.data_id
        self.datasource.inputs.template_args = dict(struct=[["data_id", 'CT']],
                                                    func=[["data_id", 'PET']])
        self.datasource.inputs.sort_filelist = True

        self.convert_func = Node(interface=Dcm2niix(),
                                 name='dcm_conv_func')

        self.convert_func.inputs.out_filename = "func"
        self.convert_func.inputs.merge_imgs = True

        self.convert_struct = Node(interface=Dcm2niix(),
                                   name='dcm_conv_struct')
        self.convert_struct.inputs.out_filename = "struct"
        self.convert_struct.inputs.merge_imgs = True

        self.wf.connect([(self.datasource,
                          self.convert_func,
                          [("func", "source_dir")])])

        self.wf.connect([(self.datasource,
                          self.convert_struct,
                          [("struct", "source_dir")])])

        self.pet_to_ct = Node(interface=FLIRT(),
                              name='pet_to_ct')
        self.pet_to_ct.inputs.cost_func = "corratio"
        self.pet_to_ct.inputs.cost = "corratio"
        self.pet_to_ct.inputs.dof = 6
        self.pet_to_ct.inputs.verbose = True

        self.wf.connect([(self.convert_struct,
                          self.pet_to_ct,
                          [("converted_files", "reference")]),
                         (self.convert_func,
                          self.pet_to_ct,
                          [("converted_files", "in_file")])])

        ct_threshold_below_above = Node(interface=Threshold(),
                                        name='ct_threshold')
        ct_threshold_below_above.inputs.thresh = 0
        ct_threshold_below_above.inputs.direction = "below"
        ct_threshold_below_above.inputs.args = " -uthr 100 "

        self.wf.connect([(self.convert_struct,
                          ct_threshold_below_above,
                          [("converted_files", "in_file")])
                         ])

        ct_smooth = Node(interface=IsotropicSmooth(),
                         name='ct_smooth')
        ct_smooth.inputs.sigma = 1

        self.wf.connect([(ct_threshold_below_above,
                          ct_smooth,
                          [("out_file", "in_file")])
                         ])

        ct_bet = Node(interface=BET(),
                      name='ct_bet')
        ct_bet.inputs.robust = True
        ct_bet.inputs.mask = True

        self.wf.connect([(ct_smooth,
                          ct_bet,
                          [("out_file", "in_file")])
                         ])

        self.ct_to_avg = Node(interface=FLIRT(),
                              name='ct_to_avg')
        self.ct_to_avg.inputs.cost_func = "corratio"
        self.ct_to_avg.inputs.cost = "corratio"
        self.ct_to_avg.inputs.dof = 12
        self.ct_to_avg.inputs.verbose = True
        self.ct_to_avg.inputs.reference = os.path.join(
            settings.ETC_DIR, settings.TEMPLATE)

        self.wf.connect([(ct_bet,
                          self.ct_to_avg,
                          [("out_file", "in_file")])
                         ])

        self.ct_to_avg_nonbet = Node(interface=FLIRT(),
                                     name="ct_to_avg_nonbet")
        self.ct_to_avg_nonbet.inputs.apply_xfm = True
        self.ct_to_avg_nonbet.inputs.reference = os.path.join(
            settings.ETC_DIR, settings.TEMPLATE)
        self.wf.connect([(self.ct_to_avg,
                          self.ct_to_avg_nonbet,
                          [("out_matrix_file", "in_matrix_file")]),
                         (self.convert_struct,
                          self.ct_to_avg_nonbet,
                          [("converted_files", "in_file")])
                         ])

        seg_to_pet_xfm = Node(interface=ConvertXFM(),
                              name="seg_to_pet_xfm")
        seg_to_pet_xfm.inputs.concat_xfm = True
        self.wf.connect([(self.ct_to_avg,
                          seg_to_pet_xfm,
                          [("out_matrix_file", "in_file2")]),
                         (self.pet_to_ct,
                          seg_to_pet_xfm,
                          [("out_matrix_file", "in_file")])
                         ])

        self.pet_to_avg = Node(interface=FLIRT(),
                               name="pet_to_avg")
        self.pet_to_avg.inputs.apply_xfm = True
        self.wf.connect([(seg_to_pet_xfm,
                          self.pet_to_avg,
                          [("out_file", "in_matrix_file")]),
                         (self.ct_to_avg,
                          self.pet_to_avg,
                          [("out_file", "reference")]),
                         (self.convert_func,
                          self.pet_to_avg,
                          [("converted_files", "in_file")])
                         ])

        self.segment_lab4 = Node(interface=LabelFusion(),
                                 name="label4")
        self.segment_lab4.inputs.classifier_type = "STEPS"
        self.segment_lab4.inputs.kernel_size = 5
        self.segment_lab4.inputs.template_num = 8
        self.segment_lab4.inputs.mrf_value = 0.5
        self.segment_lab4.inputs.in_file = os.path.join(
            settings.ETC_DIR, settings.ATLAS4)
        self.segment_lab4.inputs.template_file = os.path.join(
            settings.ETC_DIR, settings.TEMPLATES)
        self.wf.connect([(self.ct_to_avg,
                          self.segment_lab4,
                          [("out_file", "file_to_seg")])
                         ])

    def start(self):
        """
        Method for starting processing.
        """

        self.wf.run()


if __name__ == "__main__":

    patients = [d for d in os.listdir(settings.DATA_DIR) if os.path.isdir(os.path.join(settings.DATA_DIR, d))]
    
    for c, data_id in enumerate(patients):
        print "processing patient {}/{} with id {}".format(c+1, len(patients), data_id)
        job = Job(data_id)
        p = JobProcess(job)
        p.start()
