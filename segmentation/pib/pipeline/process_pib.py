
#!/usr/bin/env python2
 
import os
from nipype import Node, Workflow, DataGrabber, MapNode
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.utility import Function, Merge
from nipype.interfaces.freesurfer import ApplyMask
from nipype.interfaces.ants.registration import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.fsl import (Threshold,
                                   BET,
                                   IsotropicSmooth)
from nipype.interfaces.fsl.maths import BinaryMaths, ApplyMask
from nipype.interfaces.niftyseg import LabelFusion
from settings import DATA_DIR, ETC_DIR, THRESHOLDED_REGIONS_DIR, SIGMA, REGION_NAMES
import compute_quantities


def process(data_id):

    wf = Workflow(name='PETCTPIB',
                  base_dir=os.path.join(DATA_DIR, data_id))

    datasource = Node(interface=DataGrabber(infields=['data_id'],
                                            outfields=['struct', "func"]),
                      name='data')
    datasource.inputs.template = "%s/%s"
    datasource.inputs.base_directory = DATA_DIR
    datasource.inputs.data_id = data_id
    datasource.inputs.template_args = dict(struct=[["data_id", 'CT']],
                                           func=[["data_id", 'PET']])
    datasource.inputs.sort_filelist = True

    convert_func = Node(interface=Dcm2niix(),
                        name='dcm_conv_func')

    convert_func.inputs.out_filename = "func"

    convert_struct = Node(interface=Dcm2niix(),
                          name='dcm_conv_struct')

    convert_struct.inputs.out_filename = "struct"

    wf.connect([(datasource,
                 convert_func,
                 [("func", "source_dir")])])

    wf.connect([(datasource,
                 convert_struct,
                 [("struct", "source_dir")])])

    ct_threshold_below_above = Node(interface=Threshold(),
                                    name='ct_threshold')
    ct_threshold_below_above.inputs.thresh = 0
    ct_threshold_below_above.inputs.direction = "below"
    ct_threshold_below_above.inputs.args = " -uthr 100 "

    wf.connect([(convert_struct,
                 ct_threshold_below_above,
                 [("converted_files", "in_file")])
                ])

    ct_smooth = Node(interface=IsotropicSmooth(),
                     name='ct_smooth')
    ct_smooth.inputs.sigma = 1

    wf.connect([(ct_threshold_below_above,
                 ct_smooth,
                 [("out_file", "in_file")])
                ])

    ct_bet = Node(interface=BET(),
                  name='ct_bet')
    ct_bet.inputs.robust = True
    ct_bet.inputs.threshold = True
    ct_bet.inputs.frac = 0.01
    ct_bet.inputs.mask = True

    wf.connect([(ct_smooth,
                 ct_bet,
                 [("out_file", "in_file")])
                ])

    ct_bet_mask_thresh = Node(interface=ApplyMask(),
                              name='ct_bet_mask_thresh')

    wf.connect([(ct_bet, ct_bet_mask_thresh,
                 [("mask_file", "mask_file")]),
                (ct_threshold_below_above, ct_bet_mask_thresh,
                 [("out_file", "in_file")])])

    ct_to_pet = Node(interface=Registration(),
                     name="ct_to_pet")

    ct_to_pet.inputs.metric = ["Mattes"]
    ct_to_pet.inputs.metric_weight = [1]
    ct_to_pet.inputs.smoothing_sigmas = [
        [3.0, 2.0, 1.0, 0.0]]
    ct_to_pet.inputs.shrink_factors = [[8, 4, 2, 1]]
    ct_to_pet.inputs.convergence_window_size = [10]
    ct_to_pet.inputs.transforms = ["Rigid"]
    ct_to_pet.inputs.number_of_iterations = [
        [1000, 500, 50, 100]]

    ct_to_pet.inputs.initial_moving_transform_com = 0
    ct_to_pet.inputs.transform_parameters = [(0.1,)]
    ct_to_pet.inputs.radius_or_number_of_bins = [32]
    ct_to_pet.inputs.num_threads = 6
    ct_to_pet.inputs.winsorize_lower_quantile = 0
    ct_to_pet.inputs.winsorize_upper_quantile = 1
    ct_to_pet.inputs.use_histogram_matching = False
    ct_to_pet.inputs.initialize_transforms_per_stage = False
    ct_to_pet.inputs.write_composite_transform = False
    ct_to_pet.inputs.collapse_output_transforms = True
    ct_to_pet.inputs.verbose = True

    wf.connect([(convert_struct, ct_to_pet,
                 [("converted_files", "moving_image")]),
                (convert_func, ct_to_pet,
                 [("converted_files", "fixed_image")])])

    ct_to_avg_lin = Node(interface=Registration(),
                         name="ct_to_avg_lin")
    ct_to_avg_lin.inputs.output_warped_image = True
    ct_to_avg_lin.inputs.fixed_image = os.path.join(
        ETC_DIR, "avg_template.nii.gz")

    ct_to_avg_lin.inputs.metric = ["Mattes"]
    ct_to_avg_lin.inputs.metric_weight = [1]
    ct_to_avg_lin.inputs.smoothing_sigmas = [
        [3, 2, 1, 0]]
    ct_to_avg_lin.inputs.shrink_factors = [
        [8, 4, 2, 1]]
    ct_to_avg_lin.inputs.convergence_window_size = [10]
    ct_to_avg_lin.inputs.transforms = ["Affine"]
    ct_to_avg_lin.inputs.number_of_iterations = [
        [1000, 500, 250, 100]
    ]
    ct_to_avg_lin.inputs.initial_moving_transform_com = 0
    ct_to_avg_lin.inputs.transform_parameters = [
        (0.1,)]
    ct_to_avg_lin.inputs.radius_or_number_of_bins = [32]
    ct_to_avg_lin.inputs.num_threads = 6
    ct_to_avg_lin.inputs.winsorize_lower_quantile = 0.005
    ct_to_avg_lin.inputs.winsorize_upper_quantile = 0.995
    ct_to_avg_lin.inputs.use_histogram_matching = False
    ct_to_avg_lin.inputs.initialize_transforms_per_stage = False
    ct_to_avg_lin.inputs.write_composite_transform = False
    ct_to_avg_lin.inputs.collapse_output_transforms = True
    ct_to_avg_lin.inputs.verbose = True

    wf.connect([(ct_bet,
                 ct_to_avg_lin,
                 [("out_file", "moving_image")])
                ])

    ct_to_avg_nonlin = Node(interface=Registration(),
                            name="ct_to_avg_nonlin")

    ct_to_avg_nonlin.inputs.fixed_image = os.path.join(
        ETC_DIR, "avg_template.nii.gz")
    ct_to_avg_nonlin.inputs.output_warped_image = True
    ct_to_avg_nonlin.inputs.metric = ["Mattes", "Mattes"]
    ct_to_avg_nonlin.inputs.metric_weight = [1, 1]
    ct_to_avg_nonlin.inputs.smoothing_sigmas = [
        [3, 2, 1, 0], [2, 1, 0]]
    ct_to_avg_nonlin.inputs.shrink_factors = [
        [8, 4, 2, 1], [3, 2, 1]]
    ct_to_avg_nonlin.inputs.convergence_window_size = [10, 10]
    ct_to_avg_nonlin.inputs.transforms = ["Affine", "SyN"]
    ct_to_avg_nonlin.inputs.number_of_iterations = [
        [1000, 500, 250, 100], [100, 50, 30]
    ]
    ct_to_avg_nonlin.inputs.initial_moving_transform_com = 0
    ct_to_avg_nonlin.inputs.transform_parameters = [
        (0.1,), (0.25, 3.0, 0.0)]
    ct_to_avg_nonlin.inputs.radius_or_number_of_bins = [32, 32]
    ct_to_avg_nonlin.inputs.num_threads = 6
    ct_to_avg_nonlin.inputs.winsorize_lower_quantile = 0.005
    ct_to_avg_nonlin.inputs.winsorize_upper_quantile = 0.995
    ct_to_avg_nonlin.inputs.use_histogram_matching = False
    ct_to_avg_nonlin.inputs.initialize_transforms_per_stage = False
    ct_to_avg_nonlin.inputs.write_composite_transform = False
    ct_to_avg_nonlin.inputs.collapse_output_transforms = True
    ct_to_avg_nonlin.inputs.verbose = True

    wf.connect([(ct_bet, ct_to_avg_nonlin,
                 [("out_file", "moving_image")])])

    segment_lab4 = Node(interface=LabelFusion(),
                        name="label4")
    segment_lab4.inputs.classifier_type = "STEPS"
    segment_lab4.inputs.kernel_size = 5
    segment_lab4.inputs.template_num = 8
    segment_lab4.inputs.mrf_value = 0.5
    segment_lab4.inputs.in_file = os.path.join(
        ETC_DIR, "atlas4.nii.gz")
    segment_lab4.inputs.template_file = os.path.join(
        ETC_DIR, "templates.nii.gz")
    wf.connect([(ct_to_avg_nonlin,
                 segment_lab4,
                 [("warped_image", "file_to_seg")])
                ])

    template_to_pet_nonlinxfm = Node(interface=Merge(1),
                                     name="template_to_pet_nonlinxfm")

    wf.connect(ct_to_avg_nonlin, "reverse_transforms",
               template_to_pet_nonlinxfm, "in1")
#    wf.connect(ct_to_pet, "forward_transforms",
#               template_to_pet_nonlinxfm, "in1")

    regions_to_pet = MapNode(interface=ApplyTransforms(),
                             iterfield=["input_image"],
                             name="regions_to_pet")
    regions_to_pet.inputs.interpolation = "NearestNeighbor"
    regions_to_pet.inputs.invert_transform_flags = [
        True, False]  # [False, True, False]
    regions_to_pet.inputs.input_image = [
        os.path.join(THRESHOLDED_REGIONS_DIR,
                     "{}_{}_thresholded.nii.gz".format(region_name.lower(),
                                                       SIGMA))
        for region_name in REGION_NAMES
    ]

    wf.connect(template_to_pet_nonlinxfm, "out",
               regions_to_pet, "transforms")
    wf.connect(convert_func, "converted_files",
               regions_to_pet, "reference_image")

    template_to_pet_linxfm = Node(interface=Merge(1),
                                  name="template_to_pet_linxfm")
    wf.connect(ct_to_avg_lin, "reverse_transforms",
               template_to_pet_linxfm, "in1")
#    wf.connect(ct_to_pet, "forward_transforms",
#               template_to_pet_linxfm, "in1")

    cerebellumgm_to_pet = Node(interface=ApplyTransforms(),
                               name="cerebellumgm_to_pet")
    cerebellumgm_to_pet.inputs.interpolation = "NearestNeighbor"
    cerebellumgm_to_pet.inputs.invert_transform_flags = [True]  # [False, True]
    wf.connect(convert_func, "converted_files",
               cerebellumgm_to_pet, "reference_image")
    wf.connect(template_to_pet_linxfm, "out",
               cerebellumgm_to_pet, "transforms")
    wf.connect(segment_lab4, "out_file",
               cerebellumgm_to_pet, "input_image")

    def switch_two_elements(xfms):
        x, y = xfms
        return [y, x]

    reverse_transforms = Node(interface=Function(function=switch_two_elements,
                                                 input_names=["xfms"],
                                                 output_names=["forward_transforms"]),
                              name="reverse_transforms")
    wf.connect(ct_to_avg_nonlin, "forward_transforms",
               reverse_transforms, "xfms")

    brainmask_to_avg = Node(interface=ApplyTransforms(),
                            name="brainmask_to_avg")
    brainmask_to_avg.inputs.invert_transform_flags = [False, False]
    brainmask_to_avg.inputs.reference_image = os.path.join(
        ETC_DIR, "avg_template.nii.gz")
    brainmask_to_avg.inputs.interpolation = "NearestNeighbor"
    wf.connect(reverse_transforms, "forward_transforms",
               brainmask_to_avg, "transforms")
    wf.connect(ct_bet, "mask_file",
               brainmask_to_avg, "input_image")

    pet_to_avg_xfm = Node(interface=Merge(1),
                          name="pet_to_avg_xfm")
    wf.connect(reverse_transforms, "forward_transforms",
               pet_to_avg_xfm, "in1")
#    wf.connect(ct_to_pet, "reverse_transforms",
#               pet_to_avg_xfm, "in2")

    pet_to_avg = Node(interface=ApplyTransforms(),
                      name="pet_to_avg")
    pet_to_avg.inputs.invert_transform_flags = [
        False, False]  # [False, False, True]
    pet_to_avg.inputs.reference_image = os.path.join(
        ETC_DIR, "avg_template.nii.gz")
    wf.connect(pet_to_avg_xfm, "out",
               pet_to_avg, "transforms")
    wf.connect(convert_func, "converted_files",
               pet_to_avg, "input_image")

    compute_volumes = Node(interface=Function(function=compute_quantities.compute_volumes,
                                              input_names=["region_paths"],
                                              output_names=["volumes"]),
                           name="compute_volumes")
    wf.connect(regions_to_pet, "output_image",
               compute_volumes, "region_paths")

    compute_suvrs = Node(interface=Function(function=compute_quantities.compute_suvrs,
                                            input_names=["region_paths",
                                                         "pet_image_path",
                                                         "reference_region_path"],
                                            output_names=["suvrs"]),
                         name="compute_suvrs")
    wf.connect(regions_to_pet, "output_image",
               compute_suvrs, "region_paths")
    wf.connect(convert_func, "converted_files",
               compute_suvrs, "pet_image_path")
    wf.connect(cerebellumgm_to_pet, "output_image",
               compute_suvrs, "reference_region_path")

    compute_volume_weighted_mean = Node(interface=Function(function=compute_quantities.compute_volume_weighted_mean,
                                                           input_names=["region_paths",
                                                                        "pet_image_path",
                                                                        "reference_region_path"],
                                                           output_names=["volume_weighted_mean"]),
                                        name="compute_volume_weighted_mean")
    wf.connect(regions_to_pet, "output_image",
               compute_volume_weighted_mean, "region_paths")
    wf.connect(convert_func, "converted_files",
               compute_volume_weighted_mean, "pet_image_path")
    wf.connect(cerebellumgm_to_pet, "output_image",
               compute_volume_weighted_mean, "reference_region_path")

    normalization_value = Node(interface=Function(function=(compute_quantities
                                                            .compute_normalizing_activity),
                                                  input_names=["region_path",
                                                               "pet_image_path"],
                                                  output_names=["normalizing_activity"]),
                               name="normalization_value")

    wf.connect(cerebellumgm_to_pet, "output_image",
               normalization_value, "region_path")
    wf.connect(convert_func, "converted_files",
               normalization_value, "pet_image_path")

    pet_to_avg_normalized = Node(interface=BinaryMaths(),
                                 name="pet_to_avg_normalized")
    pet_to_avg_normalized.inputs.operation = "div"
    wf.connect(pet_to_avg, "output_image",
               pet_to_avg_normalized, "in_file")
    wf.connect(normalization_value, "normalizing_activity",
               pet_to_avg_normalized, "operand_value")
    
    pet_to_avg_normalized_brain = Node(interface=ApplyMask(),
                                       name="pet_to_avg_normalized_brain")

    wf.connect(brainmask_to_avg, "output_image",
               pet_to_avg_normalized_brain, "mask_file")

    wf.connect(pet_to_avg_normalized, "out_file",
               pet_to_avg_normalized_brain, "in_file")

    return wf


if __name__ == "__main__":

    patients = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    def process_wrapper(data_id):
        try:
            process(data_id).run()
        except Exception as e:
            print "Error processing: %s" % e

    from multiprocessing import Pool

    p = Pool(32)

    p.map(process_wrapper, patients)
