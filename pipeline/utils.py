"""
Functions called in fsl.Function() are executed in a stand alone env
Imports are to be made within each functions.
"""


def infer_from_model(in_file, configs):
    """ Inference node of the pipeline. Meant to process one patient data at a time.py

    Args:
        in_file (pathlib.Path): input nifti file for the model
        configs (UserConfig): config object of the saved model

    Returns:
        out_file (pathlib.Path): path to inferred nifti file
    """

    import torchio as tio
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from rhtorch.utilities.modules import recursive_find_python_class
    import pathlib

    in_file = pathlib.Path(in_file)
    image = tio.ScalarImage(in_file)

    ####################  NORMALIZE and CROP  #####################
    normalize = tio.Lambda(lambda x: x / configs['pet_normalization_constant'])
    image = normalize(image)
    if 'cropping' in configs:
        crop_config = configs['cropping']
        crop = tio.Crop((*crop_config['x_lim'], *crop_config['y_lim'],
                         *crop_config['z_lim']))
        image = crop(image)

    ####################  load the model  #######################
    data_shape_in = configs['data_shape_in']
    patch_size = configs['patch_size']
    patch_overlap = int(np.min(patch_size) / 2)
    module_name = recursive_find_python_class(configs['module'])
    model = module_name(configs, data_shape_in)
    ckpt_path = pathlib.Path(configs['best_model'])
    ckpt = torch.load(ckpt_path)
    # epoch_suffix = f"_e={ckpt['epoch']}"
    model.load_state_dict(ckpt['state_dict'])
    # inference on CPU - or on GPU?
    model.eval()

    ##################### prepare loader and grid sampler ##########
    patch_size = configs['patch_size']
    patch_overlap = int(np.min(patch_size) / 2)
    # Subject instantiation
    subject = tio.Subject({"input0": image, "target0": image})
    grid_sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=configs['batch_size'])
    aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode='average')
    with torch.no_grad():
        for patches_batch in patch_loader:
            patch_x, _ = model.prepare_batch(patches_batch)
            locations = patches_batch[tio.LOCATION]
            patch_y = model(patch_x)
            aggregator.add_batch(patch_y, locations)
    full_volume = aggregator.get_output_tensor()

    ################### de-normalize and pad ##########################
    full_volume = full_volume * configs['pet_normalization_constant']
    final_image = tio.ScalarImage(tensor=full_volume,
                                  affine=subject.input0.affine)

    if 'cropping' in configs:
        crop_config = configs['cropping']
        pad = tio.Pad((*crop_config['x_lim'], *crop_config['y_lim'],
                       *crop_config['z_lim']))
        final_image = pad(final_image)

    # save nifty with torchio - temporarily in input folder
    out_dir = in_file.parent
    out_file = out_dir.joinpath(f"inferred_{out_dir.name}.nii.gz")
    final_image.save(out_file)
    return out_file


def nifty_to_dicom(in_file, ref_container, out_container, patient_id):
    """ Node taking care of converting the inferred PET image
        from NIFTI format to DICOM.

    Args:
        in_file (pathlib.Path): path to de-noised nifti file.
        ref_container (pathlib.Path): path to dicom folder containing original PET image
        out_container (pathlib.Path): path to output dicom folder containing de-noised PET image
        patient_id (str): patient anonymous id
    """

    import nibabel as nib
    from rhscripts.conversion import to_dcm

    np_nifti = nib.load(in_file).get_fdata()

    to_dcm(np_array=np_nifti,
           dicomcontainer=ref_container,
           dicomfolder=out_container,
           patient_id=patient_id,
           forceRescaleSlope=True,
           from_type='nifty')