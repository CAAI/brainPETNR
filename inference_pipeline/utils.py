import os, sys
from urllib.request import urlopen
from pathlib import Path

"""
Functions called in fsl.Function() are executed in a stand alone env
Imports are to be made within each functions.
"""

folder_with_parameter_files = os.path.join(os.path.expanduser('~'), 'brainPETNR_params')

def infer_from_model(in_file, configs, out_file=None):
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
    print(ckpt_path)
    ckpt = torch.load(ckpt_path)
    # epoch_suffix = f"_e={ckpt['epoch']}"
    model.load_state_dict(ckpt['state_dict'])
    # inference on CPU - or on GPU?
    model.eval().cuda(device=0)

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
            patch_y = model(patch_x.cuda(0)).cpu()
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
    if out_file is None:
        out_dir = in_file.parent
        out_file = out_dir.joinpath(f"inferred_{out_dir.name}.nii.gz")
    else:
        out_dir = Path(out_file).parent
        out_dir.mkdir(exist_ok=True, parents=True)

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

def maybe_download_parameters(force_overwrite=False):
    """
    Downloads the parameters if it is not present yet.
    :param force_overwrite: if True the old parameter file will be deleted (if present) prior to download
    :return:
    """

    if not os.path.isdir(folder_with_parameter_files):
        maybe_mkdir_p(folder_with_parameter_files)

    files = {
        'avg_template.nii.gz': 'avg_template.nii.gz',
        'PE2I.pt': 'PE2I.pt',
        'PE2I.yaml': 'config_PE2I_5pct.yaml',
        'PiB.pt': 'PiB.pt',
        'PiB.yaml': 'config_PiB_5min.yaml'
    }


    for out_file, online_file in files.items():
        out_filename = os.path.join(folder_with_parameter_files, out_file)

        if force_overwrite and os.path.isfile(out_filename):
            os.remove(out_filename)

        if not os.path.isfile(out_filename):
            url = f"https://zenodo.org/record/8063588/files/{online_file}?download=1"
            print("Downloading", url, "...")
            data = urlopen(url).read()
            with open(out_filename, 'wb') as f:
                f.write(data)

def get_template_fname():
    return os.path.join(folder_with_parameter_files, 'avg_template.nii.gz')

def get_config(model):
    return os.path.join(folder_with_parameter_files, f'{model}.yaml')

def get_ckpt_path(model):
    return os.path.join(folder_with_parameter_files, f'{model}.pt')

def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            os.mkdir(os.path.join("/", *splits[:i+1]))