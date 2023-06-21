#!/usr/bin/env python3
import os
import argparse
import shutil
import yaml
from pathlib import Path
import multiprocessing as mp
import inference_pipeline.utils as utils
import subprocess as subp
from nipype.interfaces.fsl import Threshold, IsotropicSmooth, RobustFOV, BET, ConvertXFM
import nibabel as nib

def register(file_, reference_file, out_file, dof=6, overwrite=False):
    mat_file = out_file.replace(".nii.gz", ".mat")
    if not os.path.exists(out_file) or overwrite:
        cmd = [
            "flirt",
            "-in",
            file_,
            "-ref",
            reference_file,
            "-out",
            out_file,
            "-dof",
            str(dof),
            "-omat",
            mat_file,
        ]
        output = subp.check_output(cmd)
    else:
        output = "{} already exists and overwrite=False.".format(out_file)
    return mat_file, output

def resample(file_, reference_file, mat_file, out_file, interp=None):
    cmd = [
            "flirt",
            "-in",
            file_,
            "-out",
            out_file,
            "-ref",
            reference_file,
            "-applyxfm",
            "-init",
            mat_file
        ]
    if interp is not None:
        cmd += ['-interp', interp]
    output = subp.check_output(cmd)
    return output

def skullstrip(CT):

    if isinstance(CT, str):
        CT = Path(CT)

    p = CT.parent

    # Smoothing data
    if not p.joinpath('ANAT_smoothed.nii.gz').exists():
        smoothing = IsotropicSmooth()
        smoothing.inputs.in_file = str(CT)
        smoothing.inputs.sigma = 1
        smoothing.inputs.out_file = f"{p}/ANAT_smoothed.nii.gz"
        smoothing.run()

    # Threshold image
    if not p.joinpath('ANAT_smoothed_0-100.nii.gz').exists():
        clamp = Threshold()
        clamp.inputs.in_file = f"{p}/ANAT_smoothed.nii.gz"
        clamp.inputs.thresh = 0.0
        clamp.inputs.direction = 'below'
        clamp.inputs.out_file = f"{p}/ANAT_smoothed_0.nii.gz"
        clamp.run()

        clamp = Threshold()
        clamp.inputs.in_file = f"{p}/ANAT_smoothed_0.nii.gz"
        clamp.inputs.thresh = 100.0
        clamp.inputs.direction = 'above'
        clamp.inputs.out_file = f"{p}/ANAT_smoothed_0-100.nii.gz"
        clamp.run()

    # Crop to HEAD
    if not p.joinpath('ANAT_smoothed_0-100_crop.nii.gz').exists():
        crop = RobustFOV()
        crop.inputs.in_file = f"{p}/ANAT_smoothed_0-100.nii.gz"
        crop.inputs.out_roi = f"{p}/ANAT_smoothed_0-100_crop.nii.gz"
        crop.inputs.out_transform = f"{p}/crop_transform.nii.gz"
        crop.run()

    # Skull Strip
    if not (out_BET := p / CT.name.replace('.nii.gz', '_BET.nii.gz')).exists():
        bet = BET()
        bet.inputs.in_file = f"{p}/ANAT_smoothed_0-100_crop.nii.gz"
        bet.inputs.mask = True
        bet.inputs.frac = 0.1
        bet.inputs.out_file = out_BET
        bet.run()
    
    BET_mask = p / out_BET.name.replace('_BET.nii.gz', '_BET_mask.nii.gz')
    return out_BET, BET_mask

def concat_XFMs(xfm1, xfm2, xfm_out):
    concat = ConvertXFM()
    concat.inputs.concat_xfm = True
    concat.inputs.in_file = xfm1
    concat.inputs.in_file2 = xfm2
    concat.inputs.out_file = xfm_out
    concat.run()

def apply_mask(file_, mask, out_file):
    """print(mask, os.path.exists(mask))
    mask = ApplyMask()
    mask.inputs.in_file = file_
    mask.inputs.mask_file = str(mask)
    mask.inputs.out_file = out_file
    mask.run()"""

    img = nib.load(file_)
    BET = nib.load(mask)
    arr = img.get_fdata() * BET.get_fdata()
    img = nib.Nifti1Image(arr, img.affine, img.header)
    img.to_filename(out_file)    

def preprocess(PET, CT):
    template = utils.get_template_fname()

    # Register PET to CT
    PET_to_CT = PET.replace('.nii.gz', '_to_CT.nii.gz')
    mat_to_ct, _ = register(PET, CT, out_file=PET_to_CT, dof=6)
    # Skullstrip
    CT_BET, BETmask = skullstrip(CT)
    # Align CT_BET to avg
    CT_to_avg = str(CT_BET).replace('.nii.gz', '_to_MNI.nii.gz')
    mat_to_avg, _ = register(CT_BET, template, out_file=CT_to_avg, dof=12)
    # Concat to_ct and to_avg
    concatted_xfms = os.path.join(os.path.dirname(PET), 'reg_to_ct_to_avg.mat')
    concat_XFMs(mat_to_ct, mat_to_avg, concatted_xfms)
    # Resample PET to avg
    PET_to_avg = PET.replace('.nii.gz', '_to_avg.nii.gz')
    resample(PET_to_CT, template, mat_to_avg, PET_to_avg)
    # Resample BETmask to avg
    BETmask_to_avg = str(BETmask).replace('.nii.gz', '_to_avg.nii.gz')
    resample(BETmask, template, mat_to_avg, BETmask_to_avg, interp='nearestneighbour')
    # Apply BETmask to PET in avg space
    PET_avg_BET = PET_to_avg.replace('.nii.gz', '_BET.nii.gz')
    apply_mask(PET_to_avg, BETmask_to_avg, PET_avg_BET)

    return PET_avg_BET, BETmask_to_avg

def to_patient_space(denoised, lowdosePET, BETmask, nativeLowdosePET, out_file):

    # Blur Lowdose
    if not os.path.exists(lowdosePET_blurred := str(lowdosePET).replace('.nii.gz', '_smoothed.nii.gz')):
        smoothing = IsotropicSmooth()
        smoothing.inputs.in_file = str(lowdosePET)
        smoothing.inputs.sigma = 3.9 / 2.3548  # 2mm -> 5mm
        smoothing.inputs.out_file = lowdosePET_blurred
        smoothing.run()

    # Merge lowdose and denoised PET
    if not os.path.exists(denoised_withBG := str(denoised).replace('.nii.gz', '_withBG.nii.gz')):
        img_denoised = nib.load(denoised)
        BET = nib.load(BETmask).get_fdata()
        arr = nib.load(lowdosePET_blurred).get_fdata()
        arr[BET>0] = img_denoised.get_fdata()[BET>0]
        img = nib.Nifti1Image(arr, img_denoised.affine, img_denoised.header)
        img.to_filename(denoised_withBG)    

    # Invert XFM
    if not os.path.exists(xfm_inv := os.path.join(os.path.dirname(lowdosePET), 'reg_to_ct_to_avg_inv.mat')):
        concat = ConvertXFM()
        concat.inputs.invert_xfm = True
        concat.inputs.in_file = os.path.join(os.path.dirname(lowdosePET), 'reg_to_ct_to_avg.mat')
        concat.inputs.out_file = xfm_inv
        concat.run()

    # Go back to patient space
    resample(denoised_withBG, nativeLowdosePET, xfm_inv, out_file)