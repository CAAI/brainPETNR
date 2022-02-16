#!/usr/bin/env python3
from pathlib import Path
import argparse
# import json
import nibabel as nib
import pandas as pd
import numpy as np
from tqdm import tqdm
# import sys
from rhtorch.utilities.config import UserConfig
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssmi
# from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter
from utils import (cerebellum_normalization2, _lia_img_axial_to_vox,
                   putamen_caudate_ratio, putamen_sbr, unwrap_dfs)
from postprocessing.plotting_utils import (pixelvalue_jointplot_log,
                                           brain_slices_grid_pe2i,
                                           bland_altman_plot, lmplot_compare,
                                           box_roi_percent_diff_images,
                                           boxplot_from_dataframe,
                                           ssmi_psnr_nrmse_plot)


def main():
    parser = argparse.ArgumentParser(
        description='Infer new data from input model.')

    parser.add_argument("-c",
                        "--config",
                        help="Config file for model",
                        type=str,
                        default='')
    parser.add_argument("-im",
                        "--infer-mode",
                        help="Which patient set to use: test or eval.",
                        type=str,
                        default='test')
    parser.add_argument("-t",
                        "--test",
                        help="Only use a subset of the valid dataset.",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    test = args.test
    infer_mode = args.infer_mode
    # always blur lowdose (and possibly highdose if using 2mm)
    # sigma = 2.85/2.3548 ## VALUE IF RES = 128
    sigma1 = 0  #8/2.3548 # lowdose -> highdose
    sigma2 = 0  #2.91/2.3548 # inferred -> highdose

    user_configs = UserConfig(args, mode='infer')
    model_dir = user_configs.rootdir
    configs = user_configs.hparams
    model_name = configs['model_name']
    full_data_shape = configs['data_shape']
    project_dir = Path(configs['project_dir'])

    infer_dir_name = f'inferences_{infer_mode}_set'
    infer_dir = model_dir.joinpath(infer_dir_name)
    if not infer_dir.exists():
        raise FileNotFoundError(
            'Run the inference script before attempting plotting and computing metrics.'
        )

    metrics_dir_name = f'metrics_{infer_mode}_set'
    metrics_dir = model_dir.joinpath(metrics_dir_name)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # other folder of interest
    dicom_dir = project_dir.joinpath("data_dicom")
    recon_base_dir = project_dir.joinpath("data_registered")

    # CSV patient info file
    patient_info = dicom_dir.joinpath('PE2I_anonymized_patient_info.csv')
    df = pd.read_csv(patient_info, index_col=0)
    scaling_info = dicom_dir.joinpath('PE2I_scaling_factors.csv')
    dfs = pd.read_csv(scaling_info, index_col=0)

    # slices and axis setup
    plot_axis = 1  # top view for brain scan
    # project = 'pe2i'
    plot_slices = np.arange(-22, 21, 6)
    plot_vmin = 0
    plot_vmax = 3.5

    # metrics and such to fill in for each patient
    clinicals_true, clinicals_ld, clinicals_infer = [], [], []
    metrics_ld, metrics_infer = [], []

    patients = [p.name for p in infer_dir.iterdir() if p.is_dir()]
    if test:
        patients = patients[:3]

    # for stacking all pixel values together for master jointplot
    ld_pixel = np.zeros((len(patients), *full_data_shape))
    true_pixel = np.zeros((len(patients), *full_data_shape))
    infer_pixel = np.zeros((len(patients), *full_data_shape))

    # for jointplot histogram
    nbins = 200  #Number of bins along each axis.
    hist_ld = np.zeros((nbins, nbins))
    hist_infer = np.zeros((nbins, nbins))
    m = 10000  # max activity for figure
    int_range = np.linspace(0, m, nbins)
    intervals = list(zip(int_range[:-1], int_range[1:]))

    for i, patient in tqdm(enumerate(patients)):

        # load the data and stack it in a master array along axis=3
        dat_ld_file = configs['input_files']['name'][0]
        dat_ld_name = "_".join(dat_ld_file.split("_")[:3])
        blurred_dat_ld = dat_ld_name.replace('2mm', '5mm')
        dat_name = recon_base_dir.joinpath(patient, blurred_dat_ld,
                                           'pet_to_avg_bet',
                                           'pet_to_avg_bet.nii.gz')
        dat = nib.load(dat_name).get_fdata()

        dat_true_name = recon_base_dir.joinpath(patient, 'PET_5mm',
                                                'pet_to_avg_bet',
                                                'pet_to_avg_bet.nii.gz')
        dat_true = nib.load(dat_true_name).get_fdata()

        dat_infer_name = infer_dir.joinpath(patient).joinpath(
            f'Inferred_{model_name}.nii.gz')
        dat_infer = nib.load(dat_infer_name).get_fdata()

        if isinstance(plot_slices, list) or isinstance(plot_slices,
                                                       np.ndarray):
            # transform from real space to MNI space
            pet = nib.load(dat_name)
            final_slices = [_lia_img_axial_to_vox(pet, s) for s in plot_slices]

        # load segmentation image (PE2I)
        put_caud_mask = np.zeros((4, *dat.shape))
        for j, dname in enumerate([
                f'PETCTPE2I_LD10pct_segmentation', 'PETCTPE2I_segmentation',
                'PETCTPE2I_segmentation'
        ]):
            put_caud_mask_file = dicom_dir.joinpath(patient, dname,
                                                    "nn_segmentation",
                                                    'prediction_out.nii.gz')
            pc_mask = nib.load(put_caud_mask_file).get_fdata()
            put_caud_mask[j] = pc_mask

        # mean in cerebellum GM for normalization
        cgm_file = dicom_dir.joinpath(patient, 'PETCTPE2I_segmentation',
                                      "label4", "atlas4_steps.nii.gz")
        cgm_file_mask = nib.load(cgm_file).get_fdata()
        put_caud_mask[3] = cgm_file_mask

        # normalization value
        cereb_const = cerebellum_normalization2(dat_true, cgm_file_mask)

        # load activity info from csv file
        activity_reduction = float(dfs.loc[dat_ld_name, 'activity'])
        fulldose_value = df.loc[patient, 'Activity (MBq)']
        if 'pct' in dat_ld_name:
            lowdose_value = fulldose_value * activity_reduction
            doses = [lowdose_value, fulldose_value]
        elif 'sec' in dat_ld_name:
            base_time = 600
            low_activity_time = int(base_time * activity_reduction)
            doses = [f"{low_activity_time}sec", f'{base_time}sec']

        # take a mask before blurring
        mask_file_path = recon_base_dir.joinpath(patient, dat_ld_name,
                                                 "mask_to_avg",
                                                 "mask_to_avg.nii.gz")
        mask_file = nib.load(mask_file_path).get_fdata()
        mask_true = np.where(mask_file == 1)
        dat_infer = dat_infer * mask_file

        # blur if needed
        # if sigma1:
        #     dat = gaussian_filter(dat, sigma1) * mask_file
        if sigma2:
            dat_infer = gaussian_filter(dat_infer, sigma2) * mask_file

        # contour box for figures of merit
        dat_box = dat  # [mid-extent_box:mid+extent_box, mid-extent_box:mid+extent_box, mid-extent_box:mid+extent_box]
        dat_true_box = dat_true  # [mid-extent_box:mid+extent_box, mid-extent_box:mid+extent_box, mid-extent_box:mid+extent_box]
        dat_infer_box = dat_infer  # [mid-extent_box:mid+extent_box, mid-extent_box:mid+extent_box, mid-extent_box:mid+extent_box]

        # concatenate data in one master array
        d_arr = np.stack((dat, dat_true, dat_infer), axis=0) / cereb_const

        # stacking all patients in master array for later jointplot - downsampled arrays
        ld_pixel[i] = dat_box
        true_pixel[i] = dat_true_box
        infer_pixel[i] = dat_infer_box

        ##############################################################################################################
        ###################################  METRICS EXTRACTION BASED ON MASKS #######################################
        ##############################################################################################################

        rois = {'Putamen': 2, 'Caudate Nucleus': 3}
        data = {
            'lowdose': (dat, put_caud_mask[0]),
            'true': (dat_true, put_caud_mask[1]),
            'infer': (dat_infer, put_caud_mask[2])
        }

        # CLINICAL METRICS
        clinicals_per_patient = {}
        for ph, (ph_arr, mask_arr) in data.items():
            clinicals_per_phase = {}
            # ROIS
            for k, v in rois.items():
                clinicals_per_phase[k + ' uptake'] = ph_arr[np.where(
                    mask_arr == v)].mean()
            # Full brain uptake
            clinicals_per_phase['Full brain uptake'] = ph_arr[mask_true].mean()
            # Putamen Caudate Nucleus Ratio
            clinicals_per_phase['PC ratio'] = putamen_caudate_ratio(
                ph_arr, mask_arr)
            # Putamen SBR
            clinicals_per_phase['Putamen SBR'] = putamen_sbr(
                ph_arr, mask_arr, cgm_file_mask)
            # add phase to the patient clinical metrics
            clinicals_per_patient[ph] = clinicals_per_phase

        # NOISE METRICS
        data = {'lowdose': dat_box, 'infer': dat_infer_box}
        data_range = np.max(dat_true_box) - np.min(dat_true_box)
        metrics_per_patient = {}
        for ph, ph_arr in data.items():
            metrics_per_phase = {}
            # taking from the box ROI here but might not be good for PiB
            metrics_per_phase['ssmi'] = ssmi(dat_true_box, ph_arr)
            print(patient, ph, metrics_per_phase['ssmi'])
            metrics_per_phase['psnr'] = psnr(dat_true_box,
                                             ph_arr,
                                             data_range=data_range)
            metrics_per_phase['nrmse'] = nrmse(dat_true_box, ph_arr)
            metrics_per_patient[ph] = metrics_per_phase

        # master clinical metrics lists
        clinicals_true.append(clinicals_per_patient['true'])
        clinicals_ld.append(clinicals_per_patient['lowdose'])
        clinicals_infer.append(clinicals_per_patient['infer'])

        # master noise metrics lists
        metrics_ld.append(metrics_per_patient['lowdose'])
        metrics_infer.append(metrics_per_patient['infer'])

        ########################################################################################################
        ########################################  PLOTTING PER PATIENT #########################################
        ########################################################################################################

        pc_ratio_per_patient = {
            'true': clinicals_per_patient['true']['PC ratio'],
            'lowdose': clinicals_per_patient['lowdose']['PC ratio'],
            'infer': clinicals_per_patient['infer']['PC ratio']
        }

        # BRAIN SCLICE GRID - 3 COLUMNS
        for k in range(int(len(final_slices) / 3)):
            sl = final_slices[3 * k:3 * (k + 1)]
            figname = infer_dir.joinpath(patient).joinpath(
                f"Top_view_slices_{k}.png")
            # if not figname.exists():
            brain_slices_grid_pe2i(d_arr,
                                   patient,
                                   figname,
                                   put_caud_mask,
                                   slices=sl,
                                   axis=plot_axis,
                                   v_scale=(plot_vmin, plot_vmax),
                                   doses=doses,
                                   metrics=pc_ratio_per_patient)

        # JOINTPLOT HISTOGRAM
        mask = mask_file > 0.5
        #Compute histogram
        for i1, (l1, u1) in enumerate(intervals):
            # _f: vector of voxels from sbPET, for which the PET activity is in bin i1, that is (l1 < PET < u1).
            _f = dat[(dat_true > l1) & (dat_true < u1) & mask]
            _fi = dat_infer[(dat_true > l1) & (dat_true < u1) & mask]
            for i2, (l2, u2) in enumerate(intervals):
                # Sort _f in bins depending ont the activity of sbPET and count the number of voxels.
                hist_ld[i1, i2] += ((_f > l2) & (_f < u2)).sum()
                hist_infer[i1, i2] += ((_fi > l2) & (_fi < u2)).sum()

        ## HISTOGRAM OF PIXEL VALUES
        # figname0 = infer_dir.joinpath(patient).joinpath('hist_comp.pdf')
        # if not figname0.exists():
        #     # pixelvalue_hist_kde(dat, dat_true, dat_infer, mask_true, figname0)
        #     pixelvalue_jointplot(dat, dat_true, dat_infer, figname0)

        ## JOINTPLOT PIXEL VALUE TRUE <-> LOWDOSE & TRUE <-> INFERRED
        # figname00 = infer_dir.joinpath(patient).joinpath('joint_plot_true_vs_infer.png')
        # if not figname00.exists():
        #     pixelvalue_jointplot(dat_true, dat_infer, figname00, 'Pixel value (inferred)', color=clr['ld'])
        # figname00b = infer_dir.joinpath(patient).joinpath('joint_plot_true_vs_lowdose.png')
        # if not figname00b.exists():
        #     pixelvalue_jointplot(dat_true, dat, figname00b, 'Pixel value (lowdose)', color=clr['infer'])

        ## DIFF 2D PLOTS LOWDOSE-TRUE & INFERRED-TRUE
        # figname000 = infer_dir.joinpath(patient).joinpath('percent_diff_plots.png')
        # if not figname000.exists():
        #     box_roi_percent_diff_images(dat, dat_true, dat_infer, figname000, extent=45)

    ###########################################################################################################
    ########################################   END OF PATIENT LOOP   ##########################################
    ######################################   COMPUTE GLOBAL METRICS   #########################################
    ###########################################################################################################

    # save dataframes as csv
    df_clinicals_true = pd.DataFrame(clinicals_true, index=patients)
    df_clinicals_true.to_csv(metrics_dir.joinpath("clinical_metrics_true.csv"))
    df_clinicals_ld = pd.DataFrame(clinicals_ld, index=patients)
    df_clinicals_ld.to_csv(
        metrics_dir.joinpath("clinical_metrics_lowdose.csv"))
    df_clinicals_infer = pd.DataFrame(clinicals_infer, index=patients)
    df_clinicals_infer.to_csv(
        metrics_dir.joinpath("clinical_metrics_infer.csv"))

    df_metrics_ld = pd.DataFrame(metrics_ld, index=patients)
    df_metrics_ld.to_csv(metrics_dir.joinpath("noise_metrics_lowdose.csv"))
    df_metrics_infer = pd.DataFrame(metrics_infer, index=patients)
    df_metrics_infer.to_csv(metrics_dir.joinpath("noise_metrics_infer.csv"))

    # Percent diff - does this need saving ?
    df_pdiff_ld = (df_clinicals_ld -
                   df_clinicals_true) / df_clinicals_true * 100
    df_pdiff_infer = (df_clinicals_infer -
                      df_clinicals_true) / df_clinicals_true * 100
    dfu_pdiff = unwrap_dfs([df_pdiff_ld, df_pdiff_infer],
                           keys=['lowdose - true', 'AI - true'])

    # % DIFF OF MEAN BRAIN AREAS - DATAFRAME and BOXPLOT
    boxplot_from_dataframe(dfu_pdiff, metrics_dir, "Percent_diff_brain_map")

    # SSMI, PSNR and NRMSE metrics
    df_metrics = unwrap_dfs([df_metrics_ld, df_metrics_infer],
                            keys=['lowdose - true', 'AI - true'])
    ssmi_psnr_nrmse_plot(df_metrics, metrics_dir, "ssmi_psnr_rmse")

    # PUTAMEN-CAUTATUS ratio
    df_full = unwrap_dfs(
        [df_clinicals_true, df_clinicals_ld, df_clinicals_infer],
        keys=['true', 'lowdose', 'infer'],
        full_unwrap=False)
    bland_altman_plot(df_full['PC ratio'], metrics_dir,
                      'PC_ratio')  #, xlim=[0.55, 1.35], ylim=[-0.35, 0.35])
    lmplot_compare(df_full['PC ratio'], metrics_dir, "PC_ratio_lr",
                   thresh=0.9)  #, xlim=[0.55, 1.35], ylim=[0.55,1.35])

    # PUTAMEN SBR
    bland_altman_plot(df_full['Putamen SBR'], metrics_dir, 'Putamen_SBR')
    lmplot_compare(df_full['Putamen SBR'],
                   metrics_dir,
                   "Putamen_SBR_lr",
                   thresh=3.5)

    # MASSIVE JOINTPLOT ALL DATA
    figname09 = metrics_dir.joinpath('hist_comp_all.pdf')
    pixelvalue_jointplot_log(ld_pixel, true_pixel, infer_pixel, hist_ld,
                             hist_infer, figname09)


if __name__ == '__main__':
    main()
