import matplotlib as mpl
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
from typing import List
###############################################################################################################
############################################ NORMALIZATIONS ###################################################
###############################################################################################################


def cerebellum_normalization(pet_ref: np.ndarray, mask: np.ndarray) -> float:
    """ Median value of cerebellum activity given a MNI aligned pet images
        and the aligned brain atlas where cerebellum tag is at 67 and 76.

    Args:
        pet_ref (np.ndarray): MNI registered PET image
        mask (np.ndarray): MNI registered brain atlas

    Returns:
        float: cerebellum activity mean value
    """
    pet_12mm = gaussian_filter(pet_ref, 4.2)
    # actual mask of the full cerebellum
    cereb_mask = np.array(mask == 67).astype(int) + np.array(
        mask == 76).astype(int)
    # cerebellum in the patient
    cereb_pet_12mm = pet_12mm * cereb_mask
    # 80% of max value in the cerebellum
    max_cereb_80 = np.max(cereb_pet_12mm) * 0.8
    # mask where cerebellum if > 80% max value
    cereb_mask_final = np.array(cereb_pet_12mm > max_cereb_80).astype(int)
    # apply this mask to reference pet + return mean value
    norm_const = pet_ref[np.where(cereb_mask_final == 1)].median()
    return norm_const


def cerebellum_normalization2(pet_ref: np.ndarray, mask: np.ndarray) -> float:
    """ Median value of cerebellum grey matter activity
        given the cerebellum grey matter mask obtained from Lable Fusion segmentation.

    Args:
        pet_ref (np.ndarray): MNI registered PET image
        mask (np.ndarray): cerebellum grey matter mask

    Returns:
        float: cerebellum activity mean value
    """
    # using Lasse's segmentation of Cerebellum Grey Matter
    return np.median(pet_ref[np.where(mask == 4)])


def putamen_caudate_ratio(pet_data: np.ndarray, pc_mask: np.ndarray) -> float:
    """ Putamen / Caudate nucleus relevant in Parkinson's disease diagnosis. 
        Here used with PE2I tracer.

    Args:
        pet_data (np.ndarray): PET image.
        pc_mask (np.ndarray): putamen caudate mask obtained from Lasse's segmentation model.

    Returns:
        float: PC ratio
    """
    putamen_mask = pet_data[np.where(pc_mask == 2)]
    caudate_mask = pet_data[np.where(pc_mask == 3)]

    return np.median(putamen_mask) / np.median(caudate_mask)


def putamen_sbr(pet_data: np.ndarray, pc_mask: np.ndarray,
                cgm_mask: np.ndarray) -> float:
    """ Putamen specific binding ratio: putamen mean activity 
        over cerebellum grey matter mean activity.

    Args:
        pet_data (np.ndarray): PET image.
        pc_mask (np.ndarray): Putamen caudate mask from Lasse's segmentation model.
        cgm_mask (np.ndarray): Cerebellum grey matter mask from Label Fusion segmentation.

    Returns:
        float: Putamen SBR value
    """
    putamen_mask = pet_data[np.where(pc_mask == 2)]
    cerebellumgm_mask = pet_data[np.where(cgm_mask == 4)]

    return np.median(putamen_mask) / np.median(cerebellumgm_mask)


def quantile_995_normalize(x: np.ndarray) -> np.ndarray:
    """ 99.5% quantile normalization of an array.

    Args:
        x (np.ndarray): PET image.

    Returns:
        np.ndarray: Normalized PET image.
    """
    min_x = np.quantile(x, .005)
    max_x = np.quantile(x, .995)
    return (x - min_x) / (max_x - min_x)


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)


def lr_model(x: np.ndarray, y: np.ndarray) -> tuple:
    """ Linear regression of y=a*x+b

    Args:
        x (np.ndarray): array to fit
        y (np.ndarray): true array

    Returns:
        tuple: (slope, standard error slope, r-squared)
    """
    """ 
        Returns 
    """
    x_fit = sm.add_constant(x)
    results = sm.OLS(y, x_fit).fit()
    output = results.params.true, results.bse.true, results.rsquared
    output = [round(o, 2) for o in output]
    return output


###############################################################################################################
####################################### COLORS FOR 2D SCAN IMAGES #############################################
###############################################################################################################


class _MidPointNorm(mpl.colors.Normalize):
    """
    Class defining normalization of colors to be used when plotting using matplotlib. The class allows one to define the minimum and maximum used in normalization, as usual, while it also allows one to define a "mid-point" -- this allows one to have values between the provided minimum and maximum mapped to something other than 0.5 in the colormap.

    Args:

        vmin (numeric) : Minimum used in the normalization.
        vmax (numeric) : Maximum used in the normalization.
        midpoints (numeric) : midpoints used in the normalization.
        clip (Boolean) : whether to allow clipping in normalization (standard matplotlib argument)
        defmids (list) : What point in the colorscale the provided midpoints should be mapped to (normally, this would be 0.5, but one may "stretch" the colormap using this).

    Attributes:

        vmin (numeric) : Minimum used in the normalization.
        vmax (numeric) : Maximum used in the normalization.
        midpoints (numeric) : midpoints used in the normalization.
        defmids (list) : What point in the colorscale the provided midpoints should be mapped to (normally, this would be 0.5, but one may "stretch" the colormap using this).

    """

    def __init__(self,
                 vmin=None,
                 vmax=None,
                 midpoints=None,
                 clip=False,
                 defmids=[0.5]):
        self.midpoints = midpoints
        self.defmids = defmids
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin] + self.midpoints + \
            [self.vmax], [0] + self.defmids + [1]
        result = np.ma.array(np.interp(value, x, y),
                             mask=result.mask,
                             copy=False)
        return (result[0] if is_scalar else result)


def _lia_img_axial_to_vox(img: np.ndarray, z: int) -> int:
    """ Conversion from slice in PET space to real space.

    Args:
        img (np.ndarray): PET image.
        z (int): slice in PET space.

    Returns:
        int: slice in real space.
    """
    return int(
        nib.affines.apply_affine(np.linalg.inv(img.affine), [0, -z, 0])[2])


###############################################################################################################
################################################ DATAFRAMES ###################################################
###############################################################################################################


def unwrap_dfs(dfs: List[pd.DataFrame],
               keys: List[str],
               full_unwrap=True) -> pd.DataFrame:
    """ Concatenates dataframes and unstacks to one dataframe
        given several dataframes sharing the same index (patients) and columns (attr_1, attr_2, ...)
        each dataframe corresponds to a phase (true, lowdose, denoised),
        which becomes a columns after unstacking.
        

    Args:
        dfs (List[pd.DataFrame]): dataframes for each phase
        keys (List[str]): phase names
        full_unwrap (bool, optional): Whether to unstack or just swap levels. Defaults to True.

    Returns:
        pd.DataFrame: [description]
    """
    df = pd.concat(dfs, keys=keys, axis=1)
    if not full_unwrap:
        df.columns = df.columns.swaplevel(0, 1)
        df.sort_index(axis=1, level=0, inplace=True)
    else:
        df = df.unstack().reset_index()
        df.columns = ['phase', 'metric', 'patient', 'value']
        df.set_index('patient', inplace=True)
    return df