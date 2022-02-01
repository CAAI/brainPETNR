from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm

###############################################################################################################
############################################ NORMALIZATIONS ################################################### 
###############################################################################################################

def cerebellum_normalization(pet_ref, mask):
    
    pet_12mm = gaussian_filter(pet_ref, 4.2)
    # actual mask of the full cerebellum
    cereb_mask = np.array(mask == 67).astype(
        int) + np.array(mask == 76).astype(int)
    # cerebellum in the patient
    cereb_pet_12mm = pet_12mm * cereb_mask
    # 80% of max value in the cerebellum
    max_cereb_80 = np.max(cereb_pet_12mm) * 0.8
    # mask where cerebellum if > 80% max value
    cereb_mask_final = np.array(cereb_pet_12mm > max_cereb_80).astype(int)
    # apply this mask to reference pet + return mean value
    norm_const = pet_ref[np.where(cereb_mask_final == 1)].mean()
    return norm_const


def cerebellum_normalization2(pet_ref, mask):
    
    # using Lasse's segmentation of Cerebellum Grey Matter
    return np.median(pet_ref[np.where(mask == 4)])


def putamen_caudate_ratio(pet_data, pc_mask):
    putamen_mask = pet_data[np.where(pc_mask == 2)]
    caudate_mask = pet_data[np.where(pc_mask == 3)]
    
    return np.median(putamen_mask) / np.median(caudate_mask)

def putamen_sbr(pet_data, pc_mask, cgm_mask):
    putamen_mask = pet_data[np.where(pc_mask == 2)]
    cerebellumgm_mask = pet_data[np.where(cgm_mask == 4)]
    
    return np.median(putamen_mask) / np.median(cerebellumgm_mask)

def quantile_995_normalize(x):
    min_x = np.quantile(x, .005)
    max_x = np.quantile(x, .995)
    return (x - min_x) / (max_x - min_x)

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def lr_model(x, y):
    """ Linear regression of y=a*x+b
        Returns (slope, standard error slope, r-squared)
    """
    x_fit = sm.add_constant(x)
    results = sm.OLS(y, x_fit).fit()
    output = results.params.true, results.bse.true, results.rsquared
    output = [round(o, 2) for o in output]
    return output
   
###############################################################################################################
####################################### COLORS FOR 2D SCAN IMAGES ############################################# 
###############################################################################################################

class _MidPointNorm(Normalize):

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

    def __init__(self, vmin=None, vmax=None, midpoints=None, clip=False, defmids=[0.5]):
        self.midpoints = midpoints
        self.defmids = defmids
        self.vmin = vmin
        self.vmax = vmax
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin] + self.midpoints + \
            [self.vmax], [0] + self.defmids + [1]
        result = np.ma.array(
            np.interp(value, x, y), mask=result.mask, copy=False)
        return (result[0] if is_scalar else result)
    
    
def _lia_img_axial_to_vox(img, z):
    return int(nib.affines.apply_affine(np.linalg.inv(img.affine), [0, -z, 0])[2])


###############################################################################################################
################################################ DATAFRAMES ################################################### 
###############################################################################################################

def unwrap_dfs(dfs, keys, full_unwrap=True):
    df = pd.concat(dfs, keys=keys, axis=1)
    if not full_unwrap:
        df.columns = df.columns.swaplevel(0, 1)
        df.sort_index(axis=1, level=0, inplace=True)
    else:
        df = df.unstack().reset_index()
        df.columns = ['phase', 'metric', 'patient', 'value']
        df.set_index('patient', inplace=True)
    return df