"""
Module containing function for extracting posterior putamen.
"""


def extract(putamen_filename):
    """
    Extracts posterior part of a given segmentation of putamen.
    It is expected that the file provided is of NIfTI format and
    the putamen segmentation has segmentation ID equal to 2.
    It is assumed that this function is called as part of the pipeline,
    i.e., in a very particular context, therefore it is also checked if the provided
    file also contains a segmentation of caudatus as this is part of the assumed
    context.

    Args:

       putamen_filename (str) : Path to NIfTI file containing putamen and caudatus segmentations.

    Returns:

       str : Path to file storing the extract posterior part of the given putamen segmentation.

    """
    import numpy as np
    import nibabel as nib
    import os
    from pipeline.intensities.auxiliary import get_indices_from_names

    seg = nib.load(putamen_filename)

    unique_indices = np.unique(seg.get_data())
    assert 2 in unique_indices and 3 in unique_indices, "Assuming caudatus is indexed by 3 and putamen by 2"

    put = seg.get_data() == 2

    xo, yo, zo = np.nonzero(put)

    x = xo - np.mean(xo)
    y = yo - np.mean(yo)
    z = zo - np.mean(zo)
    coords = np.vstack([x, y, z])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    xt, yt, zt = np.dot(evecs, coords)

    axis_to_use = np.dot(evecs, coords)[1]
    posterior = axis_to_use > (max(axis_to_use) + min(axis_to_use)) / 2

    coords_low = np.vstack([np.extract(posterior, xo),
                            np.extract(posterior, yo),
                            np.extract(posterior, zo)])

    print coords_low.shape[1] == np.sum(put[tuple(coords_low)] == 1)
    print coords_low.shape[1] == np.sum(put[tuple(coords_low)] == 1)

    put_3d_posterior = np.zeros(seg.shape)
    put_3d_posterior[tuple(coords_low)] = get_indices_from_names(("Putamen",))
    img_posterior = nib.Nifti1Image(put_3d_posterior, seg.affine)

    nib.save(img_posterior,
             os.path.join(
                 os.getcwd(),
                 putamen_filename.split("/")[-1].split(".")[0] + "_putamen_posterior.nii.gz")
             )

    return os.path.join(
        os.getcwd(),
        putamen_filename.split("/")[-1].split(".")[0] + "_putamen_posterior.nii.gz")


if __name__ == "__main__":
    import sys
    print sys.argv[1]

    extract(sys.argv[1])
