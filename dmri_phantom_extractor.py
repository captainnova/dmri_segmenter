import nibabel as nib
import numpy as np

import dmri_brain_extractor as dbe
import utils


def intensity_mask(img, nsigma=3):
    """
    Calculate a mask by threshholding the intensities of img.

    Parameters
    ----------
    img: 3d array
    nsigma: float
        The threshhold will be min(otsu(img), nsigma * np.std(img[img < otsu]).

    Returns
    -------
    mask: 3d array with dtype np.uint8
        1 where the voxels are "bright enough", 0 elsewhere.
    """
    mask = np.zeros(img.shape, np.bool)
    thresh = dbe.otsu(img)
    sigma = np.std(img[img < thresh])
    thresh = max(thresh, nsigma * sigma)
    mask[img >= thresh] = 1
    return mask.astype(np.uint8)


def make_phantom_mask(img, bvals, dilrad=1, dtype=np.uint8):
    """
    Make a mask for the "interesting" voxels of img.

    Parameters
    ----------
    img: str or nib.Nifti1Image
        A 4D diffusion MRI NIfTI image, or its filename.
    bvals: array
        The diffusion weightings of each volume in img.
    dilrad: float
        How many voxels to close by when filling holes.
        For anisotropic voxels it uses the maximum extent.
    dtype: type
        The data type for the output.

    Returns
    -------
    mask: 3d array with dtype np.uint8
        1 where the voxels are bright enough in either diffusion weighted or diffusion unweighted volumes,
        0 elsewhere.
    """
    if not hasattr(img, 'get_data'):
        dnii = nib.load(img)
    else:
        dnii = img
    data = dnii.get_data()
    s0 = utils.calc_average_s0(data, bvals)
    bcut = utils.bcut_from_rel(bvals)
    dwi = np.mean(data[..., bvals >= bcut], axis=-1)
    mask = intensity_mask(s0)
    dmask = intensity_mask(dwi)
    mask[dmask > 0] = 1
    voxsize = max(utils.voxel_sizes(dnii.affine))
    dilsize = dilrad * voxsize
    mask, errval = utils.fill_holes(mask, dnii.affine, dilsize)

    # Remove fluffy noise from the dilation in fill_holes().
    ball = utils.make_structural_sphere(dnii.affine, dilsize)
    mask = utils.binary_opening(mask, ball)
    
    return mask.astype(dtype)
    
    
    
